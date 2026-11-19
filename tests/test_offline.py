"""Exercise real CLIs against a local protocol stub; no external model is called."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import read_jsonl


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        assert self.path == '/v1/chat/completions'
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if payload['model'] == 'denied':
            self.send_response(401)
            self.end_headers()
            return
        text = payload['messages'][-1]['content']
        response = ('score: 4' if '== Evaluation Criteria ==' in text else
                    '안녕하세요 <script>번역</script>' if '<T_(NUMBER)>' in text else
                    '<TRAN_0>번역</TRAN_0>')
        if payload['model'] == 'malformed':
            response = 'No tagged translation'
        body = json.dumps({'choices': [{'message': {'content': response},
                           'finish_reason': 'length' if payload['model'] == 'truncated' else 'stop'}],
                           'usage': {'prompt_tokens': 10, 'completion_tokens': 3}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def cli(self, *args, script='run.py', expected=0):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=self.directory, capture_output=True, text=True,
                                env={**os.environ, 'OPENAI_API_KEY': 'local-test-only'})
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def translate(self, output, model='stub', *extra):
        return self.cli('translate', '--pair', 'en2ko', '--limit', 2, '--output', output,
                        '--model', model, '--base-url', self.endpoint, *extra)

    def test_translation_and_judging(self):
        predictions = self.directory / 'predictions.jsonl'
        self.translate(predictions)
        rows = read_jsonl(predictions)
        self.assertEqual([r['translation'] for r in rows], ['번역', '번역'])
        self.assertEqual([r['status'] for r in rows], ['ok', 'ok'])
        scores = self.directory / 'scores.jsonl'
        self.cli('score', '--pair', 'en2ko', '--limit', 2, '--predictions', predictions,
                 '--output', scores, '--model', 'stub', '--base-url', self.endpoint)
        self.assertEqual([r['score'] for r in read_jsonl(scores)], [4, 4])
        metadata = json.loads(Path(str(scores) + '.meta.json').read_text())
        self.assertEqual(metadata['summary']['mean_valid'], 4)
        self.assertEqual(metadata['summary']['coverage'], 1)

    def test_error_and_truncation_are_explicit(self):
        for model, status in [('denied', 'backend_error'), ('truncated', 'truncated'),
                              ('malformed', 'parse_error')]:
            output = self.directory / (model + '.jsonl')
            self.cli('translate', '--pair', 'en2ko', '--limit', 1, '--output', output,
                     '--model', model, '--base-url', self.endpoint, expected=1)
            self.assertEqual(read_jsonl(output)[0]['status'], status)

    def test_missing_reference_remains_in_score_denominator(self):
        output = self.directory / 'missing.jsonl'
        self.cli('score', '--pair', 'en2ko', '--sample-ids', 'en2ko:67', '--candidate',
                 'reference', '--dry-run', '--output', output, expected=1)
        rows = read_jsonl(output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'missing_translation')

    def test_full_page_selection_preserves_dom_and_escapes_model_markup(self):
        source = '<html><head><script>keep()</script></head><body><p class="keep">Hello</p><p>World</p></body></html>'
        html = self.directory / 'source.html'
        html.write_text(source)
        output = self.directory / 'result.html'
        self.cli('--input', html, '--output', output, '--ids', 0, '--model', 'stub',
                 '--base-url', self.endpoint, script='translate_page.py')
        result = BeautifulSoup(output.read_text(), 'html.parser')
        before = BeautifulSoup(source, 'html.parser')
        self.assertEqual([(t.name, t.attrs) for t in before.find_all()],
                         [(t.name, t.attrs) for t in result.find_all()])
        self.assertEqual(result.find_all('p')[0].get_text(), '안녕하세요 <script>번역</script>')
        self.assertEqual(result.find_all('p')[1].get_text(), 'World')
        self.assertEqual(result.script.string, 'keep()')

    def test_dry_run_and_no_overwrite(self):
        output = self.directory / 'prompts.jsonl'
        self.cli('translate', '--pair', 'en2ko', '--limit', 1, '--dry-run', '--output', output)
        self.assertEqual(read_jsonl(output)[0]['status'], 'dry_run')
        self.cli('translate', '--pair', 'en2ko', '--limit', 1, '--dry-run', '--output', output, expected=2)

    def test_contextless_prompt_excludes_html_and_reference_translations(self):
        output = self.directory / 'contextless.jsonl'
        self.cli('prepare', '--pair', 'en2ko', '--limit', 1, '--mode', 'contextless',
                 '--output', output)
        row = read_jsonl(output)[0]
        sample = read_jsonl(ROOT / 'data/samples.jsonl')[0]
        prompt = '\n'.join(message['content'] for message in row['messages'])
        self.assertEqual(row['status'], 'prepared')
        self.assertIn(sample['original'], prompt)
        self.assertNotIn('<html', prompt.lower())
        self.assertNotIn(sample['chrome_translation'], prompt)
        self.assertNotIn(sample['reference_translation'], prompt)


if __name__ == '__main__':
    unittest.main()
