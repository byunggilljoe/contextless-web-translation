"""Create an attachment ZIP and SHA256SUMS. Does not publish or use git add.

Run after all edits and validation. Outputs and caches are deliberately excluded.
"""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def files():
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(p in ('outputs', '.venv', '__pycache__', '.git', '.pytest_cache') for p in relative.parts):
            continue
        if path.name.startswith('.env') and path.name != '.env.example':
            continue
        if path.suffix in ('.pyc', '.zip') or path.name == 'SHA256SUMS':
            continue
        if path.is_symlink():
            raise ValueError(f'Bundle must not contain symlinks: {relative}')
        if path.is_file():
            yield path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / (ROOT.name + '.zip'))
    args = parser.parse_args()
    paths = list(files())
    if args.output.resolve().is_relative_to(ROOT):
        parser.error('Put the ZIP outside the bundle directory')
    if args.output.exists():
        parser.error('ZIP already exists; choose another name or remove your previous generated ZIP')
    manifest = ROOT / 'SHA256SUMS'
    manifest.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' +
                               p.relative_to(ROOT).as_posix() + '\n' for p in paths), encoding='utf-8')
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths + [manifest]:
            archive.write(path, f'{ROOT.name}/{path.relative_to(ROOT).as_posix()}')
    print(f'{args.output.name}: {args.output.stat().st_size:,} bytes, {len(paths) + 1} files')
