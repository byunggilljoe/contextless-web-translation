TRANSLATE_TAG = "TRAN" # T

generate_args = {
    "temperature": 0.01,
    "max_new_tokens": 32,
}

# EN->KO visible-text token expansion ratio r, measured on the en2ko survey
# dataset (38/73 structure-matched pages, Qwen2.5-Coder-32B reference translations).
# Used to size the Seq2Seq required-output / max_new_tokens cap:
#   required_output_tokens = markup_tokens + text_tokens * r
TOKEN_EXPANSION_RATIO = {
    "qwen": 1.36,   # Qwen2.5 tokenizer (Qwen2Tokenizer)
    "llama": 1.28,  # Llama-3.1 tokenizer
}

# whole-cleansed-HTML output/input token ratio, for sanity cross-check only.
FULL_HTML_OUTPUT_RATIO = {
    "qwen": 1.20,
    "llama": 1.15,
}