# Assignment: BPE Tokenizer Training  
**CS 639: Deep Learning for NLP**

This assignment focuses on implementing a **byte-level Byte-Pair Encoding (BPE) tokenizer**, a core component in modern NLP systems. You will complete missing parts of a tokenizer training pipeline and train a tokenizer on real text data.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Download data
Download the TinyStories data

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```







## Files

### `tokenizer_hw.py` (main file)

You must implement the following functions:

- `build_split_expr()` — regex split that preserves special tokens  
- `pretokenize_text()` — regex pre-tokenization + byte conversion  
- `process_chunk()` — chunk-based preprocessing  
- `count_pairs()` — count adjacent token pairs  
- `merge_pair()` — apply BPE merge updates  

**Do not change function signatures.**

---

## Running Training

Train the tokenizer with:

```bash
uv run tokenizer_hw.py
```



Outputs will be saved to:

```
tokenizer_results/
  *_vocab.pkl
  *_merges.pkl
```

Training should complete within a few minutes on CPU.



## Allowed Libraries

**Allowed:**
- Python standard library  
- `regex`  
- `numpy` (optional)

**Not allowed:**
- HuggingFace tokenizers  
- SentencePiece  
- Any external tokenizer implementations  

You must implement BPE yourself.

---

## Submission Format

Submit a zip file structured as:

```text
CAMPUSID/
  tokenizer_hw.py
  tokenizer_results/
    *_vocab.pkl
    *_merges.pkl
  run.sh


## Grading Overview

Score | Criteria
---|---
90–100 | Correct implementation; tokenizer trains successfully
85–89 | Minor correctness or performance issues
80–84 | Some missing functionality
<80 | Major missing parts or code does not run


