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






## Implementation Details

### Two-Stage Approach

#### 1. `tokenizer_v2.py` - Prototype Implementation
- **Target**: Small to medium datasets (< 3GB)
- **Strategy**: Speed-optimized with full in-memory processing
- **Performance on TinyStories**: 66.94s total (17.7s pre-tokenization, 48.8s merging)
- **Memory usage**: ~27.8GB peak with 40 processes
- **Best for**: Development, testing, and smaller datasets

#### 2. `tokenizer_v2_optimized.py` - Production Implementation
- **Target**: Large datasets (> 3GB, tested on 11.1GB OpenWebText)
- **Strategy**: Memory-efficient with streaming processing
- **Key optimizations**:
  - Streaming pre-tokenization with batch processing
  - Adaptive memory thresholds (200K sequences max)
  - Automatic cleanup of low-frequency sequences
  - Checkpoint/resume functionality
- **Performance on OpenWebText**: 6+ hours (time-space tradeoff)
- **Memory usage**: 30-50GB peak (within cluster limits)

### Memory Scaling Analysis

Dataset size directly impacts memory requirements:

| Dataset | Size | Memory Peak | Processing Time | Implementation |
|---------|------|-------------|----------------|----------------|
| TinyStories | 2.07GB | ~27.8GB | 66.94s | tokenizer_v2.py |
| OpenWebText | 11.1GB | 30-50GB* | 6+ hours | tokenizer_v2_optimized.py |

*Without optimization, OpenWebText would require ~180-220GB memory

### Key Features

#### Adaptive Processing
```python
FAST_MODE_THRESHOLD = 3 * 1024 * 1024 * 1024  # 3GB threshold
is_small_file = file_size < FAST_MODE_THRESHOLD
```

#### Memory Management
- `MAX_MEMORY_SEQUENCES = 200000`: Limits in-memory sequences
- `MEMORY_CLEANUP_THRESHOLD = 300000`: Triggers automatic cleanup
- Preserves high-frequency sequences, removes low-frequency ones

#### Streaming Pre-tokenization
```python
def pre_tokenize_streaming(file_path, chunk_size=1000):
    for batch in process_batches():
        yield batch_result
        del batch_result; gc.collect()  # Immediate cleanup
```

#### Checkpoint System
- Periodic saves every N merges
- Resume training from interruption
- State includes vocabulary, merges, and progress

## Usage

### Quick Start

For small datasets (< 3GB):
```sh
uv run cs336_basics/tokenizer_v2.py
```

For large datasets (> 3GB):
```sh
uv run cs336_basics/tokenizer_v2_optimized.py
```

### Configuration

Key parameters for memory optimization:
- `MAX_MEMORY_SEQUENCES`: Maximum sequences in memory
- `CHECKPOINT_FREQUENCY`: How often to save state
- `MEMORY_CLEANUP_THRESHOLD`: When to trigger cleanup

## Results and Analysis

### Tokenizer Quality
Both implementations produce high-quality tokenizers. Example long tokens from OpenWebText training:
- `b' disproportionately'` (19 bytes)
- `b' telecommunications'` (19 bytes)

These demonstrate proper learning of:
- Word boundary patterns (leading spaces)
- High-frequency complete words as single tokens
- Efficient vocabulary utilization

### Performance Insights
1. **Memory is the primary constraint** for large-scale BPE training
2. **Streaming processing** enables training on memory-constrained systems
3. **Frequency-based cleanup** minimally impacts tokenizer quality
4. **Checkpointing** is essential for long-running jobs

### Engineering Trade-offs
- **Accuracy vs. Memory**: Minor quality impact from low-frequency sequence cleanup
- **Time vs. Space**: 6+ hours vs. 180GB+ memory savings
- **Complexity vs. Scalability**: Added engineering complexity enables real-world deployment

## Technical Details

### Memory Optimization Strategies
1. **Streaming Processing**: Process file in chunks, immediate cleanup
2. **Adaptive Thresholds**: Different strategies based on dataset size
3. **Frequency-based Pruning**: Retain high-value sequences
4. **Checkpoint Recovery**: Resume from interruptions

### System Requirements
- **Minimum**: 32GB RAM for small datasets
- **Recommended**: 64GB+ RAM for large datasets
- **Tested on**: 125GB RAM, 8GB swap cluster environment

This implementation demonstrates that with proper engineering, BPE tokenizer training can scale to large datasets even with memory constraints, making it practical for real-world NLP applications.

