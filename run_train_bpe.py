"""Runner for Problem `train_bpe_tinystories`.

Trains a byte-level BPE tokenizer on a corpus, captures time + peak memory,
serializes `vocab` and `merges` to disk, and reports the longest token.

Run from this directory (the submodule root) so `cs336_basics` is on the
`uv`-managed path and `data/` resolves correctly:

    cd cs336/assignments/assignment1/assignment1-basics

    # Debug on the small validation file first.
    uv run python run_train_bpe.py \
        --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 1000

    # Full TinyStories training run, target vocab 10k.
    uv run python run_train_bpe.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 \
        --out-prefix tokenizer/tinystories

To collect a cProfile trace alongside the run:

    uv run python -m cProfile -o train_bpe.prof run_train_bpe.py --input ...
    uv run python -c "import pstats; pstats.Stats('train_bpe.prof').sort_stats('cumulative').print_stats(30)"
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
import tracemalloc
from pathlib import Path

from cs336_basics.bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--out-prefix", type=Path, default=Path("tokenizer/tinystories"))
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to add to the vocab and use as document boundaries.",
    )
    args = parser.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    t0 = time.perf_counter()
    tokenizer = BPETokenizer.train(
        input_path=str(args.input),
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )
    vocab, merges = tokenizer.vocab, tokenizer.merges
    elapsed = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    vocab_path = args.out_prefix.with_suffix(".vocab.json")
    merges_path = args.out_prefix.with_suffix(".merges.pkl")
    # bytes <-> latin-1 is a 1:1 mapping for all byte values, so json round-trips cleanly.
    vocab_serializable = {str(i): b.decode("latin1") for i, b in vocab.items()}
    vocab_path.write_text(json.dumps(vocab_serializable, ensure_ascii=False))
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)

    longest = max(vocab.values(), key=len)
    print(f"input          : {args.input}")
    print(f"vocab size     : {len(vocab)}  (merges: {len(merges)})")
    print(f"wall time      : {elapsed:.2f} s")
    print(f"python heap pk : {py_peak / 1024**2:.1f} MiB  (tracemalloc, parent only)")
    print(f"process RSS pk : {rss_peak_kib / 1024:.1f} MiB  (getrusage, parent only)")
    print(f"longest token  : {len(longest)} bytes  raw={longest!r}")
    print(f"  decoded      : {longest.decode('utf-8', errors='replace')!r}")
    print(f"saved          : {vocab_path}  {merges_path}")


if __name__ == "__main__":
    main()
