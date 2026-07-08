"""Build Stage-1 pretraining data: contiguous token blocks from BooksCorpus
(.txt) and Wikipedia (.parquet).  Matches the original two-stage recipe of
robert_retrain_step1.py, which used both corpora.

Produces {input_ids: (N,BLOCK) int32, special_tokens_mask: (N,BLOCK) int8}
pickled dict compatible with scripts/train.py.
"""

import sys, pickle, glob, numpy as np
from transformers import RobertaTokenizerFast

MP = "/root/src/pretrain_model/roberta-base"             # adjust for your setup
BOOKS_SRCS = [
    "/root/autodl-tmp/RoBERTa/books_large_p1.txt",
    "/root/autodl-tmp/RoBERTa/books_large_p2.txt",
]
WIKI_DIR = "/root/autodl-tmp/RoBERTa"                     # folder with train-*.parquet
OUT = "/root/autodl-tmp/stage1_books_384.pkl"
TARGET_BLOCKS = 300000
BLOCK = 384


def process_text_lines(tok, srcs):
    """Yield contiguous token blocks from plain .txt files."""
    buf, lines = [], []
    for src in srcs:
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
                if len(lines) >= 2000:
                    for ids in tok(lines, add_special_tokens=False)["input_ids"]:
                        buf.extend(ids)
                    lines = []
                    while len(buf) >= BLOCK:
                        yield buf[:BLOCK]; buf = buf[BLOCK:]
    while len(buf) >= BLOCK:
        yield buf[:BLOCK]; buf = buf[BLOCK:]


def process_wikipedia(tok, wiki_dir):
    """Yield contiguous token blocks from Wikipedia .parquet files."""
    import pyarrow.parquet as pq
    buf = []
    for f in sorted(glob.glob(f"{wiki_dir}/train-*.parquet")):
        tbl = pq.read_table(f)
        for batch in tbl.to_batches(500):
            texts = batch.column("text").to_pylist()
            for ids in tok(texts, add_special_tokens=False,
                           truncation=True, max_length=BLOCK)["input_ids"]:
                buf.extend(ids)
                while len(buf) >= BLOCK:
                    yield buf[:BLOCK]; buf = buf[BLOCK:]
    while len(buf) >= BLOCK:
        yield buf[:BLOCK]; buf = buf[BLOCK:]


def main():
    tok = RobertaTokenizerFast.from_pretrained(MP)
    blocks = []

    print("Processing BooksCorpus ...", flush=True)
    for blk in process_text_lines(tok, BOOKS_SRCS):
        blocks.append(blk)
        if len(blocks) >= TARGET_BLOCKS // 2:
            break
    print(f"  BooksCorpus: {len(blocks)} blocks", flush=True)

    print("Processing Wikipedia ...", flush=True)
    for blk in process_wikipedia(tok, WIKI_DIR):
        blocks.append(blk)
        if len(blocks) >= TARGET_BLOCKS:
            break
        if len(blocks) % 20000 < 1:
            print(f"  total {len(blocks)} blocks", flush=True)
    print(f"  total {len(blocks)} blocks", flush=True)

    arr = np.asarray(blocks[:TARGET_BLOCKS], dtype=np.int32)
    stm = np.zeros_like(arr, dtype=np.int8)
    with open(OUT, "wb") as f:
        pickle.dump({"input_ids": arr, "special_tokens_mask": stm}, f)
    print(f"saved {OUT}: {arr.shape} ({arr.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
