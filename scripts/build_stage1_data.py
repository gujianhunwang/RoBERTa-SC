"""Build Stage-1 pretraining data: contiguous 512-token blocks from BooksCorpus.

Produces a compact dict {input_ids: (N,512) int32, special_tokens_mask: (N,512) int8}
compatible with scripts/train.py. Stage-1 teaches a general, robust token<->I/Q
alignment (as in the original two-stage recipe) before Europarl fine-tuning.
"""
import sys, pickle, numpy as np
from transformers import RobertaTokenizerFast

MP = "/root/src/pretrain_model/roberta-base"
SRCS = ["/root/autodl-tmp/RoBERTa/books_large_p1.txt",
        "/root/autodl-tmp/RoBERTa/books_large_p2.txt"]
OUT = "/root/autodl-tmp/stage1_books_384.pkl"
TARGET_BLOCKS = 300000
BLOCK = 384

tok = RobertaTokenizerFast.from_pretrained(MP)
buf = []
blocks = []
lines = []
n_lines = 0
done = False
for SRC in SRCS:
    if done:
        break
    with open(SRC, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line); n_lines += 1
            if len(lines) >= 2000:
                enc = tok(lines, add_special_tokens=False)["input_ids"]
                for ids in enc:
                    buf.extend(ids)
                lines = []
                while len(buf) >= BLOCK:
                    blocks.append(buf[:BLOCK]); buf = buf[BLOCK:]
                if len(blocks) >= TARGET_BLOCKS:
                    done = True; break
                if len(blocks) % 20000 < 1:
                    print(f"blocks={len(blocks)}", flush=True)

arr = np.asarray(blocks[:TARGET_BLOCKS], dtype=np.int32)
stm = np.zeros_like(arr, dtype=np.int8)   # contiguous text: count all tokens
with open(OUT, "wb") as f:
    pickle.dump({"input_ids": arr, "special_tokens_mask": stm}, f)
print(f"saved {OUT}: {arr.shape} ({arr.nbytes/1e6:.0f} MB), from {n_lines} lines")
