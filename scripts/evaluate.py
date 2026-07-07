"""Reproduce BLEU-1..4 and sentence-similarity vs SNR curves for RoBERTa-SC.

Usage:
    python scripts/evaluate.py --checkpoint <best.pt> --channel awgn \
        --val data/Eurp_sentences_robert_val.pkl --sample 200

Outputs a JSON file with per-SNR BLEU-1..4 and (optionally) similarity.
The SNR convention follows roberta_sc.channel (unit average symbol power).
"""

import argparse
import json
import os
import random

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roberta_sc.model import load_roberta_sc, Transmitter, Receiver
from roberta_sc.channel import apply_channel
from roberta_sc.metrics import bleu_scores
from roberta_sc.data import load_pickle
from transformers import RobertaTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_path", default="roberta-base")
    ap.add_argument("--val", required=True, help="pickle list of raw sentences")
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    ap.add_argument("--snrs", default="0,3,6,9,12,15,18")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--c_in", type=int, default=32)
    ap.add_argument("--s2e_depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--similarity", action="store_true", help="also compute BERT sentence similarity")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_roberta_sc(args.checkpoint, args.model_path, c_in=args.c_in, device=dev,
                           s2e_depth=args.s2e_depth)
    tx, rx = Transmitter(model).to(dev), Receiver(model).to(dev)
    tok = RobertaTokenizer.from_pretrained(args.model_path)
    val = load_pickle(args.val)

    sim_metric = None
    if args.similarity:
        from roberta_sc.metrics import SentenceSimilarity
        sim_metric = SentenceSimilarity(device=dev)

    snrs = [int(s) for s in args.snrs.split(",")]
    results = {"channel": args.channel, "c_in": args.c_in,
               "symbols_per_token": args.c_in // 2, "sample": args.sample, "snr": snrs}
    b1 = b2 = b3 = b4 = sim = None
    out_b = {k: [] for k in ["bleu1", "bleu2", "bleu3", "bleu4"]}
    out_sim = []

    sentences = [random.choice(val) for _ in range(args.sample)]
    for snr in snrs:
        acc = np.zeros(4); sims = []; n = 0
        recs, refs = [], []
        for sent in sentences:
            ids = torch.tensor(tok.encode(sent)).unsqueeze(0).to(dev)
            if ids.shape[-1] > 512:
                continue
            sig = tx(ids)
            rxsig = apply_channel(sig, snr, args.channel)
            logits = rx(rxsig)
            rec = tok.decode(torch.argmax(logits, -1)[0][1:-1])
            acc += np.array(bleu_scores(rec, sent))
            n += 1
            if sim_metric is not None:
                recs.append(rec); refs.append(sent)
        acc /= max(n, 1)
        for i, k in enumerate(["bleu1", "bleu2", "bleu3", "bleu4"]):
            out_b[k].append(round(float(acc[i]), 4))
        line = f"[{args.channel}] SNR={snr:2d}dB  BLEU-1={acc[0]:.3f} BLEU-2={acc[1]:.3f} BLEU-3={acc[2]:.3f} BLEU-4={acc[3]:.3f} (n={n})"
        if sim_metric is not None:
            s = float(np.mean(sim_metric(refs, recs)))
            out_sim.append(round(s, 4)); line += f"  sim={s:.3f}"
        print(line, flush=True)

    results.update(out_b)
    if sim_metric is not None:
        results["similarity"] = out_sim
    out = args.out or f"results/eval_{args.channel}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
