"""Information-theoretic feasibility analysis for the 16-symbols-per-token design.

Answers the reviewer question: why can 16 complex symbols per token carry the
semantic content of a token, and how does this relate to channel capacity?

We quantify three things on the Europarl test sentences:
  1. The *source* information rate of the text:
       - H1: empirical unigram token entropy (bits/token);
       - H_cond: conditional entropy estimated by masked-LM cross-entropy of a
         frozen RoBERTa-base (bits/token) -- a realistic estimate of the
         residual semantic information per token given its context.
  2. The *channel* capacity available per token under the unit-power,
     average-SNR convention: C = D * log2(1 + SNR) bits/token, with D = 16
     complex symbols (Gaussian-input AWGN capacity per complex dimension).
  3. The implied operating point: source rate vs. available capacity, and the
     effective bits carried per complex symbol.
"""

import argparse, os, sys, json, math, random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roberta_sc.data import load_pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="roberta-base")
    ap.add_argument("--val", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--D", type=int, default=16, help="complex symbols per token")
    ap.add_argument("--out", default="results/information_analysis.json")
    args = ap.parse_args()

    from transformers import RobertaTokenizer, RobertaForMaskedLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = RobertaTokenizer.from_pretrained(args.model_path)
    mlm = RobertaForMaskedLM.from_pretrained(args.model_path).eval().to(dev)

    val = load_pickle(args.val)
    random.seed(0)
    sents = [random.choice(val) for _ in range(args.sample)]

    # ---- chars/token and unigram entropy ----
    from collections import Counter
    counts = Counter()
    total_chars = total_tokens = 0
    enc_cache = []
    for s in sents:
        ids = tok.encode(s)
        enc_cache.append(ids)
        counts.update(ids[1:-1])               # exclude <s>,</s>
        total_chars += len(s)
        total_tokens += len(ids) - 2
    n = sum(counts.values())
    probs = np.array([c / n for c in counts.values()])
    H1 = float(-np.sum(probs * np.log2(probs)))
    chars_per_token = total_chars / total_tokens

    # ---- conditional entropy via masked-LM cross-entropy (bits/token) ----
    mask_id = tok.mask_token_id
    ce_bits = []
    with torch.no_grad():
        for ids in enc_cache[:200]:
            if len(ids) > 512:
                continue
            ids_t = torch.tensor(ids, device=dev).unsqueeze(0)
            # mask 15% of the content tokens
            pos = list(range(1, len(ids) - 1))
            random.shuffle(pos)
            pos = pos[:max(1, int(0.15 * len(pos)))]
            masked = ids_t.clone()
            for p in pos:
                masked[0, p] = mask_id
            logits = mlm(masked).logits[0]
            for p in pos:
                lp = F.log_softmax(logits[p], dim=-1)[ids[p]].item()
                ce_bits.append(-lp / math.log(2))
    H_cond = float(np.mean(ce_bits))

    # ---- channel capacity per token ----
    cap = {}
    for snr in [0, 3, 6, 9, 12, 15, 18]:
        snr_lin = 10 ** (snr / 10)
        cap_per_symbol = math.log2(1 + snr_lin)         # complex AWGN, Gaussian input
        cap[f"{snr}dB"] = {
            "capacity_bits_per_symbol": round(cap_per_symbol, 3),
            "capacity_bits_per_token": round(args.D * cap_per_symbol, 2),
            "source_rate_over_capacity": round(H_cond / (args.D * cap_per_symbol), 4),
        }

    res = {
        "sample": args.sample,
        "chars_per_token": round(chars_per_token, 3),
        "unigram_entropy_bits_per_token_H1": round(H1, 3),
        "conditional_entropy_bits_per_token_Hcond": round(H_cond, 3),
        "bits_per_char_equivalent": round(H_cond / chars_per_token, 3),
        "complex_symbols_per_token_D": args.D,
        "effective_bits_per_symbol_at_Hcond": round(H_cond / args.D, 4),
        "capacity_per_token": cap,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
