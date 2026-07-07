"""Traditional separate source--channel coding baseline.

Pipeline:  text --Huffman--> bits --conv. coding--> coded bits --64-QAM-->
symbols --channel--> demod --Viterbi--> bits --Huffman^-1--> text.

Reports, vs the *same* per-symbol SNR convention used for RoBERTa-SC:
  * BER   (coded bit error rate, post-decoding on information bits),
  * BLER  (block/sentence error rate),
  * BLEU-1 of the recovered text,
  * channel uses per token (bandwidth) for fair bandwidth accounting.

This is the only baseline for which bit-level BER/BLER are well defined; neural
semantic baselines (JSCC/DeepSC/LLM-SC) transmit learned symbols, not bits, so
for them we report BLEU and token error rate instead (see semantic_eval.py).
"""

import argparse, os, sys, json, heapq, random
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roberta_sc.data import load_pickle


# ----------------------------- Huffman ------------------------------------- #
def build_huffman(freqs):
    heap = [[w, [sym, ""]] for sym, w in freqs.items()]
    heapq.heapify(heap)
    if len(heap) == 1:                      # degenerate
        return {heap[0][1][0]: "0"}
    while len(heap) > 1:
        lo = heapq.heappop(heap); hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return {sym: code for sym, code in heap[0][1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="roberta-base")
    ap.add_argument("--val", required=True)
    ap.add_argument("--sample", type=int, default=80)
    ap.add_argument("--snrs", default="0,3,6,9,12,15,18")
    ap.add_argument("--code", choices=["conv", "turbo", "none"], default="conv")
    ap.add_argument("--qam", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baseline_traditional.json")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    from transformers import RobertaTokenizer
    from commpy.modulation import QAMModem
    from commpy.channelcoding import Trellis, conv_encode, viterbi_decode
    from nltk.translate.bleu_score import sentence_bleu

    tok = RobertaTokenizer.from_pretrained(args.model_path)
    val = load_pickle(args.val)

    # --- build Huffman table from corpus token frequencies ---
    freqs = Counter()
    for s in val:
        freqs.update(tok.encode(s))
    code = build_huffman(freqs)
    avg_bits_per_token = sum(freqs[s] * len(code[s]) for s in freqs) / sum(freqs.values())

    # --- channel code (standard K=7 (133,171) rate-1/2 convolutional, soft-decision) ---
    if args.code == "conv":
        trellis = Trellis(np.array([6]), np.array([[0o133, 0o171]]))  # K=7, rate 1/2
        rate = 0.5
    else:
        trellis = None; rate = 1.0

    modem = QAMModem(args.qam)
    bps = modem.num_bits_symbol
    Es = np.mean(np.abs(modem.constellation) ** 2)  # avg constellation energy

    sents = [random.choice(val) for _ in range(args.sample)]
    snrs = [int(x) for x in args.snrs.split(",")]
    res = {"modulation": f"{args.qam}-QAM", "bits_per_symbol": bps,
           "coding": args.code, "code_rate": rate,
           "avg_huffman_bits_per_token": round(avg_bits_per_token, 3),
           "snr": snrs, "ber": [], "bler": [], "bleu1": [], "channel_uses_per_token": []}

    for snr in snrs:
        snr_lin = 10 ** (snr / 10.0)
        n0 = Es / snr_lin                      # Es/N0 convention (matches RoBERTa-SC per-symbol SNR)
        bit_err = bit_tot = blk_err = blk_tot = 0
        bleus = []; cu_per_tok = []
        for sent in sents:
            ids = tok.encode(sent)
            bits = np.array([int(b) for t in ids for b in code[t]], dtype=int)
            info_len = len(bits)
            coded = conv_encode(bits, trellis) if trellis is not None else bits
            # pad to multiple of bps
            pad = (-len(coded)) % bps
            coded_p = np.concatenate([coded, np.zeros(pad, dtype=int)])
            syms = modem.modulate(coded_p)
            syms = syms / np.sqrt(Es)          # unit average symbol power
            noise = (np.random.randn(*syms.shape) + 1j * np.random.randn(*syms.shape)) * np.sqrt(n0 / Es / 2)
            rx = syms + noise
            if trellis is not None:
                llr = modem.demodulate(rx * np.sqrt(Es), demod_type="soft", noise_var=n0)
                llr = llr[:len(coded)]
                dec = viterbi_decode(llr, trellis, decoding_type="unquantized")[:info_len]
            else:
                demod = modem.demodulate(rx * np.sqrt(Es), demod_type="hard")[:info_len]
                dec = demod[:info_len]
            n_err = int(np.sum(dec[:info_len] != bits[:info_len]))
            bit_err += n_err; bit_tot += info_len
            blk_err += int(n_err > 0); blk_tot += 1
            cu_per_tok.append(len(syms) / len(ids))
            # Huffman decode -> tokens -> text -> BLEU
            rec_tokens = huffman_decode(dec[:info_len], code)
            rec = tok.decode(rec_tokens[1:-1]) if len(rec_tokens) > 2 else ""
            bleus.append(sentence_bleu([rec.split()], sent.split(), weights=(1, 0, 0, 0)))
        res["ber"].append(round(bit_err / max(bit_tot, 1), 5))
        res["bler"].append(round(blk_err / max(blk_tot, 1), 4))
        res["bleu1"].append(round(float(np.mean(bleus)), 4))
        res["channel_uses_per_token"].append(round(float(np.mean(cu_per_tok)), 2))
        print(f"SNR={snr:2d}dB BER={res['ber'][-1]:.4f} BLER={res['bler'][-1]:.3f} "
              f"BLEU-1={res['bleu1'][-1]:.3f} ch.uses/token={res['channel_uses_per_token'][-1]}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", args.out)


def huffman_decode(bits, code):
    inv = {v: k for k, v in code.items()}
    out, cur = [], ""
    for b in bits:
        cur += str(int(b))
        if cur in inv:
            out.append(inv[cur]); cur = ""
    return out


if __name__ == "__main__":
    main()
