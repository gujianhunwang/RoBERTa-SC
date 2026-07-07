"""Aggregate the c_in sweep into a semantic rate--distortion curve.

Reads results/eval_cin{C}_awgn.json for the swept c_in values and prints a
table of BLEU-1 vs (complex symbols per token = c_in/2) at selected SNRs,
then renders assets/rate_distortion.png.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CINS = [2, 4, 8, 16, 32]
SNR_SHOW = [0, 6, 12, 18]

rows = []
for c in CINS:
    p = f"results/eval_cin{c}_awgn.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    snr = d["snr"]
    row = {"c_in": c, "complex_per_token": c // 2, "tag": "single-stage"}
    for s in SNR_SHOW:
        if s in snr:
            row[f"bleu1@{s}"] = d["bleu1"][snr.index(s)]
    rows.append(row)

# warm-start (rescue) variants, if present
for c in [4, 8]:
    p = f"results/eval_cin{c}_warm_awgn.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p)); snr = d["snr"]
    row = {"c_in": c, "complex_per_token": c // 2, "tag": "warm-start"}
    for s in SNR_SHOW:
        if s in snr:
            row[f"bleu1@{s}"] = d["bleu1"][snr.index(s)]
    rows.append(row)

print(f"{'sym/token':>9} {'recipe':>12} | " + " | ".join(f'BLEU1@{s}dB' for s in SNR_SHOW))
for r in rows:
    print(f"{r['complex_per_token']:>9} {r['tag']:>12} | " +
          " | ".join(f"{r.get(f'bleu1@{s}', float('nan')):.3f}    " for s in SNR_SHOW))

# figure: BLEU-1 vs symbols/token at each SNR
plt.figure(figsize=(5, 3.5))
for s in SNR_SHOW:
    xs = [r["complex_per_token"] for r in rows if f"bleu1@{s}" in r]
    ys = [r[f"bleu1@{s}"] for r in rows if f"bleu1@{s}" in r]
    plt.plot(xs, ys, marker="o", label=f"{s} dB")
plt.xlabel("complex symbols per token"); plt.ylabel("BLEU-1"); plt.ylim(0, 1)
plt.title("Semantic rate--distortion (AWGN)"); plt.grid(alpha=.3); plt.legend(fontsize=8, title="SNR")
plt.tight_layout(); plt.savefig("assets/rate_distortion.png", dpi=150)
print("saved assets/rate_distortion.png")
