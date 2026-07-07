"""Plot BLEU/semantic/baseline curves from the saved results/*.json files."""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(name):
    with open(os.path.join("results", name)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out_dir", default="assets"); a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # BLEU vs SNR (AWGN + Rayleigh)
    for ch in ["awgn", "rayleigh"]:
        d = load(f"eval_{ch}.json")
        plt.figure(figsize=(5, 3.5))
        for k in ["bleu1", "bleu2", "bleu3", "bleu4"]:
            plt.plot(d["snr"], d[k], marker="o", label=k.upper().replace("BLEU", "BLEU-"))
        plt.xlabel("SNR (dB)"); plt.ylabel("BLEU"); plt.ylim(0, 1)
        plt.title(f"RoBERTa-SC ({ch.upper()})"); plt.grid(alpha=.3); plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{a.out_dir}/bleu_{ch}.png", dpi=150); plt.close()

    # RoBERTa-SC vs traditional baseline (AWGN BLEU-1)
    e = load("eval_awgn.json"); b = load("baseline_traditional.json")
    plt.figure(figsize=(5, 3.5))
    plt.plot(e["snr"], e["bleu1"], marker="o", label="RoBERTa-SC")
    plt.plot(b["snr"], b["bleu1"], marker="s", label="Huffman+Conv+64QAM")
    plt.xlabel("SNR (dB)"); plt.ylabel("BLEU-1"); plt.ylim(0, 1)
    plt.title("Graceful degradation vs. cliff effect"); plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{a.out_dir}/vs_traditional.png", dpi=150); plt.close()
    print("saved figures to", a.out_dir)


if __name__ == "__main__":
    main()
