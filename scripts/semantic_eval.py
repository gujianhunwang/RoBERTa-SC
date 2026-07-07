"""Multi-dimensional semantic evaluation for RoBERTa-SC.

For each SNR we recover sentences through the channel and score them with
several complementary metrics so the evaluation does not rely on BLEU alone:

  * BLEU-1 (lexical n-gram overlap);
  * Sentence similarity -- BERT (bert-base-uncased, mean-pooled last hidden
    state, cosine). This is the exact metric used for Fig. 3 of the paper;
  * Sentence similarity -- RoBERTa (roberta-base, mean-pooled, cosine), a second
    independent sentence encoder to show the conclusion is encoder-agnostic;
  * BERTScore-F1 (token-level contextual similarity, local roberta-base);
  * Token error rate (TER) -- fraction of mismatched tokens, a semantic-system
    analogue of bit error rate.

All sentence encoders are loaded from local paths (offline-friendly).
"""

import argparse, os, sys, json, random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roberta_sc.model import load_roberta_sc, Transmitter, Receiver
from roberta_sc.channel import apply_channel
from roberta_sc.metrics import bleu_scores, token_error_rate
from roberta_sc.data import load_pickle
from transformers import RobertaTokenizer, AutoTokenizer, AutoModel


class MeanPoolCosine:
    """Mean-pooled hidden-state cosine similarity with a local encoder."""
    def __init__(self, path, device, max_norm=True):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).eval().to(device)
        self.device = device
        self.max_norm = max_norm

    @torch.no_grad()
    def embed(self, sents):
        enc = self.tok(sents, add_special_tokens=True, max_length=256,
                       padding="max_length", truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        v = self.model(**enc).last_hidden_state.mean(dim=1)
        return v

    @torch.no_grad()
    def __call__(self, refs, recs):
        from sklearn.preprocessing import normalize
        from scipy.spatial.distance import cosine
        v1 = self.embed(refs).cpu().numpy(); v2 = self.embed(recs).cpu().numpy()
        if self.max_norm:
            v1 = normalize(v1, axis=0, norm="max"); v2 = normalize(v2, axis=0, norm="max")
        return [1 - cosine(a, b) for a, b in zip(v1, v2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_path", default="roberta-base")
    ap.add_argument("--bert_path", default="bert-base-uncased")
    ap.add_argument("--val", required=True)
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], default="rayleigh")
    ap.add_argument("--snrs", default="0,3,6,9,12,15,18")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--c_in", type=int, default=16)
    ap.add_argument("--s2e_depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_roberta_sc(args.checkpoint, args.model_path, c_in=args.c_in,
                            s2e_depth=args.s2e_depth, device=dev)
    tx, rx = Transmitter(model).to(dev), Receiver(model).to(dev)
    tok = RobertaTokenizer.from_pretrained(args.model_path)
    val = load_pickle(args.val)

    sim_bert = MeanPoolCosine(args.bert_path, dev)
    sim_roberta = MeanPoolCosine(args.model_path, dev)
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type=args.model_path, num_layers=10, device=dev, lang="en")

    sents = [random.choice(val) for _ in range(args.sample)]
    snrs = [int(s) for s in args.snrs.split(",")]
    res = {"channel": args.channel, "snr": snrs, "sample": args.sample,
           "bleu1": [], "sim_bert": [], "sim_roberta": [], "bertscore_f1": [], "token_error_rate": []}

    for snr in snrs:
        refs, recs, b1, ters = [], [], [], []
        for sent in sents:
            ids = torch.tensor(tok.encode(sent)).unsqueeze(0).to(dev)
            if ids.shape[-1] > 512:
                continue
            logits = rx(apply_channel(tx(ids), snr, args.channel))
            rec_ids = torch.argmax(logits, -1)[0]
            rec = tok.decode(rec_ids[1:-1])
            refs.append(sent); recs.append(rec)
            b1.append(bleu_scores(rec, sent)[0])
            ters.append(token_error_rate(ids[0, 1:-1].cpu().numpy(), rec_ids[1:-1].cpu().numpy()))
        sb = float(np.mean(sim_bert(refs, recs)))
        sr = float(np.mean(sim_roberta(refs, recs)))
        _, _, f1 = scorer.score(recs, refs)
        res["bleu1"].append(round(float(np.mean(b1)), 4))
        res["sim_bert"].append(round(sb, 4))
        res["sim_roberta"].append(round(sr, 4))
        res["bertscore_f1"].append(round(float(f1.mean()), 4))
        res["token_error_rate"].append(round(float(np.mean(ters)), 4))
        print(f"[{args.channel}] SNR={snr:2d}dB BLEU-1={res['bleu1'][-1]:.3f} "
              f"sim_BERT={sb:.3f} sim_RoBERTa={sr:.3f} BERTScore-F1={res['bertscore_f1'][-1]:.3f} "
              f"TER={res['token_error_rate'][-1]:.3f}", flush=True)

    out = args.out or f"results/semantic_{args.channel}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
