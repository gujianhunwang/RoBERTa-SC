"""Theoretical token-error-rate analysis (M-ary modulation view, AWGN).

The learned I/Q embedding table maps each vocabulary token to a codeword in R^32
(= 16 complex symbols * 2 real/complex). Transmitting a token means selecting one
of M=50265 codewords and adding AWGN. This is M-ary signalling; we compute:

  (1) THEORY no-prior   : union bound  P_e ~= mean_i Q(d_min(i) * sqrt(SNR/2))
      where d_min(i) is the distance from codeword i to its nearest neighbour.
  (2) SIM   no-prior    : nearest-codeword detection over all 50265 codewords
      (validates the union bound).
  (3) THEORY with-prior : same union bound, but the candidate set at each position
      is only the top-k tokens that collectively cover 99% of the frozen RoBERTa's
      predicted probability mass. k is measured per-position from a clean-context
      forward pass (no channel noise).
  (4) SIM  with-prior   : full RoBERTa-SC receiver (S2E + frozen transformer + head).

All four curves share the same SNR convention (unit avg complex-symbol power, N0=1/SNR).
Positional encodings and token-type embeddings are not part of the codeword distance
computation (they are identical at each position). The frozen transformer is treated
as a decoder that restricts the candidate set, exactly as the paper's unified
objective (Eq. 2) describes.
"""
import sys, os, math, json, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roberta_sc.model import load_roberta_sc, Transmitter, Receiver
from roberta_sc.channel import apply_channel
from roberta_sc.data import load_pickle
from transformers import RobertaTokenizer

MP = "/root/src/pretrain_model/roberta-base"
CKPT = "/autodl-fs/data/logs_euro/euro_finetune_2025-12-23_08-20-23/model_best.pt"
VAL = "/root/src/data/Eurp_sentences_robert_val.pkl"
C = 16          # complex symbols/token
SNRS = [0, 3, 6, 9, 12, 15, 18]
dev = "cuda"
random.seed(0); torch.manual_seed(0); np.random.seed(0)

def Q(x): return 0.5 * torch.erfc(x / math.sqrt(2.0))

print("loading model ...", flush=True)
model = load_roberta_sc(CKPT, MP, c_in=2*C, device=dev)
tok = RobertaTokenizer.from_pretrained(MP)
w = model.bert.roberta.embeddings.word_embeddings.embedding.weight.detach()  # (M,32)
M = w.shape[0]

# --- build the constellation: per-complex-dimension unit average power ---
re, im = w[:, :C], w[:, C:]                              # (M,16),(M,16)
ps = (re**2+im**2).mean(dim=0)                           # avg power per cplx dim
sc = torch.sqrt(1.0/(ps+1e-8))                           # (16,)
cw = torch.cat([re*sc, im*sc], dim=1).to(dev)            # (M,32), unit avg cplx power
print(f"constellation: {M} codewords in R^{cw.shape[1]} "
      f"(={C} complex symbols), avg power={((re*sc)**2+(im*sc)**2).mean().item():.3f}")

# --- load val sentences for the with-prior measurements ---
val = load_pickle(VAL)
sents = random.sample(val, 80)

# ======== EFFECTIVE CANDIDATE-SET SIZE (measured, not assumed) ========
tx, rx = Transmitter(model).to(dev), Receiver(model).to(dev)
all_k = []                    # per-position k for 99% coverage
with torch.no_grad():
    batch_logits = []
    for s in sents:
        ids = torch.tensor(tok.encode(s)).unsqueeze(0).to(dev)
        if ids.shape[-1] > 512: continue
        clean_logits = rx(tx(ids))   # no channel noise -> clean logits
        batch_logits.append(clean_logits[0, 1:-1])        # strip <s>,</s>
logits_cat = torch.cat(batch_logits, dim=0)               # (total_L, M)
probs = torch.softmax(logits_cat.float(), dim=-1)
sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
cum = torch.cumsum(sorted_probs, dim=-1)                  # (total_L, M)
k99 = (cum < 0.99).sum(dim=-1) + 1                        # first index where cum>=0.99
k99 = k99.float()
# also: the effective candidates for the TRUE token (not the argmax)
# gather the true-token index for each position — but we don't have the
# true token here; we use the LLM's own distribution's concentration.
print(f"effective candidate set (99% mass):  mean k = {k99.mean().item():.1f} "
      f"  median = {k99.median().item():.0f}  "
      f"  p10 = {k99.quantile(0.1).item():.0f}  "
      f"  p90 = {k99.quantile(0.9).item():.0f}  "
      f"  max = {k99.max().item():.0f}", flush=True)

# For the with-prior theory we use a pragma: at each position the true token
# is among the top-k where k is a typical effective set size.  We take the median
# because the mean is pulled up by a heavy tail (very uncertain positions).
K_EFF = max(1, int(k99.median().item()))
print(f"using K_EFF = {K_EFF} for the with-prior bound (median 99%-coverage size)")

# ======== ACTUAL POSITIONS FOR MEASUREMENT ========
# for the no-prior theory, we sample tokens weighted by their frequency in the
# validation set (operating distribution)
tok_freq = torch.zeros(M, dtype=torch.long)
for s in val[:2000]:
    ids = tok.encode(s)
    for t_ in ids[1:-1]:
        if t_ < M: tok_freq[t_] += 1
# sample 2000 tokens proportional to frequency
idx = torch.multinomial(tok_freq.float(), 2000, replacement=True).to(dev)
cw_samp = cw[idx]                                        # (2000,32)

# ======== (1) THEORY no-prior: union bound using d_min per sampled token ========
# memory-efficient: chunk over the full codebook, keep only d_min per query
print("computing d_min for 2000 frequency-weighted tokens (chunked over M=50265) ...", flush=True)
dmin_samp = torch.full((cw_samp.shape[0],), 1e9, device=dev)
qnorm = (cw_samp*cw_samp).sum(1)
for st in range(0, M, 2048):
    blk = cw[st:st+2048]                                 # (b,32)
    bn = (blk*blk).sum(1)
    d2 = qnorm[:, None] + bn[None, :] - 2*cw_samp @ blk.T  # (2000,b)
    d2 = torch.clamp(d2, min=0)
    # skip self (diagonal) by masking with a large value
    if st <= idx.max().item() < st+2048:
        mask_self = (idx[:, None] == (torch.arange(st, st+len(blk), device=dev)[None, :]))
        d2 = torch.where(mask_self, torch.tensor(1e12, device=dev), d2)
    d = torch.sqrt(d2)
    row_min = d.min(dim=1).values
    dmin_samp = torch.minimum(dmin_samp, row_min)
print(f"d_min (mean over 2000 freq-wtd tokens) = {dmin_samp.mean().item():.3f}  "
      f"(min={dmin_samp.min().item():.3f})", flush=True)

theory_noprior = []
for snr in SNRS:
    a = math.sqrt((10**(snr/10))/2.0)
    ter = Q(dmin_samp * a).mean().item()
    theory_noprior.append(min(ter, 1.0))

# ======== (2) SIM no-prior: nearest-codeword detection ========
print("simulating no-prior (nearest-codeword) ...", flush=True)
sim_noprior = []
for snr in SNRS:
    sigma = math.sqrt(1.0/(2*10**(snr/10)))
    o = cw_samp + torch.randn_like(cw_samp)*sigma
    on = (o*o).sum(1)
    best = torch.full((o.shape[0],), 1e12, device=dev)
    best_idx = torch.zeros(o.shape[0], dtype=torch.long, device=dev)
    for st in range(0, M, 2048):
        blk = cw[st:st+2048]
        d2 = on[:, None] + (blk*blk).sum(1)[None, :] - 2*o @ blk.T
        mv, mi = d2.min(dim=1)
        upd = mv < best; best = torch.where(upd, mv, best)
        best_idx = torch.where(upd, mi+st, best_idx)
    sim_noprior.append((best_idx != idx).float().mean().item())

# ======== (3) SIM with-prior: full RoBERTa-SC receiver ========
print("simulating with-prior (full Rx) ...", flush=True)
sim_prior = []
for snr in SNRS:
    err = n = 0
    for s in sents:
        ids = torch.tensor(tok.encode(s)).unsqueeze(0).to(dev)
        if ids.shape[-1] > 512: continue
        logits = rx(apply_channel(tx(ids), snr, "awgn"))
        pred = logits.argmax(-1)[0]
        t_ = ids[0]
        err += (pred[1:-1] != t_[1:-1]).sum().item()
        n += t_.shape[0]-2
    sim_prior.append(err/max(n,1))

# ======== (4) THEORY with-prior ========
# For each position in the val set, the top-K_EFF tokens from the LLM's clean
# prediction are the "plausible" set. The with-prior theory bound uses the
# minimum codeword distance among those K_EFF candidates (excluding the true token).
print(f"computing with-prior d_min_eff (K_EFF={K_EFF}) ...", flush=True)
deff = []
with torch.no_grad():
    for s in sents:
        ids = torch.tensor(tok.encode(s)).unsqueeze(0).to(dev)
        if ids.shape[-1] > 512: continue
        clean_logits = rx(tx(ids))
        topk_vals, topk_idx = clean_logits[0].topk(K_EFF+1, dim=-1)   # +1 in case true is in top-K
        for pos in range(1, ids.shape[-1]-1):
            tt = ids[0, pos].item()
            cand = topk_idx[pos].tolist()
            others = [c_ for c_ in cand if c_ != tt][:K_EFF]
            if not others: continue
            d = torch.norm(cw[tt] - cw[torch.tensor(others, device=dev)], dim=1)
            deff.append(d.min().item())
deff_t = torch.tensor(deff, device=dev)
print(f"with-prior d_min: mean={deff_t.mean().item():.3f} median={deff_t.median().item():.3f} "
      f"(vs global d_min mean={dmin_samp.mean().item():.3f})", flush=True)

theory_prior = []
for snr in SNRS:
    a = math.sqrt((10**(snr/10))/2.0)
    theory_prior.append(min(Q(deff_t*a).mean().item(), 1.0))

# ======== report & save ========
print("\n SNR | theoryNoP  simNoP | theoryP   simFullRx | gap (semantic gain)")
for i, snr in enumerate(SNRS):
    gap = theory_noprior[i]/max(sim_prior[i], 1e-9)
    print(f" {snr:2d}  | {theory_noprior[i]:.5f}  {sim_noprior[i]:.5f} | "
          f"{theory_prior[i]:.5f}  {sim_prior[i]:.5f} | {gap:.1f}x")

out = {
    "snr": SNRS,
    "theory_noprior": theory_noprior, "sim_noprior": sim_noprior,
    "theory_prior": theory_prior, "sim_fullsystem": sim_prior,
    "dmin_mean_global": dmin_samp.mean().item(),
    "dmin_mean_with_prior": deff_t.mean().item(),
    "dmin_ratio": deff_t.mean().item() / max(dmin_samp.mean().item(), 1e-9),
    "K_effective": K_EFF,
    "K_distribution": {"mean": k99.mean().item(), "median": k99.median().item(),
                        "p10": k99.quantile(0.1).item(), "p90": k99.quantile(0.9).item()},
    "method": "fixed-K_eff from median 99%-mass coverage of the frozen-RoBERTa prior"
}
json.dump(out, open("results/theory_ter_awgn.json", "w"), indent=2)
print("\nsaved results/theory_ter_awgn.json")
