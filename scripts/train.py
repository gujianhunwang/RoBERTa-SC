"""Train / fine-tune RoBERTa-SC on tokenised text.

Only the I/Q embedding (incl. the S2E projection) and the output head are
updated; the 12 transformer blocks stay frozen.  The channel (AWGN or Rayleigh)
is simulated differentiably inside the model so training is end-to-end.

Example:
    python scripts/train.py --model_path roberta-base \
        --train data/Eurp_tokens_robert_train.pkl \
        --val   data/Eurp_tokens_robert_val.pkl \
        --channel rayleigh --max_steps 1000 --batch_size 64 --out_dir runs/euro
"""

import argparse, os, sys, time, datetime, pickle
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roberta_sc.model import RobertaSC


class TokenBatches:
    """Iterate over a dict {input_ids, attention_mask, special_tokens_mask}."""
    def __init__(self, data, batch_size, train=True, max_len=512):
        self.d = data; self.bs = batch_size; self.train = train; self.max_len = max_len
        self.n = len(data["input_ids"]); self.pos = 0

    def next(self):
        if self.pos >= self.n:
            if not self.train:
                self.pos = 0; return None
            self.pos = 0
        ids_list = self.d["input_ids"][self.pos:self.pos + self.bs]
        stm_list = self.d["special_tokens_mask"][self.pos:self.pos + self.bs]
        self.pos += self.bs
        if len(ids_list) == 0:
            return None
        # truncate to the backbone's max length (RoBERTa-base: 512 tokens)
        ids_list = [x[:self.max_len] for x in ids_list]
        stm_list = [x[:self.max_len] for x in stm_list]
        # pad to the longest sequence in the batch (RoBERTa pad id = 1);
        # padded positions are marked in the special-tokens mask so they are
        # excluded from the loss.
        maxlen = max(len(x) for x in ids_list)
        ids = torch.full((len(ids_list), maxlen), 1, dtype=torch.long)
        stm = torch.ones((len(ids_list), maxlen), dtype=torch.long)
        for i, (a, b) in enumerate(zip(ids_list, stm_list)):
            ids[i, :len(a)] = torch.tensor(a, dtype=torch.long)
            stm[i, :len(b)] = torch.tensor(b, dtype=torch.long)
        return ids, stm


def masked_ce(logits, ids, special_mask):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids.view(-1), reduction="none").view(ids.size())
    valid = 1 - special_mask
    return (loss * valid).sum() / (valid.sum() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="roberta-base")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out_dir", default="runs/roberta_sc")
    ap.add_argument("--channel", choices=["awgn", "rayleigh"], default="rayleigh")
    ap.add_argument("--c_in", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--snr_skew", type=float, default=1.0,
                    help=">1 biases the training-SNR curriculum toward low SNR")
    ap.add_argument("--s2e_depth", type=int, default=2,
                    help="depth of S2E MLP (2=original pure-linear, >=3 adds GELU)")
    ap.add_argument("--s2e_activation", dest="s2e_activation", action="store_true", default=None,
                    help="force GELU activations in S2E (auto: s2e_depth>2)")
    ap.add_argument("--no_s2e_activation", dest="s2e_activation", action="store_false",
                    help="force pure-linear S2E (no activations)")
    ap.add_argument("--clean_snr_prob", type=float, default=0.0,
                    help="fraction of batches trained at very high SNR to combat error floor")
    ap.add_argument("--clean_freq", type=int, default=100,
                    help="apply clean-SNR every N steps (requires --clean_snr_prob > 0)")
    ap.add_argument("--repel_margin", type=float, default=0.0,
                    help="repulsion loss margin (0=off, typical 0.5-2.0)")
    ap.add_argument("--repel_weight", type=float, default=0.0,
                    help="repulsion loss weight in total loss (typical 0.01-0.1)")
    ap.add_argument("--eval_freq", type=int, default=200)
    ap.add_argument("--ckpt_freq", type=int, default=100)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_len", type=int, default=512, help="truncate training sequences to this length")
    ap.add_argument("--init_from", default=None,
                    help="warm-start: load all shape-matching params from this checkpoint "
                         "(e.g. a well-trained model with a different c_in); the I/Q code "
                         "(embedding, linear0) is left randomly initialised")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.train, "rb") as f:
        train_data = pickle.load(f)
    val_data = None
    if args.val:
        with open(args.val, "rb") as f:
            val_data = pickle.load(f)

    model = RobertaSC(model_path=args.model_path, c_in=args.c_in, channel=args.channel,
                      train_snr_range=(args.snr_min, args.snr_max),
                      train_snr_skew=args.snr_skew, s2e_depth=args.s2e_depth,
                      s2e_activation=args.s2e_activation,
                      repel_margin=args.repel_margin,
                      repel_weight=args.repel_weight).to(dev)
    if args.init_from:
        src = torch.load(args.init_from, map_location="cpu")
        src = src.get("model", src)
        own = model.state_dict()
        kept = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        skipped = [k for k in own if k not in kept]
        own.update(kept); model.load_state_dict(own)
        print(f"warm-start from {args.init_from}: loaded {len(kept)} tensors, "
              f"reinitialised {len(skipped)} (e.g. {skipped[:3]})")
    print("params:", model.param_summary())
    opt = model.configure_optimizers(args.lr, args.weight_decay, device_type=dev)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.max_steps,
        pct_start=args.warmup / args.max_steps, anneal_strategy="cos")
    scaler = torch.cuda.amp.GradScaler()
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); start = ck.get("step", 0)

    tr = TokenBatches(train_data, args.batch_size, train=True, max_len=args.max_len)
    best = float("inf")
    for step in range(start, args.max_steps):
        model.train(); opt.zero_grad(); t0 = time.time(); tot = 0.0
        for _ in range(args.grad_accum):
            batch = tr.next()
            if batch is None:
                tr.pos = 0; batch = tr.next()
            ids, stm = (x.to(dev) for x in batch)
            with torch.cuda.amp.autocast():
                logits = model(ids)
                loss = masked_ce(logits, ids, stm) / args.grad_accum
            if args.repel_weight > 0:
                rl = model.get_repel_loss()
                if rl.item() > 0:
                    loss = loss + args.repel_weight * rl
            scaler.scale(loss).backward(); tot += loss.item() * args.grad_accum
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        # ---- clean-SNR anti-floor ----
        if args.clean_snr_prob > 0 and (step + 1) % args.clean_freq == 0 and step > 0:
            orig_range = model.bert.roberta.embeddings.word_embeddings.train_snr_range
            try:
                model.bert.roberta.embeddings.word_embeddings.train_snr_range = (30.0, 30.0)
                ids, stm = (x.to(dev) for x in tr.next())
                if ids is None or len(ids) == 0:
                    ids, stm = (x.to(dev) for x in tr.next())
                with torch.cuda.amp.autocast():
                    lc = masked_ce(model(ids), ids, stm)
                if args.repel_weight > 0:
                    rl = model.get_repel_loss()
                    if rl.item() > 0:
                        lc = lc + args.repel_weight * rl
                scaler.scale(lc).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            finally:
                model.bert.roberta.embeddings.word_embeddings.train_snr_range = orig_range

        if (step + 1) % 10 == 0:
            print(f"step {step+1}/{args.max_steps} loss {tot/args.grad_accum:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.2f}s", flush=True)
        if val_data is not None and (step + 1) % args.eval_freq == 0:
            vl = evaluate(model, val_data, dev)
            print(f"  [val] step {step+1} loss {vl:.4f}", flush=True)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "step": step + 1, "val_loss": vl},
                           os.path.join(args.out_dir, "model_best.pt"))
        if (step + 1) % args.ckpt_freq == 0:
            torch.save({"model": model.state_dict(), "step": step + 1},
                       os.path.join(args.out_dir, f"model_{step+1:05d}.pt"))
    print("done. best val loss:", best)


@torch.no_grad()
def evaluate(model, val_data, dev, max_batches=50, bs=20):
    model.eval(); vb = TokenBatches(val_data, bs, train=False); tot = n = 0.0
    for _ in range(max_batches):
        batch = vb.next()
        if batch is None:
            break
        ids, stm = (x.to(dev) for x in batch)
        with torch.cuda.amp.autocast():
            logits = model(ids); loss = masked_ce(logits, ids, stm)
        tot += loss.item() * ids.size(0); n += ids.size(0)
    return tot / max(n, 1)


if __name__ == "__main__":
    main()
