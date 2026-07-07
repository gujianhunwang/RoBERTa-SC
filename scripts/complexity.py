"""Complexity & edge-deployment metrics for RoBERTa-SC.

Reports, for L=512 tokens:
  * parameter counts (total / trainable / frozen, + storage in MB);
  * FLOPs / MACs separately for the transmitter (edge modulator) and the
    receiver (semantic demodulator), since in an uplink the resource-constrained
    edge device only runs the transmitter;
  * inference latency (transmitter, receiver, end-to-end) on GPU and CPU;
  * peak GPU memory for a forward pass.

This directly answers the complexity / "edge-device friendly" questions in review.
"""

import argparse, os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roberta_sc.model import RobertaSC, Transmitter, Receiver


def mb(n_params, bytes_per=4):
    return n_params * bytes_per / 1024 / 1024


def time_module(fn, inp, device, iters=50, warmup=10):
    for _ in range(warmup):
        fn(inp)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(inp)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="roberta-base")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--c_in", type=int, default=32)
    ap.add_argument("--s2e_depth", type=int, default=2)
    ap.add_argument("--out", default="results/complexity.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RobertaSC(model_path=args.model_path, c_in=args.c_in,
                     s2e_depth=args.s2e_depth)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ck.get("model", ck))
    model.eval().to(dev)

    ps = model.param_summary()
    res = {"seq_len": args.seq_len, "c_in": args.c_in,
           "symbols_per_token": args.c_in // 2}
    res["params"] = {
        "total": ps["total"], "trainable": ps["trainable"], "frozen": ps["frozen"],
        "trainable_ratio_pct": round(ps["trainable_ratio"] * 100, 2),
        "total_MB_fp32": round(mb(ps["total"]), 1),
        "trainable_MB_fp32": round(mb(ps["trainable"]), 1),
    }

    tx, rx = Transmitter(model).to(dev).eval(), Receiver(model).to(dev).eval()
    # transmitter param count (edge device footprint): just the I/Q embedding table
    tx_params = sum(p.numel() for p in tx.iq.embedding.parameters())
    res["transmitter_params"] = tx_params
    res["transmitter_MB_fp32"] = round(mb(tx_params), 2)

    ids = torch.randint(4, 50000, (1, args.seq_len), device=dev)

    # FLOPs via calflops
    try:
        from calflops import calculate_flops
        with torch.no_grad():
            sig = tx(ids)
        def fl(mod, inp):
            flops, macs, _ = calculate_flops(model=mod, kwargs={}, args=[inp],
                                             output_as_string=False, print_results=False,
                                             print_detailed=False)
            return flops, macs
        tx_flops, tx_macs = fl(tx, ids)
        rx_flops, rx_macs = fl(rx, sig)
        res["flops"] = {
            "transmitter_GFLOPs": round(tx_flops / 1e9, 4),
            "receiver_GFLOPs": round(rx_flops / 1e9, 4),
            "end_to_end_GFLOPs": round((tx_flops + rx_flops) / 1e9, 4),
            "transmitter_GMACs": round(tx_macs / 1e9, 4),
            "receiver_GMACs": round(rx_macs / 1e9, 4),
        }
    except Exception as e:
        res["flops_error"] = str(e)

    # latency
    lat = {}
    with torch.no_grad():
        sig = tx(ids)
        lat["transmitter_ms_gpu"] = round(time_module(lambda x: tx(x), ids, dev), 3)
        lat["receiver_ms_gpu"] = round(time_module(lambda x: rx(x), sig, dev), 3)
        lat["end_to_end_ms_gpu"] = round(lat["transmitter_ms_gpu"] + lat["receiver_ms_gpu"], 3)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = rx(tx(ids))
            torch.cuda.synchronize()
            res["peak_gpu_memory_MB"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        # CPU transmitter latency (edge device)
        tx_cpu = Transmitter(model.to("cpu")).eval()
        ids_cpu = ids.to("cpu")
        lat["transmitter_ms_cpu"] = round(time_module(lambda x: tx_cpu(x), ids_cpu, "cpu", iters=20, warmup=5), 3)
        model.to(dev)
    res["latency"] = lat

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
