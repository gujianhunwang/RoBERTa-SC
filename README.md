# RoBERTa-SC: LLM-Powered Semantic Communication via Efficient Signal–Text Alignment

Official implementation of **RoBERTa-SC**, a parameter-efficient
semantic-communication framework. A frozen, pre-trained RoBERTa is turned
into a *semantic-aware joint source–channel codec operating directly on
physical-layer I/Q symbols*. Only the I/Q-embedding, the Signal-to-Embedding
(S2E) projection, and the output head are trained (≈ 32 % of the parameters);
the 12 transformer blocks stay frozen, preserving their linguistic prior.

> **Architecture note (per-token, not per-sentence).**  The I/Q-embedding is a
> per-token look-up table (`nn.Embedding(50265, c_in)`) that maps every token
> to `c_in` real-valued channel dimensions, i.e. `c_in/2` complex symbols per
> token.  With the default `c_in = 16`, an `L`-token sentence is transmitted
> with `8L` complex symbols — a per-symbol budget comparable to DeepSC's
> 8 complex symbols/word.  The advantage of RoBERTa-SC lies in (i) graceful
> low-SNR degradation through the frozen-LLM semantic prior, (ii) support for
> long sequences (up to 512 tokens versus ≈ 30 words for prior systems), and
> (iii) parameter-efficient training with a lightweight edge transmitter.

---

## Installation

```bash
git clone https://github.com/gujianhunwang/RoBERTa-SC
cd RoBERTa-SC
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

All scripts accept `--model_path` (path to a local `roberta-base` directory or a
Hugging Face model name) and `--bert_path` (path to a local
`bert-base-uncased`).  Our environment uses local model directories; if you
have internet access, the Hugging Face hub names (`roberta-base`,
`bert-base-uncased`) work directly.

---

## Repository layout

```
roberta_sc/              core Python package
  ├── model.py           RobertaSC, IQ_Embedding, Transmitter, Receiver
  ├── channel.py         AWGN & Rayleigh (flat / fast fading) differentiable channels
  ├── metrics.py         BLEU, sentence similarity, BERTScore, token error rate
  └── data.py            Europarl preprocessing utilities
scripts/
  ├── preprocess_europarl.py    build tokenised Europarl train/val splits
  ├── build_stage1_data.py      build Stage-1 BooksCorpus contiguous blocks
  ├── train.py                  train / fine-tune RoBERTa-SC (two-stage recipe)
  ├── evaluate.py               BLEU-1..4 vs SNR (AWGN / Rayleigh)
  ├── semantic_eval.py          multi-metric semantic evaluation
  ├── complexity.py             params / FLOPs / latency / memory
  ├── information_analysis.py   token entropy vs channel capacity
  ├── baselines.py              Huffman + convolutional code + 64-QAM (BER/BLER)
  ├── rate_distortion.py        semantic rate–distortion aggregation table + figure
  ├── shannon_ter_bound.py      information-theoretic TER lower bound (converse)
  ├── theory_ter.py             constellation union-bound TER analysis
  └── plot_results.py           render BLEU / semantic / baseline curves
configs/default.yaml     default hyper-parameters (reference)
results/                 measured metrics (JSON, tracked)
docs/CHECKPOINTS.md      checkpoint loading and inference snippet
```

---

## Data preparation

### Stage-1 (BooksCorpus, alignment pre-training)

Download BooksCorpus (or any large English plain-text corpus) and build
contiguous 384-token blocks used for the alignment pre-training stage:

```bash
python scripts/build_stage1_data.py
```

The script is hard-coded to read from `/root/autodl-tmp/RoBERTa/books_large_p1.txt`
and `/root/autodl-tmp/RoBERTa/books_large_p2.txt` by default; edit the
`SRCS` list and the `OUT` path inside the script for your own data locations.

### Stage-2 (Europarl, semantic fine-tuning)

Download **Europarl v7 English** (plain text, `txt/en/`) from
<https://www.statmt.org/europarl/> and build the tokenised 10/90 val/train
splits:

```bash
python scripts/preprocess_europarl.py \
    --data_dir /path/to/europarl/txt/en \
    --model_path roberta-base --out_dir data
```

Produces `data/Eurp_{sentences,tokens}_robert_{train,val}.pkl`.

---

## Pre-trained checkpoints

Two checkpoints are provided, corresponding to the two symbol budgets
evaluated in the paper.

| Checkpoint | `c_in` | Complex sym./token | `s2e_depth` | Eval flags | Notes |
|---|---|---|---|---|---|
| `runs/beats4/model_best.pt` (484 MB) | 16 | **8** | 3 | `--c_in 16 --s2e_depth 3` | Strong recipe (deep S2E, ultra-low SNR); operating point in the revised paper |
| `model_00700.pt` (1.5 GB, legacy) | 32 | **16** | 2 | `--c_in 32 --s2e_depth 2` | Original two-stage recipe; reproduces original submission numbers |

> **Backward compatibility.**  The loader (`load_roberta_sc`) automatically
> detects legacy checkpoints that use the original `linear0`/`linear1` S2E
> architecture and adapts the model (remaps keys, disables intermediate
> activations) so that both checkpoints work with the **same** `evaluate.py`
> script.  You only need to pass the correct `--c_in` and `--s2e_depth`.

**Loading for inference:**

```python
from roberta_sc.model import load_roberta_sc

# 8 complex symbols/token (new)
m8 = load_roberta_sc("runs/beats4/model_best_cin16.pt",
                     model_path="roberta-base", c_in=16, s2e_depth=3, device="cuda")

# 16 complex symbols/token (legacy)
m16 = load_roberta_sc("model_best_cin32.pt",
                      model_path="roberta-base", c_in=32, s2e_depth=2, device="cuda")
```

See [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md) for public download URLs,
md5 checksums, and an end-to-end inference snippet.

---

## Training from scratch

The released checkpoint was obtained with a two-stage recipe.  The key
hyper-parameters that control the training are summarised below.

### Key training parameters

| Parameter | Meaning | Final value |
|---|---|---|
| `--c_in` | Real channel dims per token (= 2 × complex symbols) | 16 (8 complex/token) |
| `--s2e_depth` | S2E MLP depth (2 = original, 3 = deeper for small `c_in`) | 3 |
| `--snr_min / --snr_max` | SNR range (dB) sampled per example during training | –5 / 15 |
| `--snr_skew` | Power-law skew: >1 biases the SNR sampler toward the low end | 5.0 (Stage-1), 3.0 (Stage-2) |
| `--repel_margin / --repel_weight` | Codeword repulsion loss (hinge margin / scalar weight) | 1.0 / 0.02 (Stage-1), 0.5 / 0.01 (Stage-2) |
| `--clean_snr_prob / --clean_freq` | Anti-floor: every N steps, one batch at SNR = 30 dB | 0.03 / 60 (Stage-2) |
| `--init_from` | Warm-start decoder & backbone from a previous checkpoint | Stage-1 ckpt (Stage-2) |

### Stage 1 — Alignment pre-training (BooksCorpus, 5000 steps)

Stage 1 teaches a robust per-token I/Q alignment on a large, diverse corpus.
Sequences are kept short (`max_len = 192`) because the per-token I/Q mapping
does not require long-range context; this allows a large batch size.

```bash
python scripts/train.py --model_path roberta-base \
    --train ~/autodl-tmp/stage1_books_384.pkl \
    --val   ../src/data/Eurp_tokens_robert_val.pkl \
    --channel rayleigh --c_in 32  \
    --batch_size 100 --grad_accum 2 --max_len 512 \
    --max_steps 3000 --warmup 25 --lr 1.5e-4 \
    --snr_skew 5.0 --snr_min 0 --snr_max 20 \
    --repel_margin 1.0 --repel_weight 0.02 \
    --out_dir runs/stage1
```

**What happens in Stage 1:**  for each batch of contiguous-token blocks, the
model encodes each token into `c_in/2` complex symbols, passes them through a
differentiable Rayleigh channel with a randomly sampled SNR (heavily skewed
toward the low end via `snr_skew`), demaps them back through the S2E MLP, and
reconstructs the original tokens via the frozen backbone + output head.  The
repulsion loss (`repel_margin/repel_weight`) explicitly penalises pairs of
codewords that are closer than `repel_margin` in Euclidean distance, pushing
the learned constellation apart.

### Stage 2 — Semantic fine-tuning (Europarl, 4000 steps)

Stage 2 warm-starts the decoder (S2E, frozen backbone, output head) from the
Stage-1 checkpoint and fine-tunes on the downstream Europarl corpus with a
longer sequence length and an ultra-low SNR curriculum to maximise 0 dB
robustness.

```bash
python scripts/train.py --model_path roberta-base \
    --train ../src/data/Eurp_tokens_robert_train.pkl \
    --val   ../src/data/Eurp_tokens_robert_val.pkl \
    --channel rayleigh --c_in 32 --s2e_depth 3 \
    --batch_size 128 --grad_accum 2 --max_len 512 \
    --max_steps 1000 --warmup 80 --lr 1e-4 \
    --snr_min 0 --snr_max 18 --snr_skew 3.0 \
    --clean_snr_prob 0.03 --clean_freq 60 \
    --repel_margin 0.5 --repel_weight 0.01 \
    --init_from runs/stage1/model_01100.pt \
    --out_dir runs/stage2
```

python scripts/train.py --model_path roberta-base \
    --train ../src/data/Eurp_tokens_robert_train.pkl \
    --val   ../src/data/Eurp_tokens_robert_val.pkl \
    --channel rayleigh --c_in 32 \
    --batch_size 64 --grad_accum 2 --max_len 512 \
    --max_steps 3000 --warmup 80 --lr 1e-4 \
    --snr_min 0 --snr_max 18 --snr_skew 3.0 \
    --clean_snr_prob 0.03 --clean_freq 60 \
    --repel_margin 0.5 --repel_weight 0.01 \
    --init_from runs/stage1/model_best.pt \
    --out_dir runs/stage2

**What the extra parameters do in Stage 2:**

* `--snr_min -5 --snr_max 15`: the SNR range is shifted below 0 dB so the model
  learns to cope with extreme noise; the skew heavily weights the low end.
* `--clean_snr_prob 0.03 --clean_freq 60`: every 60 optimizer steps, one batch
  is trained at SNR = 30 dB (i.e. near-clean).  This prevents the S2E from
  "forgetting" how to map clean symbols and eliminates the high-SNR error floor.
* `--repel_margin 0.5 --repel_weight 0.01`: a gentler repulsion in Stage 2 to
  fine-tune the codeword spacing without disrupting the semantic alignment.

### Training a different symbol rate

The number of complex symbols per token is set by `--c_in` (`c_in/2` complex
symbols).  To explore the semantic rate–distortion frontier, train models at
several values (e.g. `--c_in 8, 16, 20, 24, 32`) and aggregate:

```bash
# Train models at different c_in (same recipe)
for CIN in 8 16 32; do
  python scripts/train.py ... --c_in $CIN ...
done

# Aggregate and plot rate–distortion
python scripts/rate_distortion.py
```

---

## Reproducing the paper results

The same scripts reproduce both symbol budgets.  The only difference is the
checkpoint path and the `--c_in` / `--s2e_depth` flags shown in the table
above.

```bash
VAL=../src/data/Eurp_sentences_robert_val.pkl       # (produced by preprocess_europarl.py)
MODEL=roberta-base
BERT=bert-base-uncased
```

### 8 complex symbols per token (`c_in = 16`, revised paper)

```bash
CKPT=runs/stage2/model_best.pt

python scripts/evaluate.py --checkpoint $CKPT --model_path $MODEL \
    --val $VAL --channel awgn  --c_in 16 --s2e_depth 3
python scripts/evaluate.py --checkpoint $CKPT --model_path $MODEL \
    --val $VAL --channel rayleigh --c_in 32 --s2e_depth 3
python scripts/semantic_eval.py --checkpoint $CKPT --model_path $MODEL \
    --bert_path $BERT --val $VAL --channel awgn --c_in 16 --s2e_depth 3
python scripts/complexity.py --checkpoint $CKPT --model_path $MODEL \
    --c_in 16 --s2e_depth 3
```

### 16 complex symbols per token (`c_in = 32`, original submission)

```bash
CKPT=model_00700.pt

python scripts/evaluate.py --checkpoint $CKPT --model_path $MODEL \
    --val $VAL --channel awgn  --c_in 32 --s2e_depth 2
python scripts/evaluate.py --checkpoint $CKPT --model_path $MODEL \
    --val $VAL --channel rayleigh --c_in 32 --s2e_depth 2
python scripts/semantic_eval.py --checkpoint $CKPT --model_path $MODEL \
    --bert_path $BERT --val $VAL --channel awgn --c_in 32 --s2e_depth 2
python scripts/complexity.py --checkpoint $CKPT --model_path $MODEL \
    --c_in 32 --s2e_depth 2
```

### Shared (c_in-independent) analyses

```bash
python scripts/information_analysis.py --model_path $MODEL --val $VAL
python scripts/baselines.py --model_path $MODEL --val $VAL --code conv
python scripts/rate_distortion.py
python scripts/shannon_ter_bound.py
```

---

## Headline results (reproduced by this code, `c_in = 16`, `s2e_depth = 3`)

| Metric | AWGN,  0 dB | AWGN, 12 dB | Rayleigh,  0 dB | Rayleigh, 12 dB |
|---|---|---|---|---|
| BLEU-1 | 0.67 | 0.96 | 0.52 | 0.90 |
| BLEU-4 | 0.27 | 0.91 | 0.09 | 0.74 |
| BERTScore-F1 | 0.92 | 0.99 | 0.88 | 0.98 |
| Sim-BERT  | 0.70 | 0.97 | 0.57 | 0.93 |
| Token error rate | 0.29 | 0.03 | 0.47 | 0.07 |

| Complexity (for a 512-token sentence) | Value |
|---|---|
| Total parameters | 126.9 M |
| Trainable parameters | 41.5 M (32.7 %) |
| **Transmitter parameters (edge device)** | **1.61 M (≈ 6 MB)** |
| Receiver FLOPs | 275 GFLOPs |
| Transmitter latency (H20 GPU / CPU) | 0.10 ms / 0.09 ms |
| Receiver latency (H20 GPU) | 8.4 ms |
| Peak GPU memory (inference) | 0.62 GB |

The resource-constrained edge device only runs the **transmitter** — a 1.6 M-
parameter embedding look-up (no transformer) — while the heavy LLM cost is
borne by the **receiver** (infrastructure side).

---

## Citation

```bibtex
@article{wang2026robertasc,
  title   = {RoBERTa-SC: {LLM}-Powered Semantic Communication via
             Efficient Signal-Text Alignment},
  author  = {Wang, Zhenyi and Liu, Jiaqi and Liao, Feifan and Zou, Li and
             Li, Kai and Mi, Haibo and Wei, Shengyun and Lai, Rongxuan},
  journal = {Physical Communication},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
