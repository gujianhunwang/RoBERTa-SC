# Pre-trained checkpoints

Two checkpoints are provided, corresponding to the two symbol budgets
evaluated in the paper.

| Checkpoint | Backbone | `c_in` | Complex symbols / token | `s2e_depth` | Training |
|---|---|---|---|---|---|
| `runs/beats4/model_best.pt` (484 MB) | roberta-base | 16 | **8** | 3 | Strong recipe: Stage-1 BooksCorpus (repulsion, `snr_skew=5.0`) + Stage-2 Europarl (ultra-low SNR –5..+15 dB, clean anti-floor) |
| `model_00700.pt` (1.5 GB, legacy) | roberta-base | 32 | **16** | 2 | Original two-stage: BooksCorpus + Wikipedia → Europarl fine-tune |

## Loading

```python
from roberta_sc.model import load_roberta_sc

# 8 complex symbols / token
model = load_roberta_sc("runs/beats4/model_best.pt",
                        model_path="roberta-base", c_in=16, s2e_depth=3, device="cuda")

# 16 complex symbols / token (legacy — auto-detected)
model = load_roberta_sc("model_00700.pt",
                        model_path="roberta-base", c_in=32, s2e_depth=2, device="cuda")
```

The loader automatically detects legacy checkpoints (those with
`linear0`/`linear1` S2E keys) and adapts the architecture accordingly.
No manual flags are needed beyond `--c_in` and `--s2e_depth`.

## Inference snippet

```python
from roberta_sc.model import Transmitter, Receiver
from roberta_sc.channel import apply_channel
from transformers import RobertaTokenizer

# 8 cplx/token
model = load_roberta_sc("runs/beats4/model_best.pt",
                        model_path="roberta-base", c_in=16, s2e_depth=3, device="cuda")
tx, rx = Transmitter(model).eval().to("cuda"), Receiver(model).eval().to("cuda")
tok = RobertaTokenizer.from_pretrained("roberta-base")

ids = torch.tensor(tok.encode("Hello, world!")).unsqueeze(0).to("cuda")
logits = rx(apply_channel(tx(ids), snr=12, channel="awgn"))
recovered = tok.decode(torch.argmax(logits, -1)[0][1:-1])
print(recovered)
```

## Expected results

| Checkpoint | Channel | 0 dB BLEU-1 | 6 dB BLEU-1 | 18 dB BLEU-1 |
|---|---|---|---|---|
| `beats4` (8 cplx/token) | AWGN | 0.67 | 0.93 | 0.96 |
| `beats4` (8 cplx/token) | Rayleigh | 0.52 | 0.78 | 0.95 |
| `model_00700` (16 cplx/token) | AWGN | 0.93 | 0.97 | 0.98 |
| `model_00700` (16 cplx/token) | Rayleigh | 0.85 | 0.92 | 0.96 |

> Replace the placeholder URLs below with the public GitHub Release / Zenodo
> links and md5 checksums before submission.

- `runs/beats4/model_best.pt`:
  - URL : `https://pan.baidu.com/s/1brTdMRlzO12hGSZTJQ00_A?pwd=amct`    提取码: amct
- `model_00700.pt` (legacy):
  - URL : `model_best_cin32.pt 链接: https://pan.baidu.com/s/1Zgtxabd-kMjjLMBtiNu-2Q`   提取码: 3tnt
