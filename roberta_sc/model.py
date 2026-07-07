"""RoBERTa-SC model definition.

RoBERTa-SC turns a frozen, pre-trained RoBERTa into a semantic-aware joint
source--channel codec that operates directly on physical-layer I/Q symbols.
Only three lightweight components are trained:

* ``IQ_Embedding`` -- replaces RoBERTa's word-embedding table; maps each token
  to ``c_in`` real values, i.e. ``c_in / 2`` **complex symbols per token**, and
  (during training) passes them through a differentiable channel before the
  ``linear0``/``linear1`` Signal-to-Embedding (S2E) projection back to the
  768-d hidden space;
* the S2E projection (``linear0``, ``linear1``) inside ``IQ_Embedding``;
* the output ``lm_head``.

The 12 transformer blocks of RoBERTa-base remain **frozen**, preserving their
linguistic prior and avoiding catastrophic forgetting.

Important clarification (see the manuscript, Sec. 2): the I/Q embedding is a
per-token look-up table, so a sentence of ``L`` tokens is transmitted with
``L * (c_in / 2)`` complex symbols -- i.e. ``c_in / 2 = 16`` complex symbols
**per token**, not 16 symbols for the whole sentence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaForMaskedLM

from .channel import apply_channel

VOCAB_SIZE = 50265  # RoBERTa-base vocabulary


class IQ_Embedding(nn.Module):
    """Trainable I/Q embedding + differentiable channel + S2E projection.

    Parameters
    ----------
    c_in : int
        Number of real channel dimensions per token. The number of complex
        symbols per token is ``c_in // 2`` (default 16, i.e. ``c_in = 32``).
    d_model : int
        Hidden size of the frozen backbone (768 for RoBERTa-base).
    channel : str
        Channel used during training (``"awgn"`` or ``"rayleigh"``).
    train_snr_range : tuple(float, float)
        Range (dB) from which a per-example SNR is sampled during training.
    """

    def __init__(self, c_in: int = 32, d_model: int = 768,
                 channel: str = "rayleigh", train_snr_range=(0.0, 20.0),
                 train_snr_skew: float = 1.0, s2e_depth: int = 2,
                 s2e_hidden: int = None, repel_margin: float = 0.0,
                 repel_weight: float = 0.0, s2e_activation: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, c_in)
        self.c_in = c_in
        self.channel = channel
        self.train_snr_range = train_snr_range
        self.train_snr_skew = train_snr_skew
        self.repel_margin = repel_margin
        self.repel_weight = repel_weight
        self._repel_loss = None
        # Build S2E MLP.  When s2e_activation=False (legacy mode), the
        # layers are pure linear stacks without activation, matching the
        # original linear0→linear1 architecture.
        layers = [nn.Linear(c_in, s2e_hidden or d_model)]
        for i in range(s2e_depth - 1):
            if s2e_activation:
                layers.append(nn.GELU())
            layers.append(nn.Linear(s2e_hidden or d_model, s2e_hidden or d_model))
        if len(layers) == 1 and s2e_activation:
            layers.append(nn.GELU())  # at least one GELU when activation is on
        self.s2e = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

        for m in self.s2e:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    @property
    def n_complex_symbols(self) -> int:
        return self.c_in // 2

    def modulate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> power-normalised complex symbols, shape ``(B, L, c_in/2)``."""
        x = self.embedding(input_ids)
        c = self.c_in // 2
        signal = torch.complex(x[:, :, :c], x[:, :, c:])
        # Normalise to unit average symbol power (per complex dimension, over L).
        power = torch.mean(torch.abs(signal) ** 2, dim=1, keepdim=True)
        signal = signal * torch.sqrt(1.0 / (power + 1e-8))

        # ---- repulsion loss: penalise close codewords in the current batch ----
        if self.training and self.repel_margin > 0:
            tokens = input_ids.unique()
            if len(tokens) > 1:
                ew = self.embedding(tokens)           # (U, c_in*2) real
                cs = torch.complex(ew[:, :c], ew[:, c:])
                cp = (cs.abs()**2).mean(dim=1, keepdim=True)
                cs = cs * torch.sqrt(1.0/(cp+1e-8))
                cr = torch.cat([cs.real, cs.imag], dim=-1)  # (U, 2c)
                d = torch.cdist(cr, cr)               # (U, U) Euclidean
                mask = ~torch.eye(len(tokens), dtype=torch.bool, device=d.device)
                if mask.sum() > 0:
                    self._repel_loss = torch.relu(self.repel_margin - d[mask]).mean()
                else:
                    self._repel_loss = torch.tensor(0.0, device=signal.device, dtype=signal.real.dtype)
            else:
                self._repel_loss = torch.tensor(0.0, device=signal.device, dtype=signal.real.dtype)

        return signal

    def demodulate(self, signal: torch.Tensor) -> torch.Tensor:
        """Complex symbols -> 768-d embedding via the S2E projection."""
        x = torch.cat([signal.real, signal.imag], dim=-1)
        if x.dtype != torch.float32:
            x = x.float()
        return self.s2e(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        signal = self.modulate(input_ids)
        if self.training:
            B = signal.shape[0]
            lo, hi = self.train_snr_range
            u = torch.rand(B, 1, device=signal.device) ** self.train_snr_skew
            snr_db = lo + (hi - lo) * u
            signal = apply_channel(signal, snr_db, self.channel)
        return self.demodulate(signal)


class RobertaSC(nn.Module):
    """Frozen RoBERTa backbone with a trainable I/Q embedding and output head."""

    def __init__(self, model_path: str = "roberta-base", c_in: int = 32,
                 hidden_size: int = 768, channel: str = "rayleigh",
                 train_snr_range=(0.0, 20.0), train_snr_skew: float = 1.0,
                 s2e_depth: int = 2, repel_margin: float = 0.0,
                 repel_weight: float = 0.0, s2e_activation: bool = True):
        super().__init__()
        config = RobertaConfig.from_pretrained(model_path)
        config.hidden_size = hidden_size
        self.bert = RobertaForMaskedLM.from_pretrained(model_path, config=config)
        self.bert.roberta.embeddings.word_embeddings = IQ_Embedding(
            c_in, hidden_size, channel=channel, train_snr_range=train_snr_range,
            train_snr_skew=train_snr_skew, s2e_depth=s2e_depth,
            repel_margin=repel_margin, repel_weight=repel_weight,
            s2e_activation=s2e_activation)
        self.c_in = c_in
        self.repel_weight = repel_weight
        self.freeze_backbone()

    def get_repel_loss(self):
        iq = self.bert.roberta.embeddings.word_embeddings
        if iq._repel_loss is not None:
            v = iq._repel_loss; iq._repel_loss = None
            return v
        return torch.tensor(0.0)

    def freeze_backbone(self):
        """Train only the I/Q embedding (incl. S2E) and the output head."""
        for name, param in self.bert.named_parameters():
            param.requires_grad = ("word_embeddings" in name) or ("lm_head" in name)

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable, "trainable_ratio": trainable / total}

    def configure_optimizers(self, learning_rate, weight_decay, device_type="cuda"):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
        use_fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95),
                                 eps=1e-7, fused=use_fused)

    def forward(self, input_ids, attention_mask=None):
        """End-to-end training forward (channel applied inside the embedding).

        The skip connection ``encoder(x) + x`` injects the signal-fidelity term
        alongside the transformer's semantic-prior term (manuscript Eq. 2).
        """
        x = self.bert.roberta.embeddings(input_ids)
        x = self.bert.roberta.encoder(x).last_hidden_state + x
        return self.bert.lm_head(x)


# --------------------------------------------------------------------------- #
# Deployment view: explicit transmitter (Encoder) and receiver (Decoder).     #
# --------------------------------------------------------------------------- #
class Transmitter(nn.Module):
    """Semantic modulator: token ids -> complex I/Q symbols ``(B, L, c_in/2)``."""

    def __init__(self, model: RobertaSC):
        super().__init__()
        self.iq = model.bert.roberta.embeddings.word_embeddings

    @torch.no_grad()
    def forward(self, input_ids):
        return self.iq.modulate(input_ids)


class Receiver(nn.Module):
    """Semantic demodulator: received complex symbols -> token logits."""

    def __init__(self, model: RobertaSC):
        super().__init__()
        self.model = model
        emb = model.bert.roberta.embeddings
        self.position_embeddings = emb.position_embeddings
        self.token_type_embeddings = emb.token_type_embeddings
        self.layernorm = emb.LayerNorm
        self.dropout = emb.dropout
        self.iq = emb.word_embeddings
        self.encoder = model.bert.roberta.encoder
        self.head = model.bert.lm_head

    @torch.no_grad()
    def forward(self, signal):
        L = signal.size(1)
        pos_ids = torch.arange(L, device=signal.device).unsqueeze(0).expand(signal.size(0), L)
        type_ids = torch.zeros(signal.size(0), L, dtype=torch.long, device=signal.device)
        x = self.iq.demodulate(signal)
        x = x + self.position_embeddings(pos_ids) + self.token_type_embeddings(type_ids)
        x = self.dropout(self.layernorm(x))
        x = self.encoder(x).last_hidden_state + x
        return self.head(x)


def load_roberta_sc(checkpoint_path: str, model_path: str = "roberta-base",
                    c_in: int = 32, device: str = "cuda",
                    s2e_depth: int = 2) -> RobertaSC:
    """Build a RobertaSC and load weights from a training checkpoint.

    Backward-compatible: old checkpoints that use ``linear0``/``linear1``
    (legacy S2E) are automatically detected; keys are remapped to
    ``s2e.0``/``s2e.2`` and the S2E is built without intermediate GELU
    to exactly match the original pure-linear data-path.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # Auto-detect legacy format:  linear0/linear1 → pure-linear S2E
    wemb = "bert.roberta.embeddings.word_embeddings."
    remapped = dict(state)
    legacy = f"{wemb}linear0.weight" in remapped and f"{wemb}s2e.0.weight" not in remapped
    s2e_act = not legacy  # no GELUs in legacy mode

    model = RobertaSC(model_path=model_path, c_in=c_in, s2e_depth=s2e_depth,
                      s2e_activation=s2e_act)

    if legacy:
        for old, new in [("linear0.weight", "s2e.0.weight"),
                         ("linear0.bias", "s2e.0.bias"),
                         ("linear1.weight", "s2e.1.weight"),
                         ("linear1.bias", "s2e.1.bias")]:
            if wemb + old in remapped:
                remapped[wemb + new] = remapped.pop(wemb + old)
        print(f"load_roberta_sc: legacy checkpoint detected, "
              f"remapped linear0/1→s2e.0/1 (no activations)", flush=True)

    own = model.state_dict()
    matched = {k: v for k, v in remapped.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    return model.eval().to(device)
