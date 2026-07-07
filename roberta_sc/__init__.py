"""RoBERTa-SC: LLM-powered semantic communication via efficient signal-text alignment."""

from .model import (IQ_Embedding, RobertaSC, Transmitter, Receiver,
                    load_roberta_sc, VOCAB_SIZE)
from .channel import apply_channel, awgn_channel, rayleigh_channel

__all__ = ["IQ_Embedding", "RobertaSC", "Transmitter", "Receiver",
           "load_roberta_sc", "VOCAB_SIZE",
           "apply_channel", "awgn_channel", "rayleigh_channel"]
__version__ = "1.0.0"
