"""Evaluation metrics for RoBERTa-SC.

Two families are provided:

* **Lexical** -- BLEU-1..4 (n-gram overlap), following the protocol used by
  DeepSC and the original RoBERTa-SC experiments.
* **Semantic** -- sentence-embedding cosine similarity.  We make the metric
  explicit (it was under-specified in the first submission): by default we use
  a frozen ``bert-base-uncased`` encoder, mean-pool the last hidden states, and
  compute cosine similarity (this exactly matches the original code).  An
  optional Sentence-BERT (``all-MiniLM-L6-v2``) variant and BERTScore are also
  provided for the multi-dimensional semantic evaluation requested in review.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


# --------------------------------------------------------------------------- #
# Lexical metric                                                              #
# --------------------------------------------------------------------------- #
def bleu_scores(reference: str, hypothesis: str, smooth: bool = False):
    """Return (BLEU-1, BLEU-2, BLEU-3, BLEU-4).

    ``reference`` is the recovered sentence and ``hypothesis`` the transmitted
    one, matching the convention of the original experiments and DeepSC.
    """
    sf = SmoothingFunction().method1 if smooth else None
    ref, hyp = [reference.split()], hypothesis.split()
    weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    return tuple(sentence_bleu(ref, hyp, weights=w, smoothing_function=sf) for w in weights)


# --------------------------------------------------------------------------- #
# Semantic metrics                                                            #
# --------------------------------------------------------------------------- #
class SentenceSimilarity:
    """Cosine similarity between mean-pooled BERT sentence embeddings.

    This is the *exact* metric used to produce Fig. 3 of the manuscript:
    a frozen ``bert-base-uncased`` encoder, last-hidden-state mean pooling,
    per-feature max-normalisation, then cosine similarity.
    """

    def __init__(self, bert_path: str = "bert-base-uncased", device: str = "cpu"):
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model = BertModel.from_pretrained(bert_path).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, references: List[str], predictions: List[str]) -> List[float]:
        from sklearn.preprocessing import normalize
        from scipy.spatial.distance import cosine

        def embed(sents):
            enc = self.tokenizer(sents, add_special_tokens=True, max_length=256,
                                 padding="max_length", truncation=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc).last_hidden_state.mean(dim=1)
            return normalize(out.cpu().numpy(), axis=0, norm="max")

        v1, v2 = embed(references), embed(predictions)
        return [1 - cosine(a, b) for a, b in zip(v1, v2)]


class SBERTSimilarity:
    """Sentence-BERT cosine similarity (multi-dimensional semantic evaluation)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, references: List[str], predictions: List[str]) -> List[float]:
        from sentence_transformers import util
        e1 = self.model.encode(references, convert_to_tensor=True, normalize_embeddings=True)
        e2 = self.model.encode(predictions, convert_to_tensor=True, normalize_embeddings=True)
        return util.pairwise_cos_sim(e1, e2).cpu().tolist()


def bertscore_f1(references: List[str], predictions: List[str], lang: str = "en") -> List[float]:
    """BERTScore F1 (token-level contextual similarity)."""
    from bert_score import score
    _, _, f1 = score(predictions, references, lang=lang, verbose=False)
    return f1.tolist()


def token_error_rate(reference_ids, recovered_ids) -> float:
    """Fraction of mismatched tokens -- a semantic-system analogue of BER."""
    ref = np.asarray(reference_ids)
    rec = np.asarray(recovered_ids)
    n = min(len(ref), len(rec))
    if n == 0:
        return 1.0
    return float(np.mean(ref[:n] != rec[:n]))
