"""Preprocess the Europarl corpus into tokenised train/val splits.

Reproduces the data pipeline used in the paper:
  * extract & clean sentences (100--500 chars) from the Europarl `.txt` files;
  * tokenise with the RoBERTa tokeniser (max length 512);
  * hold out 10% as an unseen test/val split.

Download Europarl (English) from https://www.statmt.org/europarl/ and point
`--data_dir` at the extracted `txt/en` directory (or any folder of `.txt`).
"""

import argparse, os, sys, random, pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roberta_sc.data import build_sentence_corpus, save_pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder of Europarl .txt files")
    ap.add_argument("--model_path", default="roberta-base")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--min_chars", type=int, default=100)
    ap.add_argument("--max_chars", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import RobertaTokenizer
    tok = RobertaTokenizer.from_pretrained(args.model_path)

    print("building sentence corpus ...")
    sents = build_sentence_corpus(args.data_dir, args.min_chars, args.max_chars)
    random.seed(args.seed); random.shuffle(sents)
    n_val = int(len(sents) * args.val_ratio)
    val, train = sents[:n_val], sents[n_val:]
    print(f"{len(train)} train / {len(val)} val sentences")

    os.makedirs(args.out_dir, exist_ok=True)
    save_pickle(train, os.path.join(args.out_dir, "Eurp_sentences_robert_train.pkl"))
    save_pickle(val, os.path.join(args.out_dir, "Eurp_sentences_robert_val.pkl"))

    def tokenise(sentences):
        return tok(sentences, max_length=args.max_len, truncation=True,
                   padding="max_length", return_special_tokens_mask=True)

    print("tokenising ...")
    save_pickle(dict(tokenise(train)), os.path.join(args.out_dir, "Eurp_tokens_robert_train.pkl"))
    save_pickle(dict(tokenise(val)), os.path.join(args.out_dir, "Eurp_tokens_robert_val.pkl"))
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()
