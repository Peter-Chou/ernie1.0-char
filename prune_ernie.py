# -*- coding: utf-8 -*-

from models.utils.prune_vocab_embedding import prune_ernie_vocab_model_embedding
from models.tokenizers.bert_char_tokenizer import BertCharTokenizer


def main():
  tokenizer = BertCharTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
  prune_ernie_vocab_model_embedding(tokenizer, "nghuyong/ernie-1.0-base-zh",
                                    "./resources/char-ernie1.0")


if __name__ == '__main__':
  main()
