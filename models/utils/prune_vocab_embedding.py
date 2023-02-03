# -*- coding: utf-8 -*-

import collections
import os
import re

import torch

from transformers.models.ernie.modeling_ernie import ErnieForMaskedLM

SPECIAL_TOKENS_PATTERN = re.compile(r"^\[.+\]$")


def prune_ernie_vocab_model_embedding(
    tokenizer,
    model_path,
    save_folder,
):
  vocab_dict = tokenizer.vocab
  model = ErnieForMaskedLM.from_pretrained(model_path)

  model.ernie.config.name_or_path = "peterchou/char-ernie1.0"

  pruned_vocabs, pruned_vocab_indexes = prune_vocab(vocab_dict)
  tokenizer.vocab = _get_vocab_dict_from_list(pruned_vocabs)
  tokenizer.save_pretrained(save_folder)

  embedding = model.ernie.embeddings.word_embeddings

  pruned_embedding_weight = prune_word_embedding(
      embedding,
      pruned_vocab_indexes,
      hidden_size=model.ernie.config.hidden_size,
      padding_idx=model.ernie.config.pad_token_id)

  model.ernie.config.vocab_size = len(pruned_vocabs)
  model.save_pretrained(save_folder)

  state_dict = model.state_dict()
  state_dict[
      "ernie.embeddings.word_embeddings.weight"] = pruned_embedding_weight
  pruned_state_dict = collections.OrderedDict()
  for k, v in state_dict.items():
    if "cls.predictions" not in k:
      pruned_state_dict[k] = v

  torch.save(pruned_state_dict, os.path.join(save_folder, "pytorch_model.bin"))

  # pruned_embedding = prune_word_embedding(
  #     embedding,
  #     pruned_vocab_indexes,
  #     hidden_size=model.ernie.config.hidden_size,
  #     padding_idx=model.ernie.config.pad_token_id)
  # model.ernie.embeddings.word_embeddings = pruned_embedding
  # model.ernie.config.vocab_size = len(pruned_vocabs)
  # model.save_pretrained(save_folder)


def _get_vocab_dict_from_list(vocabs):
  vocab_dict = collections.OrderedDict()
  for idx, token in enumerate(vocabs):
    vocab_dict[token] = idx
  return vocab_dict


def prune_word_embedding(embedding,
                         filtered_indexes,
                         hidden_size=768,
                         padding_idx=0):
  # vocab_size = len(filtered_indexes)
  pruned_embedding_weight = embedding(torch.LongTensor(filtered_indexes))
  return pruned_embedding_weight.detach()
  # pruned_word_embedding = nn.Embedding(vocab_size,
  #                                      hidden_size,
  #                                      padding_idx=padding_idx)
  # pruned_word_embedding.weight = torch.nn.Parameter(pruned_embedding_weight)
  # return prune_word_embedding


def prune_vocab(vocab_dict):
  """
  remove unused token, return rest tokens, token_indexes
  """
  vocabs = []
  vocab_indexes = []
  for token, index in vocab_dict.items():
    if len(token) == 1 or _is_neeeded_special_token(token):
      vocabs.append(token)
      vocab_indexes.append(index)
  return vocabs, vocab_indexes


def _is_neeeded_special_token(token):
  return SPECIAL_TOKENS_PATTERN.search(
      token) is not None and "unused" not in token.lower()
