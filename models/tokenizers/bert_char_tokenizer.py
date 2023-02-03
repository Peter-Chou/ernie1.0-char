# -*- coding: utf-8 -*-

import os
import collections
import unicodedata

from typing import List, Optional, Tuple
from transformers.models.bert.tokenization_bert import load_vocab
from transformers.utils import logging

from transformers.tokenization_utils import (
    PreTrainedTokenizer,
    _is_control,
    _is_punctuation,
    _is_whitespace,
)

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class BertCharTokenizer(PreTrainedTokenizer):
  r"""
  Construct a BERT Character tokenizer.
  This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
  this superclass for more information regarding those methods.
  Args:
      vocab_file (`str`):
          File containing the vocabulary.
      do_lower_case (`bool`, *optional*, defaults to `True`):
          Whether or not to lowercase the input when tokenizing.
      never_split (`Iterable`, *optional*):
          Collection of tokens which will never be split during tokenization.
      unk_token (`str`, *optional*, defaults to `"[UNK]"`):
          The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
          token instead.
      sep_token (`str`, *optional*, defaults to `"[SEP]"`):
          The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
          sequence classification or for a text and a question for question answering. It is also used as the last
          token of a sequence built with special tokens.
      pad_token (`str`, *optional*, defaults to `"[PAD]"`):
          The token used for padding, for example when batching sequences of different lengths.
      cls_token (`str`, *optional*, defaults to `"[CLS]"`):
          The classifier token which is used when doing sequence classification (classification of the whole sequence
          instead of per-token classification). It is the first token of the sequence when built with special tokens.
      mask_token (`str`, *optional*, defaults to `"[MASK]"`):
          The token used for masking values. This is the token used when training this model with masked language
          modeling. This is the token which the model will try to predict.
      tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
          Whether or not to tokenize Chinese characters.
          This should likely be deactivated for Japanese (see this
          [issue](https://github.com/huggingface/transformers/issues/328)).
      strip_accents (`bool`, *optional*):
          Whether or not to strip all accents. If this option is not specified, then it will be determined by the
          value for `lowercase` (as in the original BERT).
  """
  vocab_files_names = VOCAB_FILES_NAMES

  def __init__(self,
               vocab_file,
               do_lower_case=True,
               never_split=None,
               unk_token="[UNK]",
               sep_token="[SEP]",
               pad_token="[PAD]",
               cls_token="[CLS]",
               mask_token="[MASK]",
               strip_accents=None,
               **kwargs):
    super().__init__(
        do_lower_case=do_lower_case,
        never_split=never_split,
        unk_token=unk_token,
        sep_token=sep_token,
        pad_token=pad_token,
        cls_token=cls_token,
        mask_token=mask_token,
        strip_accents=strip_accents,
        **kwargs,
    )
    if not os.path.isfile(vocab_file):
      raise ValueError(
          f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
          " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
      )
    self.vocab = load_vocab(vocab_file)
    self.ids_to_tokens = collections.OrderedDict([
        (ids, tok) for tok, ids in self.vocab.items()
    ])
    self.char_tokenizer = CharTokenizer(
        do_lower_case=do_lower_case,
        never_split=never_split,
        strip_accents=strip_accents,
    )

  @property
  def do_lower_case(self):
    return self.char_tokenizer.do_lower_case

  @property
  def vocab_size(self):
    return len(self.vocab)

  def get_vocab(self):
    return dict(self.vocab, **self.added_tokens_encoder)

  def _tokenize(self, text):
    return self.char_tokenizer.tokenize(text,
                                        never_split=self.all_special_tokens)

  def _convert_token_to_id(self, token):
    """Converts a token (str) in an id using the vocab."""
    return self.vocab.get(token, self.vocab.get(self.unk_token))

  def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (str) using the vocab."""
    return self.ids_to_tokens.get(index, self.unk_token)

  def convert_tokens_to_string(self, tokens):
    """Converts a sequence of tokens (string) in a single string."""
    out_string = "".join(tokens).strip()
    return out_string

  def build_inputs_with_special_tokens(
      self,
      token_ids_0: List[int],
      token_ids_1: Optional[List[int]] = None) -> List[int]:
    """
      Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
      adding special tokens. A BERT sequence has the following format:
      - single sequence: `[CLS] X [SEP]`
      - pair of sequences: `[CLS] A [SEP] B [SEP]`
      Args:
          token_ids_0 (`List[int]`):
              List of IDs to which the special tokens will be added.
          token_ids_1 (`List[int]`, *optional*):
              Optional second list of IDs for sequence pairs.
      Returns:
          `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
      """
    if token_ids_1 is None:
      return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
    cls = [self.cls_token_id]
    sep = [self.sep_token_id]
    return cls + token_ids_0 + sep + token_ids_1 + sep

  def get_special_tokens_mask(
      self,
      token_ids_0: List[int],
      token_ids_1: Optional[List[int]] = None,
      already_has_special_tokens: bool = False) -> List[int]:
    """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

    if already_has_special_tokens:
      return super().get_special_tokens_mask(token_ids_0=token_ids_0,
                                             token_ids_1=token_ids_1,
                                             already_has_special_tokens=True)

    if token_ids_1 is not None:
      return [1] + ([0] * len(token_ids_0)) + [1] + ([0] *
                                                     len(token_ids_1)) + [1]
    return [1] + ([0] * len(token_ids_0)) + [1]

  def create_token_type_ids_from_sequences(
      self,
      token_ids_0: List[int],
      token_ids_1: Optional[List[int]] = None) -> List[int]:
    """
          Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
          pair mask has the following format:
          ```
          0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
          | first sequence    | second sequence |
          ```
          If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
          Args:
              token_ids_0 (`List[int]`):
                  List of IDs.
              token_ids_1 (`List[int]`, *optional*):
                  Optional second list of IDs for sequence pairs.
          Returns:
              `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
          """
    sep = [self.sep_token_id]
    cls = [self.cls_token_id]
    if token_ids_1 is None:
      return len(cls + token_ids_0 + sep) * [0]
    return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

  def save_vocabulary(self,
                      save_directory: str,
                      filename_prefix: Optional[str] = None) -> Tuple[str]:
    index = 0
    if os.path.isdir(save_directory):
      vocab_file = os.path.join(
          save_directory, (filename_prefix + "-" if filename_prefix else "") +
          VOCAB_FILES_NAMES["vocab_file"])
    else:
      vocab_file = (filename_prefix +
                    "-" if filename_prefix else "") + save_directory
    with open(vocab_file, "w", encoding="utf-8") as writer:
      for token, token_index in sorted(self.vocab.items(),
                                       key=lambda kv: kv[1]):
        if index != token_index:
          logger.warning(
              f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
              " Please check that the vocabulary is not corrupted!")
          index = token_index
        writer.write(token + "\n")
        index += 1
    return (vocab_file,)


class CharTokenizer(object):

  def __init__(self, do_lower_case=True, never_split=None, strip_accents=None):
    """
    Constructs a CharTokenizer that will run character tokenization (punctuation splitting, lower casing, etc.).
    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization.
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """
    if never_split is None:
      never_split = []
    self.do_lower_case = do_lower_case
    self.never_split = set(never_split)
    self.strip_accents = strip_accents

  def tokenize(self, text, never_split=None):
    """
    Character Tokenization of a piece of text. Split by char and ignore whitespaces.
    Args:
        never_split (`List[str]`, *optional*)
            Kept for backward compatibility purposes. Now implemented directly at the base class level (see
            [`PreTrainedTokenizer.tokenize`]) List of token not to split.
    """
    never_split = self.never_split.union(
        set(never_split)) if never_split else self.never_split
    text = self._clean_text(text)
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if token not in never_split:
        if self.do_lower_case:
          token = token.lower()
          if self.strip_accents is not False:
            token = self._run_strip_accents(token)
        elif self.strip_accents:
          token = self._run_strip_accents(token)
        # split token to chars
        split_tokens.extend(self._tokenize_chars(token))
      else:
        # add origin never split token
        split_tokens.append(token)
    return split_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":  # accents category
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text, never_split=None):
    """Splits punctuation on a piece of text."""
    if never_split is not None and text in never_split:
      return [text]
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chars(self, text):
    output = []
    for char in text:
      output.append(char)
    return output

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xFFFD or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)
