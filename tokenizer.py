"""
tokenizer.py
============
33-token protein tokenizer aligned with the official ESM-2 vocabulary.

Vocabulary (index -> token):
    0  <cls>
    1  <pad>
    2  <eos>
    3  <unk>
    4  L   5  A   6  G   7  V   8  S   9  E  10  R  11  T
   12  I  13  D  14  P  15  K  16  Q  17  N  18  F  19  Y
   20  M  21  H  22  W  23  C  24  X  25  B  26  U  27  Z
   28  O  29  <mask>  30  -  31  <sep>  32  <extra>
"""

from __future__ import annotations

from typing import List, Set


class ESMTokenizer:
    """Protein sequence tokenizer aligned with the ESM-2 official vocabulary.

    The vocabulary contains 33 tokens: 4 leading special tokens, 25 amino-acid
    / ambiguity characters, and 4 trailing special tokens.  The ordering is
    identical to the one used by the official ``esm`` library so that token IDs
    are directly compatible with pre-trained ESM-2 weights.

    Attributes
    ----------
    vocab_size : int
        Total number of tokens (33).
    cls_token_id : int
        ID of the ``<cls>`` token (0).
    pad_token_id : int
        ID of the ``<pad>`` token (1).
    eos_token_id : int
        ID of the ``<eos>`` token (2).
    unk_token_id : int
        ID of the ``<unk>`` token (3).
    mask_token_id : int
        ID of the ``<mask>`` token (29).

    Examples
    --------
    >>> tok = ESMTokenizer()
    >>> ids = tok.encode("ACGT")
    >>> ids[0] == tok.cls_token_id and ids[-1] == tok.eos_token_id
    True
    >>> tok.decode(ids)
    'ACGT'
    """

    # Official ESM-2 vocabulary in index order.
    _VOCAB: List[str] = [
        "<cls>",    # 0
        "<pad>",    # 1
        "<eos>",    # 2
        "<unk>",    # 3
        "L",        # 4
        "A",        # 5
        "G",        # 6
        "V",        # 7
        "S",        # 8
        "E",        # 9
        "R",        # 10
        "T",        # 11
        "I",        # 12
        "D",        # 13
        "P",        # 14
        "K",        # 15
        "Q",        # 16
        "N",        # 17
        "F",        # 18
        "Y",        # 19
        "M",        # 20
        "H",        # 21
        "W",        # 22
        "C",        # 23
        "X",        # 24
        "B",        # 25
        "U",        # 26
        "Z",        # 27
        "O",        # 28
        "<mask>",   # 29
        "-",        # 30
        "<sep>",    # 31
        "<extra>",  # 32
    ]

    def __init__(self) -> None:
        """Initialise the tokenizer and build internal lookup tables."""
        assert len(self._VOCAB) == 33, "Vocabulary must contain exactly 33 tokens."

        # token -> id
        self._token_to_id: dict = {tok: idx for idx, tok in enumerate(self._VOCAB)}
        # id -> token
        self._id_to_token: dict = {idx: tok for idx, tok in enumerate(self._VOCAB)}

        # Convenience IDs
        self.vocab_size: int = 33
        self.cls_token_id: int = self._token_to_id["<cls>"]    # 0
        self.pad_token_id: int = self._token_to_id["<pad>"]    # 1
        self.eos_token_id: int = self._token_to_id["<eos>"]    # 2
        self.unk_token_id: int = self._token_to_id["<unk>"]    # 3
        self.mask_token_id: int = self._token_to_id["<mask>"]  # 29

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, seq: str) -> List[int]:
        """Encode a protein sequence into a list of token IDs.

        The output is wrapped with ``<cls>`` at position 0 and ``<eos>`` at
        the last position, matching the ESM-2 input format.

        Unknown characters (not present in the vocabulary) are mapped to the
        ``<unk>`` token.

        Parameters
        ----------
        seq : str
            Amino-acid sequence string (e.g. ``"MKTAYIAKQRQISFVKSHFSRQ"``).
            Case-sensitive; use upper-case single-letter codes.

        Returns
        -------
        List[int]
            Token IDs: ``[cls_id, aa_id_1, ..., aa_id_n, eos_id]``.

        Examples
        --------
        >>> tok = ESMTokenizer()
        >>> tok.encode("LA")
        [0, 4, 5, 2]
        """
        aa_ids = [self._token_to_id.get(aa, self.unk_token_id) for aa in seq]
        return [self.cls_token_id] + aa_ids + [self.eos_token_id]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to an amino-acid sequence string.

        Special tokens (``<cls>``, ``<pad>``, ``<eos>``, ``<unk>``,
        ``<mask>``, ``<sep>``, ``<extra>``, and the gap ``-``) are silently
        skipped; only single-letter amino-acid / ambiguity characters are
        included in the output.

        Parameters
        ----------
        ids : List[int]
            Sequence of token IDs as returned by :meth:`encode`.

        Returns
        -------
        str
            Amino-acid sequence string (without any special tokens).

        Examples
        --------
        >>> tok = ESMTokenizer()
        >>> tok.decode([0, 4, 5, 2])
        'LA'
        """
        special_ids = self.get_special_token_ids()
        chars = [
            self._id_to_token[i]
            for i in ids
            if i in self._id_to_token and i not in special_ids
        ]
        return "".join(chars)

    def get_special_token_ids(self) -> Set[int]:
        """Return the set of IDs for all special tokens.

        Special tokens are: ``<cls>``, ``<pad>``, ``<eos>``, ``<unk>``,
        ``<mask>``, ``<sep>``, ``<extra>``, and the gap character ``-``.

        Returns
        -------
        Set[int]
            Set of integer token IDs that correspond to special tokens.

        Examples
        --------
        >>> tok = ESMTokenizer()
        >>> 0 in tok.get_special_token_ids()   # <cls>
        True
        >>> 4 in tok.get_special_token_ids()   # L  (not special)
        False
        """
        special_tokens = {"<cls>", "<pad>", "<eos>", "<unk>", "<mask>", "<sep>", "<extra>", "-"}
        return {self._token_to_id[t] for t in special_tokens if t in self._token_to_id}

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the vocabulary size (33)."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"ESMTokenizer(vocab_size={self.vocab_size})"
