"""
SDP-based encoders built on the Shortest Dependency Path (SDP).

Three classes:

  SDPEncoder (abstract)
      Shared spaCy parsing, dep-label mapping, SDP extraction, chain
      building, and verbalization.  Not directly instantiable.

  BoWSDPEncoder(SDPEncoder)
      Encodes a sentence as a multi-hot bag-of-words over the dependency
      relation labels found on the SDP.  No transformer required.

  VerbalizedSDPEncoder(SDPEncoder, SentenceEncoder)
      Verbalizes the SDP and encodes the resulting string with a
      HuggingFace transformer via SentenceEncoder.

Dependency parsing is performed with spaCy's ``en_core_web_trf`` model.

Verbalized sentence format::

    Sentence {sentence} | Entity-1: [{e1}] | Entity-2: [{e2}] | Dependency path: {dep_chain}

where ``dep_chain`` uses full dependency-relation names
(e.g., "nominal subject" instead of the abbreviated "nsubj").
"""

from __future__ import annotations

from abc import ABC
from collections import defaultdict, deque
from typing import Optional

import torch

from deepref.encoder.sentence_encoder import SentenceEncoder


# ===========================================================================
# Abstract base
# ===========================================================================

class SDPEncoder(ABC):
    """Abstract base for SDP-based encoders.

    Provides spaCy dependency parsing, dep-label mapping, SDP extraction,
    chain building, and verbalization.  Subclasses decide how to produce
    the final representation from the SDP.

    Args:
        dep_vocab: ordered list of full dependency-relation names used as the
            one-hot vocabulary.  Defaults to an alphabetically sorted list of
            all relation names known to the encoder.
    """

    # ------------------------------------------------------------------
    # Mapping from spaCy's abbreviated dependency labels to full names
    # ------------------------------------------------------------------
    DEP_FULL_NAMES: dict[str, str] = {
        'ROOT':      'root',
        'nsubj':     'nominal subject',
        'nsubjpass': 'passive nominal subject',
        'dobj':      'direct object',
        'obj':       'direct object',
        'iobj':      'indirect object',
        'obl':       'oblique nominal',
        'prep':      'prepositional modifier',
        'pobj':      'object of preposition',
        'attr':      'attribute',
        'amod':      'adjectival modifier',
        'advmod':    'adverbial modifier',
        'compound':  'compound modifier',
        'det':       'determiner',
        'cc':        'coordinating conjunction',
        'conj':      'conjunct',
        'acl':       'clausal modifier of noun',
        'advcl':     'adverbial clause modifier',
        'agent':     'agent',
        'aux':       'auxiliary',
        'auxpass':   'auxiliary (passive)',
        'ccomp':     'clausal complement',
        'xcomp':     'open clausal complement',
        'csubj':     'clausal subject',
        'csubjpass': 'clausal passive subject',
        'mark':      'marker',
        'nummod':    'numeric modifier',
        'relcl':     'relative clause modifier',
        'poss':      'possession modifier',
        'neg':       'negation modifier',
        'parataxis': 'parataxis',
        'appos':     'appositional modifier',
        'dep':       'unclassified dependent',
        'punct':     'punctuation',
        'case':      'case marking',
        'nmod':      'nominal modifier',
        'expl':      'expletive',
        'fixed':     'fixed multiword expression',
        'flat':      'flat multiword expression',
        'dative':    'dative',
        'prt':       'particle',
        'quantmod':  'quantifier phrase modifier',
        'npadvmod':  'noun phrase as adverbial modifier',
        'acomp':     'adjectival complement',
        'oprd':      'object predicate',
        'intj':      'interjection',
        'predet':    'predeterminer',
        'pcomp':     'complement of preposition',
        'meta':      'meta modifier',
        'nn':        'noun compound modifier',
        'possessive': 'possessive modifier',
        'preconj':   'preconjunct',
    }

    def __init__(self, dep_vocab: Optional[list[str]] = None) -> None:
        import spacy
        self.nlp = spacy.load('en_core_web_trf')

        if dep_vocab is None:
            dep_vocab = sorted(set(self.DEP_FULL_NAMES.values()))
        self.dep_vocab: list[str] = dep_vocab
        self.dep_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(dep_vocab)
        }

    # ------------------------------------------------------------------
    # Public API — shared by all subclasses
    # ------------------------------------------------------------------

    def dep_to_full_name(self, dep: str) -> str:
        """Map a spaCy dependency abbreviation to its full English name.

        Returns the label unchanged when it is not in the known mapping.
        """
        return self.DEP_FULL_NAMES.get(dep, dep)

    def extract_sdp(
        self,
        item: dict,
    ) -> list[tuple[str, str | None, str | None]]:
        """Extract the Shortest Dependency Path between the two marked entities.

        Args:
            item: dict with keys:
                  ``'token'`` (list[str]), ``'h'`` and ``'t'`` each
                  ``{'name': str, 'pos': [start, end)}``.
                  ``pos`` indices are into the ``token`` list (exclusive end).

        Returns:
            List of ``(token_text, dep_full_name, direction)`` tuples.

            * The dep_full_name and direction of element ``i`` describe the
              edge **from node i to node i+1**.
            * The **last** element always has ``dep_full_name=None,
              direction=None``.
            * *direction* is ``'UP'`` (child → head) or ``'DOWN'``
              (head → child).

        Raises:
            ValueError: if the entity tokens cannot be found in the parsed
                doc, or if no dependency path exists between the entities.
        """
        tokens  = item['token']
        e1_pos  = item['h']['pos']
        e2_pos  = item['t']['pos']

        sentence = ' '.join(tokens)
        doc = self.nlp(sentence)

        char_offsets = self._compute_char_offsets(tokens)

        e1_char_start = char_offsets[e1_pos[0]][0]
        e1_char_end   = char_offsets[e1_pos[1] - 1][1]
        e2_char_start = char_offsets[e2_pos[0]][0]
        e2_char_end   = char_offsets[e2_pos[1] - 1][1]

        e1_spacy = self._find_spacy_token_indices(doc, e1_char_start, e1_char_end)
        e2_spacy = self._find_spacy_token_indices(doc, e2_char_start, e2_char_end)

        if not e1_spacy or not e2_spacy:
            raise ValueError(
                f"Could not locate entity tokens in the parsed doc. "
                f"e1={item['h']['name']!r}, e2={item['t']['name']!r}"
            )

        path_indices = self._bfs_sdp(doc, e1_spacy, e2_spacy)

        if path_indices is None:
            raise ValueError(
                f"No dependency path between {item['h']['name']!r} "
                f"and {item['t']['name']!r}."
            )

        return [
            (
                doc[idx].text,
                self.dep_to_full_name(dep) if dep is not None else None,
                direction,
            )
            for idx, dep, direction in path_indices
        ]

    def build_dep_chain(
        self,
        path: list[tuple[str, str | None, str | None]],
    ) -> str:
        """Render the SDP as a human-readable arrow chain.

        Example output::

            audits --nominal subject--> were --prepositional modifier-->
            about --object of preposition--> waste

        Args:
            path: output of :meth:`extract_sdp`.

        Returns:
            Single-line string.
        """
        parts: list[str] = []
        for token, dep, _direction in path[:-1]:
            parts.append(f"{token} --{dep}-->")
        parts.append(path[-1][0])
        return ' '.join(parts)

    def verbalize(
        self,
        sentence: str,
        e1: str,
        e2: str,
        dep_chain: str,
    ) -> str:
        """Build the verbalized SDP sentence.

        Format::

            Sentence {sentence} | Entity-1: [{e1}] | Entity-2: [{e2}] |
            Dependency path: {dep_chain}
        """
        return (
            f"Sentence {sentence} | "
            f"Entity-1: [{e1}] | "
            f"Entity-2: [{e2}] | "
            f"Dependency path: {dep_chain}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_char_offsets(tokens: list[str]) -> list[tuple[int, int]]:
        """Return ``(char_start, char_end)`` for each token when joined with spaces."""
        offsets: list[tuple[int, int]] = []
        offset = 0
        for tok in tokens:
            offsets.append((offset, offset + len(tok)))
            offset += len(tok) + 1  # +1 for the space separator
        return offsets

    @staticmethod
    def _find_spacy_token_indices(doc, char_start: int, char_end: int) -> list[int]:
        """Return spaCy token indices whose span falls inside ``[char_start, char_end)``."""
        return [
            token.i
            for token in doc
            if token.idx >= char_start and token.idx + len(token.text) <= char_end
        ]

    def _bfs_sdp(
        self,
        doc,
        e1_indices: list[int],
        e2_indices: list[int],
    ) -> list[tuple[int, str | None, str | None]] | None:
        """BFS on the undirected dependency tree to find the shortest path.

        Each edge is stored twice:

        * ``(head_idx, dep_label, 'UP')``   — child → head
        * ``(child_idx, dep_label, 'DOWN')`` — head → child

        The dep_label is always the label that the **child** token carries
        (``token.dep_`` in spaCy).

        Returns:
            List of ``(spacy_token_idx, dep_label, direction)`` where the
            dep and direction at position *i* describe the edge leading from
            node *i* to node *i+1*.  The last element has
            ``dep_label=None, direction=None``.
            Returns ``None`` when no path exists.
        """
        graph: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        for token in doc:
            if token.i != token.head.i:
                graph[token.i].append((token.head.i, token.dep_, 'UP'))
                graph[token.head.i].append((token.i, token.dep_, 'DOWN'))

        e2_set = set(e2_indices)
        best_path: list[tuple[int, str | None, str | None]] | None = None

        for start_idx in e1_indices:
            visited: dict[int, tuple[int, str, str] | None] = {start_idx: None}
            queue: deque[int] = deque([start_idx])
            found: int | None = None

            while queue:
                node = queue.popleft()
                if node in e2_set:
                    found = node
                    break
                for neighbor, dep, direction in graph[node]:
                    if neighbor not in visited:
                        visited[neighbor] = (node, dep, direction)
                        queue.append(neighbor)

            if found is None:
                continue

            reverse_steps: list[tuple[int, str, str]] = []
            curr = found
            while visited[curr] is not None:
                prev, dep, direction = visited[curr]
                reverse_steps.append((curr, dep, direction))
                curr = prev
            reverse_steps.reverse()

            node_path = [start_idx] + [step[0] for step in reverse_steps]

            path: list[tuple[int, str | None, str | None]] = []
            for i, node_idx in enumerate(node_path[:-1]):
                _, dep, direction = reverse_steps[i]
                path.append((node_idx, dep, direction))
            path.append((node_path[-1], None, None))

            if best_path is None or len(path) < len(best_path):
                best_path = path

        return best_path


# ===========================================================================
# Concrete subclasses
# ===========================================================================

class BoWSDPEncoder(SDPEncoder, SentenceEncoder):
    """Encode a sentence as a multi-hot bag-of-words over SDP dependency labels.

    Each dimension of the output vector corresponds to one dependency-relation
    type in :attr:`dep_vocab`.  A dimension is set to 1 if that relation
    appears on the SDP between the two entities, 0 otherwise.

    No transformer model is required.  ``tokenize`` returns the raw SDP path
    and ``forward`` converts it to a one-hot tensor.

    Args:
        dep_vocab: ordered list of full dependency-relation names.  Defaults
            to an alphabetically sorted list of all known relation names.
    """

    def __init__(self, dep_vocab: Optional[list[str]] = None) -> None:
        torch.nn.Module.__init__(self)
        SDPEncoder.__init__(self, dep_vocab=dep_vocab)

    def tokenize(
        self,
        item: dict,
    ) -> list[tuple[str, str | None, str | None]]:
        """Extract the SDP path for one sample.

        Args:
            item: dict with ``'token'``, ``'h'``, and ``'t'`` keys.

        Returns:
            List of ``(token_text, dep_full_name, direction)`` tuples
            as returned by :meth:`extract_sdp`.
        """
        return self.extract_sdp(item)

    def forward(self, item: dict) -> torch.Tensor:
        """Return a multi-hot vector over dependency labels on the SDP.

        Returns:
            1-D ``torch.float32`` tensor of shape ``(len(dep_vocab),)``.
        """
        path = self.tokenize(item)
        vec = torch.zeros(len(self.dep_vocab), dtype=torch.float32)
        for _, dep, _ in path[:-1]:
            if dep is not None and dep in self.dep_to_idx:
                vec[self.dep_to_idx[dep]] = 1.0
        return vec

    def encode_onehot(self, item: dict) -> torch.Tensor:
        """Convenience alias for :meth:`forward`.

        Returns:
            1-D ``torch.float32`` tensor of shape ``(len(dep_vocab),)``.
        """
        return self(item)


class VerbalizedSDPEncoder(SDPEncoder, SentenceEncoder):
    """Encode a sentence by verbalizing its SDP and passing it through a transformer.

    The SDP is verbalized into a natural-language sentence and encoded by the
    inherited :class:`SentenceEncoder`, yielding an L2-normalised dense
    embedding.

    Args:
        model_name: HuggingFace model name or local path.
        dep_vocab: ordered list of full dependency-relation names.  Defaults
            to an alphabetically sorted list of all known relation names.
        device: PyTorch device string (e.g. ``'cpu'``, ``'cuda'``).
    """

    def __init__(
        self,
        model_name: str = 'HuggingFaceTB/SmolLM-135M-Instruct',
        dep_vocab: Optional[list[str]] = None,
        device: str = 'cpu',
    ) -> None:
        SDPEncoder.__init__(self, dep_vocab=dep_vocab)
        SentenceEncoder.__init__(self, model_name, device=device)

    def tokenize(self, item: dict) -> dict:
        """Verbalize the SDP for one sample and tokenize the result.

        Args:
            item: dict with keys:
                  - ``'token'`` (list[str])
                  - ``'h'``: ``{'name': str, 'pos': [start, end)}``
                  - ``'t'``: ``{'name': str, 'pos': [start, end)}``
        Returns:
            dict with ``'input_ids'`` and ``'attention_mask'`` tensors of
            shape ``(1, max_length)``.
        """
        tokens  = item['token']
        e1_name = item['h']['name']
        e2_name = item['t']['name']

        sentence   = ' '.join(tokens)
        path       = self.extract_sdp(item)
        dep_chain  = self.build_dep_chain(path)
        verbalized = self.verbalize(sentence, e1_name, e2_name, dep_chain)

        return SentenceEncoder.tokenize(self, verbalized)

    def forward(self, item: dict) -> torch.Tensor:
        """Encode the SDP verbalization for one sample.

        Args:
            item: dict with keys:
                  - ``'token'`` (list[str])
                  - ``'h'``: ``{'name': str, 'pos': [start, end)}``
                  - ``'t'``: ``{'name': str, 'pos': [start, end)}``
        Returns:
            Float32 tensor of shape ``(1, hidden_dim)``, L2-normalised.
        """
        device = next(self.model.parameters()).device
        batch = self.tokenize(item)
        batch = {k: v.to(device) for k, v in batch.items()}
        model_outputs = self.model(**batch)
        return self.average_pool(model_outputs.last_hidden_state, batch['attention_mask'])

    def encode_dense(self, item: dict) -> torch.Tensor:
        """Encode the sentence as a dense vector via SDP verbalization.

        Delegates to :meth:`forward`.

        Returns:
            Tensor of shape ``(1, hidden_dim)``, L2-normalised.
        """
        return self(item)
