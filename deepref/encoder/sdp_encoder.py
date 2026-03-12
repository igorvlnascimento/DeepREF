"""
SDP-based encoders built on the Shortest Dependency Path (SDP).

Three classes:

  SDPEncoder (abstract)
      Shared NLP-tool-agnostic dep-label mapping, SDP extraction, chain
      building, and verbalization.  Not directly instantiable.
      Accepts any :class:`~deepref.nlp.nlp_tool.NLPTool` for dependency
      parsing; defaults to :class:`~deepref.nlp.spacy_nlp_tool.SpacyNLPTool`.

  BoWSDPEncoder(SDPEncoder, SentenceEncoder)
      Encodes a sentence as a multi-hot bag-of-words over the dependency
      relation labels found on the SDP.  No transformer required.

  VerbalizedSDPEncoder(SDPEncoder, LLMEncoder)
      Verbalizes the SDP and encodes the resulting string with a
      HuggingFace transformer via LLMEncoder.

Verbalized sentence format::

    Sentence {sentence} | Entity-1: [{e1}] | Entity-2: [{e2}] | Dependency path: {dep_chain}

where ``dep_chain`` uses full dependency-relation names
(e.g., "nominal subject" instead of the abbreviated "nsubj").
"""

from __future__ import annotations

from abc import ABC
from collections import defaultdict, deque
import collections
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set, Tuple

import torch

import networkx as nx

from deepref.encoder.sentence_encoder import SentenceEncoder
from deepref.encoder.llm_encoder import LLMEncoder
from deepref.nlp.nlp_tool import NLPTool, ParsedToken, Sentence

@dataclass
class Arc:
    """A directed arc on the SDP traversal."""
    deprel: str
    direction: str          # 'UP'  (child → head, toward root)
                            # 'DOWN'(head → child, away from root)

    def __str__(self) -> str:
        return f"--{self.deprel}({self.direction})-->"


@dataclass
class SDPNode:
    """A node on the (possibly pruned) SDP."""
    token_idx: int
    text: str               # surface form
    upos: str               # POS tag
    is_entity: bool = False
    entity_role: str = ""   # "ENTITY_1" | "ENTITY_2"
    entity_span: str = ""   # full surface span of the entity
    off_path_tokens: List[str] = field(default_factory=list)  # K-neighbors

    def verbalize(self) -> str:
        """Return the string representation of this node for the Query."""
        if self.is_entity:
            return f"[{self.entity_role}: {self.entity_span}]"
        base = f"{self.text}/{self.upos}"
        if self.off_path_tokens:
            neighbors = "+".join(self.off_path_tokens)
            base = f"{self.text}[+{neighbors}]/{self.upos}"
        return base


# ===========================================================================
# Abstract base
# ===========================================================================

class SDPEncoder(ABC):
    """Abstract base for SDP-based encoders.

    Provides pluggable NLP-tool dependency parsing, dep-label mapping, SDP
    extraction, chain building, and verbalization.  Subclasses decide how to
    produce the final representation from the SDP.

    Args:
        nlp_tool: :class:`~deepref.nlp.nlp_tool.NLPTool` instance used for
            dependency parsing.  Defaults to
            :class:`~deepref.nlp.spacy_nlp_tool.SpacyNLPTool` (``en_core_web_trf``).
        dep_vocab: ordered list of full dependency-relation names used as the
            one-hot vocabulary.  Defaults to an alphabetically sorted list of
            all relation names known to the encoder.
    """

    # ------------------------------------------------------------------
    # Mapping from dependency label abbreviations to full names
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

    def __init__(
        self,
        nlp_tool: Optional[NLPTool] = None,
        dep_vocab: Optional[list[str]] = None,
    ) -> None:
        if nlp_tool is None:
            from deepref.nlp.spacy_nlp_tool import SpacyNLPTool
            nlp_tool = SpacyNLPTool()
        self.nlp_tool: NLPTool = nlp_tool

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
        """Map a dependency label abbreviation to its full English name.

        Returns the label unchanged when it is not in the known mapping.
        """
        return self.DEP_FULL_NAMES.get(dep, dep)

    # def extract_sdp(
    #     self,
    #     item: dict,
    # ) -> list[tuple[str, str | None, str | None]]:
    #     """Extract the Shortest Dependency Path between the two marked entities.

    #     Args:
    #         item: dict with keys:
    #               ``'token'`` (list[str]), ``'h'`` and ``'t'`` each
    #               ``{'name': str, 'pos': [start, end)}``.
    #               ``pos`` indices are into the ``token`` list (exclusive end).

    #     Returns:
    #         List of ``(token_text, dep_full_name, direction)`` tuples.

    #         * The dep_full_name and direction of element ``i`` describe the
    #           edge **from node i to node i+1**.
    #         * The **last** element always has ``dep_full_name=None,
    #           direction=None``.
    #         * *direction* is ``'UP'`` (child → head) or ``'DOWN'``
    #           (head → child).

    #     Raises:
    #         ValueError: if the entity tokens cannot be found in the parsed
    #             sentence, or if no dependency path exists between the entities.
    #     """
    #     tokens = item['token']
    #     e1_pos = item['h']['pos']
    #     e2_pos = item['t']['pos']

    #     sentence = ' '.join(tokens)
    #     parsed_tokens = self.nlp_tool.parse_for_sdp(sentence)

    #     char_offsets = self._compute_char_offsets(tokens)

    #     e1_char_start = char_offsets[e1_pos[0]][0]
    #     e1_char_end   = char_offsets[e1_pos[1] - 1][1]
    #     e2_char_start = char_offsets[e2_pos[0]][0]
    #     e2_char_end   = char_offsets[e2_pos[1] - 1][1]

    #     e1_indices = self._find_token_indices(parsed_tokens, e1_char_start, e1_char_end)
    #     e2_indices = self._find_token_indices(parsed_tokens, e2_char_start, e2_char_end)

    #     if not e1_indices or not e2_indices:
    #         import logging
    #         logging.warning(
    #             "Could not locate entity tokens in the parsed sentence. "
    #             "e1=%r, e2=%r — returning empty SDP.",
    #             item['h']['name'], item['t']['name'],
    #         )
    #         return []

    #     e1_head = self._get_entity_head(parsed_tokens, e1_indices)
    #     e2_head = self._get_entity_head(parsed_tokens, e2_indices)

    #     path_indices = self._bfs_sdp(parsed_tokens, [e1_head], [e2_head])

    #     if path_indices is None:
    #         import logging
    #         logging.warning(
    #             "No dependency path between %r and %r — returning empty SDP.",
    #             item['h']['name'], item['t']['name'],
    #         )
    #         return []

    #     return [
    #         (
    #             parsed_tokens[idx].text,
    #             self.dep_to_full_name(dep) if dep is not None else None,
    #             direction,
    #         )
    #         for idx, dep, direction in path_indices
    #     ]

    def make_example_sentence(self, item: dict) -> Sentence:
        text = " ".join(item["token"])
        subj_text = item["h"]["name"]
        obj_text = item["t"]["name"]
        
        text = " ".join(item["token"])
        sentence = self.nlp_tool.parse_to_sentence(text, subj_text, obj_text)
        return sentence

    def extract_sdp(
        self,
        item: dict,
        k_values: list[int] = [1]
    ) -> list[tuple[str, str | None, str | None]]:
        sentence = self.make_example_sentence(item)
        G = self.build_dep_graph(sentence)
        sh = self.find_entity_head(sentence, sentence.subj_span)
        oh = self.find_entity_head(sentence, sentence.obj_span)
        lca = self.find_lca(sentence, sh, oh)
        lca_nodes = self.get_lca_subtree(sentence, lca)
        sdp = self.find_shortest_dep_path(G, sh, oh)

        sdp_set = set(sdp)
        lca_subgraph = G.subgraph(lca_nodes)
        dist_to_path: Dict[int, int] = {}
        for node in lca_nodes:
            min_d = float("inf")
            for sp in sdp_set:
                if sp in lca_subgraph and node in lca_subgraph:
                    try:
                        d = nx.shortest_path_length(lca_subgraph, node, sp)
                        min_d = min(min_d, d)
                    except nx.NetworkXNoPath:
                        pass
            dist_to_path[node] = int(min_d) if min_d != float("inf") else 999

        for K in k_values:
            kept = {i for i in lca_nodes if dist_to_path[i] <= K}
            pruned = set(sentence.tokens[t].text for t in range(len(sentence.tokens))
                        if t not in kept)
            print(f"\n  K = {K}:")
            print(f"    Kept   ({len(kept):>2} tokens) : "
                f"[{', '.join(sentence.tokens[i].text for i in sorted(kept))}]")
            if pruned:
                print(f"    Pruned ({len(sentence.tokens)-len(kept):>2} tokens) : "
                    f"[{', '.join(pruned)}]")
            adj = self.build_pruned_adjacency(sentence, kept)
            print(f"    Adjacency list (pruned tree):")
            for node_i in sorted(adj["adj_list"]):
                nbrs = adj["adj_list"][node_i]
                if nbrs:
                    nbr_texts = [sentence.tokens[n].text for n in set(nbrs)]
                    print(f"      '{sentence.tokens[node_i].text}' ↔ {nbr_texts}")

        if not sdp:
            return []

        result = []
        for i in range(len(sdp) - 1):
            node_i = sdp[i]
            node_j = sdp[i + 1]
            tok_i = sentence.tokens[node_i]
            tok_j = sentence.tokens[node_j]
            if tok_i.head_i == node_j:   # child → head
                dep, direction = tok_i.dep_, 'UP'
            else:                         # head → child
                dep, direction = tok_j.dep_, 'DOWN'
            result.append((tok_i.text, self.dep_to_full_name(dep), direction))
        result.append((sentence.tokens[sdp[-1]].text, None, None))
        return result
    
    # ─────────────────────────────────────────────────────────────
    # 2.  BUILD DEPENDENCY GRAPH
    # ─────────────────────────────────────────────────────────────

    def build_dep_graph(self, sentence: Sentence) -> nx.Graph:
        """
        Build an **undirected** NetworkX graph from the dependency tree.
        Nodes are token indices; edges carry the dependency label.
        (The paper treats the graph as undirected for GCN convolution.)
        """
        G = nx.Graph()
        for tok in sentence.tokens:
            G.add_node(tok.i, text=tok.text, dep=tok.dep_, pos=tok.pos_)
        for tok in sentence.tokens:
            if tok.i != tok.head_i:          # skip root self-loop
                G.add_edge(tok.i, tok.head_i, dep=tok.dep_)
        return G


    # ─────────────────────────────────────────────────────────────
    # 3.  FIND ENTITY HEAD TOKEN
    # ─────────────────────────────────────────────────────────────

    def find_entity_head(self, sentence: Sentence, span: Tuple[int, int]) -> int:
        """
        The head of an entity span is the token whose syntactic governor
        lies OUTSIDE the span (i.e., the token that 'hangs' the span
        into the rest of the tree).
        """
        span_indices = set(range(span[0], span[1] + 1))
        for i in span_indices:
            tok = sentence.tokens[i]
            if tok.head_i not in span_indices:
                return i
        # fallback: return the first token
        return span[0]


    # ─────────────────────────────────────────────────────────────
    # 4.  FIND LOWEST COMMON ANCESTOR (LCA)
    # ─────────────────────────────────────────────────────────────

    def get_ancestors(self, sentence: Sentence, token_i: int) -> List[int]:
        """Climb from token_i to root, collecting ancestor indices."""
        path = []
        seen = set()
        current = token_i
        while True:
            if current in seen:
                break
            seen.add(current)
            path.append(current)
            tok = sentence.tokens[current]
            if tok.head_i == current:   # root
                break
            current = tok.head_i
        return path


    def find_lca(self, sentence: Sentence, subj_head: int, obj_head: int) -> int:
        """Find the LCA of two nodes in the dependency tree."""
        subj_ancestors = self.get_ancestors(sentence, subj_head)
        obj_ancestors  = self.get_ancestors(sentence, obj_head)
        obj_ancestor_set = set(obj_ancestors)
        for anc in subj_ancestors:
            if anc in obj_ancestor_set:
                return anc
        return sentence.tokens[0].head_i   # fallback: root


    # ─────────────────────────────────────────────────────────────
    # 5.  COLLECT LCA SUBTREE
    # ─────────────────────────────────────────────────────────────

    def get_lca_subtree(self, sentence: Sentence, lca: int) -> Set[int]:
        """
        Collect all token indices in the subtree rooted at `lca`
        using BFS on the directed (head→child) tree.
        """
        children: Dict[int, List[int]] = collections.defaultdict(list)
        for tok in sentence.tokens:
            if tok.i != tok.head_i:
                children[tok.head_i].append(tok.i)

        subtree = set()
        queue = collections.deque([lca])
        while queue:
            node = queue.popleft()
            subtree.add(node)
            for child in children[node]:
                if child not in subtree:
                    queue.append(child)
        return subtree


    # ─────────────────────────────────────────────────────────────
    # 6.  SHORTEST DEPENDENCY PATH (SDP)
    # ─────────────────────────────────────────────────────────────

    def find_shortest_dep_path(self, G: nx.Graph,
                            subj_head: int,
                            obj_head: int) -> List[int]:
        """Return the ordered list of token indices on the SDP."""
        try:
            return nx.shortest_path(G, subj_head, obj_head)
        except nx.NetworkXNoPath:
            return [subj_head, obj_head]


    # ─────────────────────────────────────────────────────────────
    # 7.  PATH-CENTRIC PRUNING  ← core contribution of the paper
    # ─────────────────────────────────────────────────────────────

    def path_centric_prune(self, sentence: Sentence,
                        K: int) -> Tuple[Set[int], List[int], int]:
        """
        Prune the dependency tree keeping only tokens that are
        at most K edges away from the shortest dependency path
        inside the LCA subtree.

        Parameters
        ----------
        sentence : Sentence
        K        : int — pruning distance (0, 1, 2, … or large int for ∞)

        Returns
        -------
        kept_nodes : set of token indices to keep
        sdp        : list of token indices on the shortest dep path
        lca        : index of the lowest common ancestor token
        """
        G = self.build_dep_graph(sentence)

        subj_head = self.find_entity_head(sentence, sentence.subj_span)
        obj_head  = self.find_entity_head(sentence, sentence.obj_span)

        lca       = self.find_lca(sentence, subj_head, obj_head)
        lca_nodes = self.get_lca_subtree(sentence, lca)
        sdp       = self.find_shortest_dep_path(G, subj_head, obj_head)
        sdp_set   = set(sdp)

        # Restrict graph to LCA subtree for distance computation
        lca_subgraph = G.subgraph(lca_nodes)

        kept_nodes = set()
        for node in lca_nodes:
            # Distance = min hops from this node to any node on the SDP
            min_dist = float("inf")
            for sdp_node in sdp_set:
                if sdp_node in lca_subgraph and node in lca_subgraph:
                    try:
                        d = nx.shortest_path_length(lca_subgraph, node, sdp_node)
                        min_dist = min(min_dist, d)
                    except nx.NetworkXNoPath:
                        pass
            if min_dist <= K:
                kept_nodes.add(node)

        return kept_nodes, sdp, lca


    def build_pruned_adjacency(self,
                                sentence: Sentence,
                                kept_nodes: Set[int]) -> Dict:
        """
        Build adjacency list (and matrix) for the kept nodes.
        Returns a dict with 'adj_list' and 'adj_matrix'.
        """
        n = len(sentence.tokens)
        kept_sorted = sorted(kept_nodes)
        idx_map = {orig: new for new, orig in enumerate(kept_sorted)}

        adj_matrix = [[0] * len(kept_sorted) for _ in range(len(kept_sorted))]
        adj_list: Dict[int, List[int]] = {i: [] for i in kept_sorted}

        for tok in sentence.tokens:
            if tok.i in kept_nodes and tok.head_i in kept_nodes and tok.i != tok.head_i:
                i, j = idx_map[tok.i], idx_map[tok.head_i]
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
                adj_list[tok.i].append(tok.head_i)
                adj_list[tok.head_i].append(tok.i)

        # Add self-loops (Ã = A + I as in the paper)
        for i in range(len(kept_sorted)):
            adj_matrix[i][i] = 1

        return {
            "adj_list":   adj_list,
            "adj_matrix": adj_matrix,
            "token_order": kept_sorted,
            "idx_map":    idx_map,
        }

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
        if not path:
            return ""
        parts: list[str] = []
        for token, dep, _direction in path[:-1]:
            parts.append(f"{token} --{dep}-->")
        parts.append(path[-1][0])
        return ' '.join(parts)

    def verbalize(
        self,
        dep_chain: str,
    ) -> str:
        """Build the verbalized SDP sentence.

        Format::

            Sentence {sentence} | Entity-1: [{e1}] | Entity-2: [{e2}] |
            Dependency path: {dep_chain}
        """
        return (
            f"Instruct: Given a syntactic dependency path between two named entities, identify the semantic relation they hold.\n"
            f"Query: {dep_chain}"
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
    def _find_token_indices(
        parsed_tokens: list[ParsedToken],
        char_start: int,
        char_end: int,
    ) -> list[int]:
        """Return indices of parsed tokens whose span falls inside ``[char_start, char_end]``."""
        return [
            pt.idx
            for pt in parsed_tokens
            if pt.char_start >= char_start and pt.char_end <= char_end
        ]

    @staticmethod
    def _get_entity_head(
        parsed_tokens: list[ParsedToken],
        indices: list[int],
    ) -> int:
        """Return the index of the syntactic head token of a multi-token entity."""
        #idx_set = set(indices)
        span_indices = set(range(indices[0], indices[1] + 1))
        for i in span_indices:
            tok = parsed_tokens[i]
            if parsed_tokens[i].head_idx not in span_indices:
                return i
        return indices[0]  # fallback

    def _bfs_sdp(
        self,
        parsed_tokens: list[ParsedToken],
        e1_indices: list[int],
        e2_indices: list[int],
    ) -> list[tuple[int, str | None, str | None]] | None:
        """BFS on the undirected dependency tree to find the shortest path.

        Each edge is stored twice:

        * ``(head_idx, dep_label, 'UP')``   — child → head
        * ``(child_idx, dep_label, 'DOWN')`` — head → child

        The dep_label is always the label that the **child** token carries.

        Returns:
            List of ``(token_idx, dep_label, direction)`` where the
            dep and direction at position *i* describe the edge leading from
            node *i* to node *i+1*.  The last element has
            ``dep_label=None, direction=None``.
            Returns ``None`` when no path exists.
        """
        graph: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        for pt in parsed_tokens:
            if pt.idx != pt.head_idx:  # ROOT tokens have head_idx == idx
                graph[pt.idx].append((pt.head_idx, pt.dep_, 'UP'))
                graph[pt.head_idx].append((pt.idx, pt.dep_, 'DOWN'))

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
    
    # ──────────────────────────────────────────────────────────────────────────────
    # Entity extraction helpers
    # ──────────────────────────────────────────────────────────────────────────────

    def extract_entities_and_clean(self, marked_sentence: str) -> Tuple[str, str, str]:
        """
        Strip [E1]...[/E1] and [E2]...[/E2] markers.

        Returns
        -------
        clean_sentence : str   — sentence without any markers
        e1_text        : str   — surface span of entity 1
        e2_text        : str   — surface span of entity 2
        """
        e1_match = re.search(r'\[E1\](.*?)\[/E1\]', marked_sentence, re.DOTALL)
        e2_match = re.search(r'\[E2\](.*?)\[/E2\]', marked_sentence, re.DOTALL)

        if not e1_match or not e2_match:
            raise ValueError(
                "Sentence must contain both [E1]...[/E1] and [E2]...[/E2] markers."
            )

        e1_text = re.sub(r'\s+', ' ', e1_match.group(1)).strip()
        e2_text = re.sub(r'\s+', ' ', e2_match.group(1)).strip()

        clean = re.sub(r'\[/?E[12]\]', '', marked_sentence)
        clean = re.sub(r'\s+', ' ', clean).strip()

        return clean, e1_text, e2_text


    def find_entity_token_indices(
        self,
        words: list,
        entity_text: str
    ) -> List[int]:
        """
        Locate the token indices in word list that correspond to entity_text.

        Strategy:
            1. Exact left-to-right n-gram match (case-insensitive).
            2. Fallback: single-token substring match.
        """
        entity_tokens = entity_text.split()
        n_words = len(words)
        n_ent = len(entity_tokens)

        # --- exact n-gram match ---
        for start in range(n_words - n_ent + 1):
            match = all(
                words[start + j].text.lower() == entity_tokens[j].lower()
                for j in range(n_ent)
            )
            if match:
                return list(range(start, start + n_ent))

        # --- fallback: partial match on first token ---
        for i, w in enumerate(words):
            if w.text.lower() == entity_tokens[0].lower():
                return [i]

        # --- last resort: substring ---
        for i, w in enumerate(words):
            if entity_tokens[0].lower() in w.text.lower():
                return [i]

        raise ValueError(
            f"Entity '{entity_text}' not found in sentence tokens: "
            f"{[w.text for w in words]}"
        )


    def get_entity_head(self, words: list, token_indices: List[int]) -> int:
        """
        Return the index of the syntactic head of the entity span.

        The head is the token whose parent (head) is OUTSIDE the span,
        i.e. the token that links the whole span to the rest of the tree.
        NLPTool uses 1-indexed heads; head == 0 means the token is the ROOT.
        """
        span_set = set(token_indices)
        for idx in token_indices:
            head = self.nlp_tool.get_entity_head(words[idx])
            parent_id = head - 1   # convert to 0-indexed
            # token is root  OR  its parent is outside the span
            if head == 0 or parent_id not in span_set:
                return idx
        return token_indices[0]              # safe fallback


    # ──────────────────────────────────────────────────────────────────────────────
    # Dependency tree helpers
    # ──────────────────────────────────────────────────────────────────────────────

    def build_graph(self, words: list) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], Arc]]:
        """
        Build an undirected adjacency list and a directed arc dictionary from
        word list.

        adjacency : { node_idx : [neighbor_idx, ...] }
        arcs      : { (from_idx, to_idx) : Arc }
                    — arc goes from_idx → to_idx
                    — UP   means traversal toward root   (child → parent)
                    — DOWN means traversal away from root (parent → child)
        """
        adjacency: Dict[int, List[int]] = defaultdict(list)
        arcs: Dict[Tuple[int, int], Arc] = {}

        for i, word in enumerate(words):
            head = self.nlp_tool.get_entity_head(word)
            if head == 0:
                continue
            parent = head - 1
            child = i

            adjacency[child].append(parent)
            adjacency[parent].append(child)

            deprel = self.nlp_tool.get_deprel(word)

            # child → parent : going UP toward root
            arcs[(child, parent)] = Arc(deprel=deprel, direction='UP')
            # parent → child : going DOWN away from root
            arcs[(parent, child)] = Arc(deprel=deprel, direction='DOWN')
        return adjacency, arcs


    def find_sdp(
        self,
        adjacency: Dict[int, List[int]],
        src: int,
        tgt: int
    ) -> Optional[List[int]]:
        """
        BFS on the undirected dependency graph to find the Shortest Dependency Path.

        Returns the ordered list of node indices from src to tgt,
        or None if no path exists.
        """
        if src == tgt:
            return [src]

        visited: Set[int] = {src}
        queue: deque = deque([(src, [src])])

        while queue:
            node, path = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor == tgt:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


    def path_centric_pruning(
        self,
        adjacency: Dict[int, List[int]],
        sdp_nodes: List[int],
        K: int
    ) -> Set[int]:
        """
        Zhang et al. (2018) path-centric pruning.

        Keep all tokens within K hops of any node on the SDP:
            K = 0  →  SDP only (pure shortest-path model)
            K = 1  →  SDP + direct neighbors  (best F1 in paper)
            K = 2  →  SDP + 2-hop neighborhood
            K = ∞  →  full LCA subtree

        Returns the set of retained token indices.
        """
        if K == 0:
            return set(sdp_nodes)

        retained: Set[int] = set(sdp_nodes)

        for sdp_node in sdp_nodes:
            frontier: Set[int] = {sdp_node}
            visited: Set[int] = {sdp_node}
            for _ in range(K):
                next_frontier: Set[int] = set()
                for node in frontier:
                    for neighbor in adjacency[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                            retained.add(neighbor)
                frontier = next_frontier

        return retained


    # ──────────────────────────────────────────────────────────────────────────────
    # Core verbalization logic
    # ──────────────────────────────────────────────────────────────────────────────

    def build_sdp_nodes(
        self,
        words: list,
        sdp_path: List[int],
        arcs: Dict[Tuple[int, int], Arc],
        adjacency: Dict[int, List[int]],
        e1_indices: List[int],
        e1_head: int,
        e1_text: str,
        e2_indices: List[int],
        e2_head: int,
        e2_text: str,
        pruned_nodes: Set[int],
        K: int
    ) -> List[Tuple[SDPNode, Optional[Arc]]]:
        """
        Build the sequence of (SDPNode, Arc_to_next) pairs that will be verbalized.

        Off-path neighbors (K > 0) are listed inside the node label as
        token[+neighbor1+neighbor2]/POS, capped at 3 neighbors for readability.
        """
        e1_set = set(e1_indices)
        e2_set = set(e2_indices)
        sdp_set = set(sdp_path)

        result: List[Tuple[SDPNode, Optional[Arc]]] = []

        for step, node_idx in enumerate(sdp_path):

            # --- determine off-path K-neighbors for this node ---
            off_path: List[str] = []
            if K > 0:
                for neighbor in adjacency[node_idx]:
                    if (
                        neighbor not in sdp_set
                        and neighbor in pruned_nodes
                        and neighbor not in e1_set
                        and neighbor not in e2_set
                    ):
                        off_path.append(words[neighbor].text)
                off_path = off_path[:3]          # cap at 3 for readability

            # --- classify as entity or intermediate token ---
            upos = self.nlp_tool.get_pos(words[node_idx])
            if node_idx in e1_set or node_idx == e1_head:
                node = SDPNode(
                    token_idx=node_idx,
                    text=words[node_idx].text,
                    upos=upos,
                    is_entity=True,
                    entity_role="ENTITY_1",
                    entity_span=e1_text,
                    off_path_tokens=[],       # no off-path inside entity brackets
                )
            elif node_idx in e2_set or node_idx == e2_head:
                node = SDPNode(
                    token_idx=node_idx,
                    text=words[node_idx].text,
                    upos=upos,
                    is_entity=True,
                    entity_role="ENTITY_2",
                    entity_span=e2_text,
                    off_path_tokens=[],
                )
            else:
                node = SDPNode(
                    token_idx=node_idx,
                    text=words[node_idx].text,
                    upos=upos,
                    is_entity=False,
                    off_path_tokens=off_path,
                )

            # --- arc to next node on path (None for last node) ---
            arc: Optional[Arc] = None
            if step < len(sdp_path) - 1:
                next_idx = sdp_path[step + 1]
                arc = arcs.get(
                    (node_idx, next_idx),
                    Arc(deprel="dep", direction="DOWN")   # safe fallback
                )

            result.append((node, arc))

        return result
    
    def mark_sentence(self, item: dict) -> str:
        token = item["token"]
        h1, h2 = item["h"]["pos"][0], item["h"]["pos"][1]
        t1, t2 = item["t"]["pos"][0], item["t"]["pos"][1]

        if h1 < t1:
            token = token[:t1] + ["[E2]"] + token[t1:t2] + ["[/E2]"] + token[t2:]
            token = token[:h1] + ["[E1]"] + token[h1:h2] + ["[/E1]"] + token[h2:]
        else:
            token = token[:h1] + ["[E1]"] + token[h1:h2] + ["[/E1]"] + token[h2:]
            token = token[:t1] + ["[E2]"] + token[t1:t2] + ["[/E2]"] + token[t2:]

        return " ".join(token)


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
        nlp_tool: :class:`~deepref.nlp.nlp_tool.NLPTool` for dependency
            parsing.  Defaults to :class:`~deepref.nlp.spacy_nlp_tool.SpacyNLPTool`.
        dep_vocab: ordered list of full dependency-relation names.  Defaults
            to an alphabetically sorted list of all known relation names.
    """

    def __init__(
        self,
        nlp_tool: Optional[NLPTool] = None,
        dep_vocab: Optional[list[str]] = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        SDPEncoder.__init__(self, nlp_tool=nlp_tool, dep_vocab=dep_vocab)

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


class VerbalizedSDPEncoder(SDPEncoder, LLMEncoder):
    """Encode a sentence by verbalizing its SDP and passing it through a transformer.

    The SDP is verbalized into a natural-language sentence and encoded by the
    inherited :class:`LLMEncoder`, yielding an L2-normalised dense
    embedding.

    Args:
        nlp_tool: :class:`~deepref.nlp.nlp_tool.NLPTool` for dependency
            parsing.  Defaults to :class:`~deepref.nlp.spacy_nlp_tool.SpacyNLPTool`.
        model_name: HuggingFace model name or local path.
        dep_vocab: ordered list of full dependency-relation names.  Defaults
            to an alphabetically sorted list of all known relation names.
        device: PyTorch device string (e.g. ``'cpu'``, ``'cuda'``).
    """

    def __init__(
        self,
        nlp_tool: Optional[NLPTool] = None,
        model_name: str = 'HuggingFaceTB/SmolLM-135M-Instruct',
        dep_vocab: Optional[list[str]] = None,
        device: str = 'cpu',
    ) -> None:
        SDPEncoder.__init__(self, nlp_tool=nlp_tool, dep_vocab=dep_vocab)
        LLMEncoder.__init__(self, model_name, device=device)

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
        # e1_name = item['h']['name']
        # e2_name = item['t']['name']

        sentence = self.mark_sentence(item)
        # path       = self.extract_sdp(item)
        # dep_chain  = self.build_dep_chain(path)
        verbalized = self.verbalize(sentence, K=1)

        return LLMEncoder.tokenize_prompt(self, verbalized)

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
        batch = self.tokenize(item)
        #print("batch:", batch)
        model_outputs = self.registry.run_from_input_ids(
            self.model_name,
            batch["input_ids"],
            batch["attention_mask"],
        )
        return self.last_token_pool(model_outputs.last_hidden_state, batch['attention_mask'])

    def encode_dense(self, item: dict) -> torch.Tensor:
        """Encode the sentence as a dense vector via SDP verbalization.

        Delegates to :meth:`forward`.

        Returns:
            Tensor of shape ``(1, hidden_dim)``, L2-normalised.
        """
        return self(item)

    def verbalize(
            self,
            marked_sentence: str,
            K: int = 0,
            task_instruction: Optional[str] = None
    ) -> str:
        """
        Full pipeline: parse → SDP → prune → verbalize.

        Parameters
        ----------
        marked_sentence   : sentence with [E1]...[/E1] and [E2]...[/E2] tags
        nlp               : initialized pipeline (tokenize, pos, depparse)
        K                 : path-centric pruning distance (Zhang et al. 2018)
                            0 = SDP only, 1 = best trade-off, 2 = wider context
        task_instruction  : override the default Qwen3 instruction prefix

        Returns
        -------
        verbalization     : complete canonical string ready for Qwen3-embedding-8B
        """

        # ── 1. strip entity markers, remember surface spans ──────────────────────
        clean_sentence, e1_text, e2_text = self.extract_entities_and_clean(marked_sentence)

        # ── 2. parse with NLPTool ─────────────────────────────────────────────────
        words = self.nlp_tool.get_words(clean_sentence)

        # ── 3. locate entity token indices & heads ───────────────────────────────
        e1_indices = self.find_entity_token_indices(words, e1_text)
        e2_indices = self.find_entity_token_indices(words, e2_text)
        e1_head = self.get_entity_head(words, e1_indices)
        e2_head = self.get_entity_head(words, e2_indices)

        # ── 4. build dependency graph ─────────────────────────────────────────────
        adjacency, arcs = self.build_graph(words)

        # ── 5. find SDP between entity heads ─────────────────────────────────────
        sdp_path = self.find_sdp(adjacency, e1_head, e2_head)
        if sdp_path is None:
            raise RuntimeError(
                f"No dependency path found between '{e1_text}' and '{e2_text}'."
            )

        # ── 6. path-centric pruning (Zhang et al. 2018) ───────────────────────────
        pruned_nodes = self.path_centric_pruning(adjacency, sdp_path, K)

        # ── 7. build structured SDP node sequence ────────────────────────────────
        node_arc_pairs = self.build_sdp_nodes(
            words, sdp_path, arcs, adjacency,
            e1_indices, e1_head, e1_text,
            e2_indices, e2_head, e2_text,
            pruned_nodes, K
        )

        # ── 8. assemble Query string ──────────────────────────────────────────────
        query_parts: List[str] = []
        for node, arc in node_arc_pairs:
            query_parts.append(node.verbalize())
            if arc is not None:
                query_parts.append(str(arc))

        query_str = " ".join(query_parts)

        # ── 9. prepend Qwen3 instruction prefix ───────────────────────────────────
        instruction = task_instruction or (
            "Given a syntactic dependency path between two named entities, "
            "identify the semantic relation they hold."
        )

        verbalization = (
            f"Instruct: {instruction}\n"
            f"Query: {query_str}"
        )

        return verbalization
