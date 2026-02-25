import os
import json
from typing import Any, Dict, List, Tuple

from deepref.nlp import babelnet_http
from deepref.nlp.wikidata_client import WikidataClient


class SemanticKNWL:

    def __init__(self):
        self.vocabulary: List[str] = []

        # BabelNet
        self.babelnet_mode = os.getenv("BABELNET_MODE", "mock")  # mock|real
        self.babelnet_key = os.getenv("BABELNET_KEY", "")
        self.babelnet_mock_path = os.getenv("BABELNET_MOCK_PATH", "deepref/resources/mock_babelnet.json")
        self.babelnet_search_lang = os.getenv("BABELNET_SEARCH_LANG", "EN")
        self.babelnet_target_lang = os.getenv("BABELNET_TARGET_LANG", "EN")
        self.babelnet_timeout = int(os.getenv("BABELNET_TIMEOUT", "20"))

        # Wikidata
        self.wikidata_enrich = os.getenv("WIKIDATA_ENRICH", "0") == "1"
        self.wikidata_mode = os.getenv("WIKIDATA_MODE", "mock")  # mock|real
        self.wikidata_mock_path = os.getenv("WIKIDATA_MOCK_PATH", "deepref/resources/mock_wikidata.json")
        self._wikidata = WikidataClient(mode=self.wikidata_mode, mock_path=self.wikidata_mock_path)

        # Mocks + caches
        self._bn_mock = self._load_mock(self.babelnet_mock_path) if self.babelnet_mode == "mock" else {}
        self._cache_super: Dict[str, Tuple[str, str]] = {}
        self._cache_wd: Dict[str, Dict[str, Any]] = {}

    def _load_mock(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {str(k).strip().lower(): v for k, v in data.items()}
        except Exception:
            return {}

    def unigram(self, entity: str) -> str:
        entity = (entity or "").strip().replace("_", " ")
        parts = entity.split()
        return parts[-1] if parts else ""

    def add(self, lst: List[str]) -> None:
        for token in lst:
            self.vocabulary.append(token)

    # -------------------- BabelNet superclasses -------------------- #
    def _babelnet_super_mock(self, entity: str) -> Tuple[str, str]:
        item = self._bn_mock.get(entity)
        if not item or not isinstance(item, dict):
            raise KeyError("entity not in babelnet mock")
        father = item.get("father") or entity
        grandpa = item.get("grandfather") or father
        return self.unigram(str(father)), self.unigram(str(grandpa))

    def _babelnet_super_real(self, entity: str) -> Tuple[str, str]:
        if not self.babelnet_key:
            raise RuntimeError("BABELNET_KEY not set")

        ids = babelnet_http.get_synset_ids(
            lemma=entity,
            key=self.babelnet_key,
            search_lang=self.babelnet_search_lang,
            timeout=self.babelnet_timeout,
        )
        synset_id = babelnet_http.pick_best_synset_id(ids)
        if not synset_id:
            return self.unigram(entity), self.unigram(entity)

        edges = babelnet_http.get_outgoing_edges(synset_id, key=self.babelnet_key, timeout=self.babelnet_timeout)
        father_id = None
        for e in edges:
            if babelnet_http.is_hypernym_edge(e):
                father_id = e.get("target")
                break
        if not father_id:
            return self.unigram(entity), self.unigram(entity)

        father_syn = babelnet_http.get_synset(
            father_id, key=self.babelnet_key, target_lang=self.babelnet_target_lang, timeout=self.babelnet_timeout
        )
        father_name = babelnet_http.synset_to_best_lemma(father_syn) or father_id

        edges2 = babelnet_http.get_outgoing_edges(father_id, key=self.babelnet_key, timeout=self.babelnet_timeout)
        grand_id = None
        for e in edges2:
            if babelnet_http.is_hypernym_edge(e):
                grand_id = e.get("target")
                break
        grand_id = grand_id or father_id

        grand_syn = babelnet_http.get_synset(
            grand_id, key=self.babelnet_key, target_lang=self.babelnet_target_lang, timeout=self.babelnet_timeout
        )
        grand_name = babelnet_http.synset_to_best_lemma(grand_syn) or grand_id

        return self.unigram(str(father_name)), self.unigram(str(grand_name))

    def _safe_super(self, entity: str) -> Tuple[str, str]:
        key = (entity or "").strip().lower()
        if not key:
            return "", ""
        if key in self._cache_super:
            return self._cache_super[key]

        try:
            if self.babelnet_mode == "mock":
                out = self._babelnet_super_mock(key)
            else:
                out = self._babelnet_super_real(key)
        except Exception:
            out = (self.unigram(key), self.unigram(key))

        self._cache_super[key] = out
        return out

    # -------------------- Wikidata attrs -------------------- #
    def _safe_wikidata(self, entity: str) -> Dict[str, Any]:
        key = (entity or "").strip().lower()
        if not key or not self.wikidata_enrich:
            return {}
        if key in self._cache_wd:
            return self._cache_wd[key]
        try:
            d = self._wikidata.get_attrs(key)
            self._cache_wd[key] = d if isinstance(d, dict) else {}
        except Exception:
            self._cache_wd[key] = {}
        return self._cache_wd[key]

    # -------------------- Public API -------------------- #
    def extract(self, entities: List[str]) -> Dict[str, Any]:
        """
        Espera 2 entidades (head/tail). Se vier menos, completa.
        Retorna sempre {'ses1': [father, grandpa], 'ses2': [father, grandpa]}.
        """
        ents = [((e or "").strip().lower()) for e in (entities or []) if (e or "").strip()]
        while len(ents) < 2:
            ents.append(ents[-1] if ents else "")

        e1, e2 = ents[0], ents[1]

        f1, g1 = self._safe_super(e1)
        f2, g2 = self._safe_super(e2)

        ses1 = [f1, g1]
        ses2 = [f2, g2]

        self.add(ses1 + ses2)

        out: Dict[str, Any] = {"ses1": ses1, "ses2": ses2}

        if self.wikidata_enrich:
            out["wikidata_ses1"] = self._safe_wikidata(e1)
            out["wikidata_ses2"] = self._safe_wikidata(e2)

        return out


_SINGLETON: SemanticKNWL | None = None


def get_semantic() -> SemanticKNWL:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = SemanticKNWL()
    return _SINGLETON