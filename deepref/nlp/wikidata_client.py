import os
import json
import requests
from typing import Any, Dict, Optional

WIKIDATA_API = os.getenv("WIKIDATA_API", "https://www.wikidata.org/w/api.php")
WIKIDATA_SPARQL = os.getenv("WIKIDATA_SPARQL", "https://query.wikidata.org/sparql")


class WikidataClient:
    def __init__(self, mode: str = "mock", mock_path: str = "deepref/resources/mock_wikidata.json"):
        self.mode = mode
        self.mock_path = mock_path
        self._mock = self._load_mock() if self.mode == "mock" else {}
        self._cache: Dict[str, Dict[str, Any]] = {}

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "DeepREF/2.0 (Wikidata client)",
            }
        )

    def _load_mock(self) -> Dict[str, Any]:
        try:
            with open(self.mock_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # normaliza chaves para lowercase
            return {str(k).strip().lower(): v for k, v in data.items()}
        except Exception:
            return {}

    def get_attrs(self, term: str) -> Dict[str, Any]:
        term_key = (term or "").strip().lower()
        if not term_key:
            return {}

        if term_key in self._cache:
            return self._cache[term_key]

        if self.mode == "mock":
            out = self._mock.get(term_key, {})
            self._cache[term_key] = out if isinstance(out, dict) else {}
            return self._cache[term_key]

        # REAL (mínimo): procurar QID por label.
        # Depois você pode expandir para puxar instance_of (P31) e subir a hierarquia.
        try:
            qid = self._search_qid(term_key)
            if not qid:
                self._cache[term_key] = {}
                return {}
            # Retorna o mínimo por enquanto; você expande depois.
            out = {"qid": qid, "name": term}
            self._cache[term_key] = out
            return out
        except Exception:
            self._cache[term_key] = {}
            return {}

    def _search_qid(self, term: str) -> Optional[str]:
        params = {
            "action": "wbsearchentities",
            "search": term,
            "language": "en",
            "format": "json",
            "limit": 1,
        }
        r = self._session.get(WIKIDATA_API, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = data.get("search") or []
        if not results:
            return None
        return results[0].get("id")