import os
import re
import requests
from typing import Any, Dict, List, Optional

BABELNET_BASE = os.getenv("BABELNET_BASE", "https://babelnet.io/v9")

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "Accept-Encoding": "gzip",
        "User-Agent": "DeepREF/2.0 (BabelNet client)",
    }
)


def _get(url: str, params: Dict[str, Any], timeout: int) -> Any:
    r = _SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_synset_ids(
    lemma: str,
    key: str,
    search_lang: str = "EN",
    pos: Optional[str] = None,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"lemma": lemma, "searchLang": search_lang, "key": key}
    if pos:
        params["pos"] = pos
    return _get(f"{BABELNET_BASE}/getSynsetIds", params=params, timeout=timeout)


def get_outgoing_edges(synset_id: str, key: str, timeout: int = 20) -> List[Dict[str, Any]]:
    params = {"id": synset_id, "key": key}
    return _get(f"{BABELNET_BASE}/getOutgoingEdges", params=params, timeout=timeout)


def get_synset(
    synset_id: str, key: str, target_lang: str = "EN", timeout: int = 20
) -> Dict[str, Any]:
    params = {"id": synset_id, "key": key, "targetLang": target_lang}
    return _get(f"{BABELNET_BASE}/getSynset", params=params, timeout=timeout)


def synset_to_best_lemma(synset_json: Dict[str, Any]) -> Optional[str]:
    """
    Tenta extrair um nome legível do synset via senses.properties.
    Retorna algo como "king" ou "King Charles" (normaliza underscores).
    """
    senses = synset_json.get("senses") or []
    for s in senses:
        props = s.get("properties") or {}
        full = props.get("fullLemma") or props.get("simpleLemma")
        if full:
            txt = str(full).strip()
            txt = re.sub(r"[_]+", " ", txt)
            return txt
    return None


def is_hypernym_edge(edge: Dict[str, Any]) -> bool:
    """
    Heurística robusta: BabelNet aponta relations em edge.pointer.
    """
    ptr = edge.get("pointer") or {}
    rg = (ptr.get("relationGroup") or "").upper()
    name = (ptr.get("name") or "").lower()
    short = (ptr.get("shortName") or "").lower()

    if "HYPERNYM" in rg:
        return True
    if "hypernym" in name:
        return True
    if short in {"@", "@i"}:
        # WordNet pointer clássico (às vezes aparece)
        return True
    return False


def pick_best_synset_id(ids: List[Dict[str, Any]]) -> Optional[str]:
    """
    Heurística simples:
    - Se existir pos == NOUN, pega o primeiro NOUN
    - Senão, pega o primeiro
    """
    for it in ids:
        if (it.get("pos") or "").upper() == "NOUN":
            return it.get("id")
    return ids[0].get("id") if ids else None