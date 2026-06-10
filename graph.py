"""
graph.py — Knowledge Graph: rule-based NER + NetworkX graph + GraphRAG retrieval.

Extracts geological entities (скважина, пласт, горизонт, свита, месторождение)
from indexed chunks via regex. Co-occurrence on the same page creates edges.
No extra API calls — runs from already-indexed text.
"""

import re
import json
import threading
from pathlib import Path

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

GRAPH_FILE = Path(__file__).parent / "knowledge_graph.json"

_lock: threading.Lock = threading.Lock()
_cached_graph = None

# ─── Rule-based NER patterns ──────────────────────────────────────────────────

_NER_PATTERNS: dict[str, list[str]] = {
    "скважина": [
        r'скважин[аыею]\s*(?:№|#|N°?)?\s*(\d+[а-яА-ЯёЁ]?)',
        r'\bскв\.?\s*(?:№|#)?\s*(\d+[а-яА-ЯёЁ]?)',
    ],
    "пласт": [
        r'пласт[аыею]?\s+([А-ЯЁа-яёA-Za-z][А-ЯЁа-яёA-Za-z0-9\-\.]{1,20})',
    ],
    "горизонт": [
        r'горизонт[аыею]?\s+([А-ЯЁа-яёA-Za-z][А-ЯЁа-яёA-Za-z0-9\-\.]{1,20})',
    ],
    "свита": [
        r'([А-ЯЁ][а-яё]{3,}(?:ск|овск|евск)(?:ая|ой|ую|ом))\s+свит',
        r'свит[аыу]\s+([А-ЯЁ][а-яё]{3,})',
    ],
    "месторождение": [
        r'([А-ЯЁ][а-яё]{3,}(?:ск|овск|евск)(?:ое|ого|ому)?)\s+месторожден',
        r'месторожден[ияе]\s+([А-ЯЁ][а-яё]{3,})',
    ],
    "формация": [
        r'([А-ЯЁ][а-яё]{3,}(?:ск|овск|евск)(?:ая|ой))\s+форма[цц]и',
    ],
}

_TYPE_COLORS = {
    "скважина":      "#3b82f6",
    "пласт":         "#22c55e",
    "горизонт":      "#f59e0b",
    "свита":         "#a855f7",
    "месторождение": "#ef4444",
    "формация":      "#06b6d4",
}


# Common Russian words that should never be entity names
_WORD_BLACKLIST = {
    "не", "на", "по", "при", "до", "из", "без", "над", "под", "про",
    "как", "так", "или", "что", "где", "все", "это", "тот", "те",
    "таких", "такой", "такие", "такое", "самой", "самого",
    "чаще", "реже", "часто", "редко", "обычно",
    "полностью", "частично", "обнажались", "сравнительно",
    "последовало", "называется", "распределение",
    "характеризуются", "отличаются", "условными",
    "возвышается",
}


def _is_valid_entity_name(name: str, ent_type: str) -> bool:
    """Filter out false-positive NER matches."""
    if not name or len(name) < 2 or len(name) > 60:
        return False
    if name.lower() in _WORD_BLACKLIST:
        return False
    # Verb endings → not an entity
    if _is_verb(name):
        return False
    # For proper-noun types, require capitalized name
    if ent_type in ("свита", "месторождение", "формация"):
        if not name[0].isupper():
            return False
    # For пласт/горизонт: allow codes like "БС10", "Д-I", "АВ1"
    # but reject single common lowercase words
    if ent_type in ("пласт", "горизонт", "скважина"):
        if name.islower() and len(name) < 4:
            return False
        # Must contain at least one letter (not just punctuation)
        if not re.search(r"[А-ЯЁа-яёA-Za-z]", name):
            return False
    return True


def extract_entities(text: str, source: str, page: int) -> list[dict]:
    """Rule-based NER: extract geological entities from text."""
    entities: list[dict] = []
    seen: set[tuple] = set()
    for ent_type, patterns in _NER_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                name = m.group(1).strip()
                if not _is_valid_entity_name(name, ent_type):
                    continue
                key = (ent_type, name.upper())
                if key in seen:
                    continue
                seen.add(key)
                canonical = (
                    ent_type.upper()
                    + "_"
                    + re.sub(r"[^А-ЯЁA-Z0-9]", "_", name.upper())
                )
                entities.append({
                    "type":      ent_type,
                    "name":      name,
                    "canonical": canonical,
                    "source":    source,
                    "page":      page,
                })
    return entities


# ─── Graph building ───────────────────────────────────────────────────────────

def build_graph_from_chunks(chunks: list[dict]):
    """Build a NetworkX graph from indexed chunks via rule-based NER."""
    if not HAS_NX:
        return None

    G = nx.Graph()
    page_entities: dict[tuple, list[str]] = {}

    for chunk in chunks:
        text   = chunk.get("text", "")
        source = chunk.get("source_file", "")
        page   = chunk.get("page", 0)

        for ent in extract_entities(text, source, page):
            cid = ent["canonical"]
            if cid not in G:
                G.add_node(cid,
                           type=ent["type"],
                           name=ent["name"],
                           sources=[],
                           pages=[])
            node = G.nodes[cid]
            if source not in node["sources"]:
                node["sources"].append(source)
            if page not in node["pages"]:
                node["pages"].append(page)

            key = (source, page)
            page_entities.setdefault(key, [])
            if cid not in page_entities[key]:
                page_entities[key].append(cid)

    # Co-occurrence edges: entities on the same page → related
    for (src, pg), ent_ids in page_entities.items():
        for i, a in enumerate(ent_ids):
            for b in ent_ids[i + 1:]:
                if G.has_edge(a, b):
                    G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                    if pg not in G[a][b].get("pages", []):
                        G[a][b].setdefault("pages", []).append(pg)
                else:
                    G.add_edge(a, b,
                               relation="co-occurrence",
                               weight=1,
                               sources=[src],
                               pages=[pg])
    return G


def save_graph(G) -> None:
    if not HAS_NX or G is None:
        return
    data = nx.node_link_data(G)
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    global _cached_graph
    with _lock:
        _cached_graph = G


def get_graph():
    global _cached_graph
    with _lock:
        if _cached_graph is not None:
            return _cached_graph
        if not HAS_NX:
            return None
        if GRAPH_FILE.exists():
            with open(GRAPH_FILE, encoding="utf-8") as f:
                data = json.load(f)
            _cached_graph = nx.node_link_graph(data)
        else:
            _cached_graph = nx.Graph()
        return _cached_graph


def invalidate_graph() -> None:
    global _cached_graph
    with _lock:
        _cached_graph = None


def graph_stats() -> dict:
    G = get_graph()
    if G is None:
        return {"nodes": 0, "edges": 0, "available": False}
    return {
        "nodes":     G.number_of_nodes(),
        "edges":     G.number_of_edges(),
        "available": True,
        "types":     _count_by_type(G),
    }


def _count_by_type(G) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in G.nodes:
        t = G.nodes[n].get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


# ─── Query graph ─────────────────────────────────────────────────────────────

_VERB_ENDINGS = (
    "ться", "тся", "ются", "ется", "ваться", "иться",
    "аться", "ывать", "ивать", "овать",
)

def _is_verb(word: str) -> bool:
    w = word.lower()
    return any(w.endswith(e) for e in _VERB_ENDINGS)


def _stem6(word: str) -> str:
    """Simple 6-char prefix as morphological pseudo-stem."""
    return word.lower()[:6]


def _terms_match(term: str, label: str) -> bool:
    """Fuzzy match: substring OR 6-char prefix (handles Russian morphology)."""
    t, l = term.lower(), label.lower()
    if t in l or l in t:
        return True
    # Prefix match for inflected forms (Мегионской vs Мегионская)
    min_len = min(len(t), len(l))
    if min_len >= 5 and t[:6] == l[:6]:
        return True
    return False


def extract_query_entities(query: str) -> list[str]:
    """
    Extract entity names from a query string.
    Uses rule-based NER + fallback capitalized-word extraction.
    Filters out verbs and short noise words.
    """
    ents = extract_entities(query, "", 0)
    terms = [e["name"] for e in ents if not _is_verb(e["name"]) and len(e["name"]) > 2]

    # Fallback: capitalized Russian words (proper nouns) not already captured
    caps = re.findall(r'\b([А-ЯЁ][а-яё]{3,})\b', query)
    term_stems = {_stem6(t) for t in terms}
    for w in caps:
        if _stem6(w) not in term_stems and not _is_verb(w):
            terms.append(w)
            term_stems.add(_stem6(w))

    return terms


def query_graph_neighborhood(query_terms: list[str], max_nodes: int = 30) -> dict:
    """Return subgraph relevant to query terms for visualization + traversal trace."""
    G = get_graph()
    if G is None or not G.nodes:
        return {
            "nodes": [], "edges": [], "seed_nodes": [],
            "extracted_terms": [], "pages_from_seeds": [],
            "hop1_count": 0,
        }

    seed_nodes: list[str] = []
    for node in G.nodes:
        label = G.nodes[node].get("name", "")
        if any(_terms_match(t, label) for t in query_terms):
            seed_nodes.append(node)

    neighborhood: set[str] = set(seed_nodes)
    hop1_nodes: set[str] = set()
    for node in seed_nodes:
        for nb in list(G.neighbors(node))[:8]:
            if nb not in seed_nodes:
                hop1_nodes.add(nb)
            neighborhood.add(nb)

    # If nothing found, show top nodes by degree
    if not neighborhood:
        top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        neighborhood = {n for n, _ in top}
        hop1_nodes = neighborhood

    neighborhood = set(list(neighborhood)[:max_nodes])
    sub = G.subgraph(neighborhood)

    # Collect pages that seed nodes appear on (traversal evidence)
    pages_from_seeds: list[dict] = []
    seen_pages: set[tuple] = set()
    for n in seed_nodes:
        if n not in G.nodes:
            continue
        for src in G.nodes[n].get("sources", []):
            for pg in G.nodes[n].get("pages", []):
                key = (src, int(pg))
                if key not in seen_pages:
                    seen_pages.add(key)
                    short = src.replace(".pdf", "").replace(".img", "")[-25:]
                    pages_from_seeds.append({"source": short, "page": int(pg)})

    nodes_out = [
        {
            "id":      n,
            "label":   sub.nodes[n].get("name", n),
            "type":    sub.nodes[n].get("type", "unknown"),
            "color":   _TYPE_COLORS.get(sub.nodes[n].get("type", ""), "#64748b"),
            "sources": sub.nodes[n].get("sources", []),
            "pages":   sub.nodes[n].get("pages", []),
            "is_seed": n in seed_nodes,
            "is_hop1": n in hop1_nodes,
        }
        for n in sub.nodes
    ]
    edges_out = [
        {
            "from":     u,
            "to":       v,
            "relation": sub.edges[u, v].get("relation", "co-occurrence"),
            "weight":   sub.edges[u, v].get("weight", 1),
            "pages":    sub.edges[u, v].get("pages", []),
            "is_seed_edge": (u in seed_nodes or v in seed_nodes),
        }
        for u, v in sub.edges
    ]

    return {
        "nodes":            nodes_out,
        "edges":            edges_out,
        "seed_nodes":       seed_nodes,
        "extracted_terms":  query_terms,
        "pages_from_seeds": pages_from_seeds,
        "hop1_count":       len(hop1_nodes),
    }


def get_graph_pages(query_terms: list[str]) -> list[tuple[str, int]]:
    """Return (source_file, page) pairs where query entities appear — for GraphRAG chunk filtering."""
    G = get_graph()
    if G is None or not G.nodes:
        return []

    pages: set[tuple[str, int]] = set()

    for node in G.nodes:
        label = G.nodes[node].get("name", "")
        if any(_terms_match(t, label) for t in query_terms):
            for src in G.nodes[node].get("sources", []):
                for pg in G.nodes[node].get("pages", []):
                    pages.add((src, int(pg)))

    return list(pages)


def rebuild_graph_from_chroma() -> dict:
    """Rebuild graph from all chunks currently in ChromaDB. Returns stats."""
    try:
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        col    = client.get_collection(COLLECTION_NAME)
        data   = col.get(include=["documents", "metadatas"])
        chunks = [
            {"text": doc, **meta}
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
    except Exception as e:
        return {"error": str(e), "nodes": 0, "edges": 0}

    G = build_graph_from_chunks(chunks)
    if G is None:
        return {"error": "networkx not available", "nodes": 0, "edges": 0}

    save_graph(G)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "types": _count_by_type(G),
    }
