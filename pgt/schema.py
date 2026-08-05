# pgt/schema.py
"""Central record contracts and runtime validators for the PGT pipeline.

The project exchanges JSON/JSONL records between six stages:

``Sentences -> Extraction -> Local Graph -> MES -> Candidates -> Reranking``.

Historically, each stage accepted slightly different field names and silently
continued when records were malformed.  This module provides one dependency-free
contract for every boundary.  It intentionally uses only the Python standard
library so validation can run before expensive LLM/API calls.

Design principles
-----------------
* Validators return a list of stable, human-readable errors; an empty list means
  the record satisfies the contract.
* ``assert_valid_*`` helpers raise :class:`RecordValidationError` and are intended
  for fail-fast production runs.
* Optional cross-record arguments allow stronger checks, e.g. that a MES really
  is a subgraph of its source local graph.
* The legacy ``validate_evidence_ids`` function is retained for compatibility
  with ``pgt.extract``.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NotRequired,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
)


# ---------------------------------------------------------------------------
# Shared aliases, enums, and regular expressions
# ---------------------------------------------------------------------------

EvidenceId = str
TechniqueId = str
RecordKind = Literal[
    "sentences",
    "extraction",
    "local_graph",
    "mes",
    "candidates",
    "reranking",
]

EVIDENCE_ID_RE = re.compile(r"^E([1-9]\d*)$")
TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

STRUCTURAL_NODE_TYPES: Tuple[str, ...] = (
    "Precondition",
    "Entry",
    "VulnType",
    "Behavior",
    "Impact",
)
GRAPH_NODE_TYPES: Tuple[str, ...] = ("CVE", "Evidence", *STRUCTURAL_NODE_TYPES)
TRACEABILITY_EDGE_TYPES: Tuple[str, ...] = ("mentions", "supported_by")
STRUCTURAL_EDGE_TYPES: Tuple[str, ...] = ("enables", "characterized_by", "causes")
GRAPH_EDGE_TYPES: Tuple[str, ...] = (*TRACEABILITY_EDGE_TYPES, *STRUCTURAL_EDGE_TYPES)
MES_STATUSES: Tuple[str, ...] = ("complete", "partial", "empty")
RERANK_MODES: Tuple[str, ...] = ("generic", "evidence", "structure", "full")

SCHEMA_VERSIONS: Dict[RecordKind, str] = {
    "sentences": "sentences-contract-v2.0.0",
    "extraction": "extraction-contract-v2.0.0",
    "local_graph": "local-graph-contract-v2.0.0",
    "mes": "mes-contract-v2.0.0",
    "candidates": "candidate-contract-v2.0.0",
    "reranking": "reranking-contract-v2.0.0",
}


# ---------------------------------------------------------------------------
# Static field contracts (TypedDicts)
# ---------------------------------------------------------------------------

class EvidenceSpanRecord(TypedDict):
    start: int
    end: int
    text_sha256: str


class SegmentationRecord(TypedDict):
    version: str
    mode: str
    source_text_field: str
    source_text_sha256: str
    evidence_count: int
    reconstruction_sha256: str
    parameters: Dict[str, Any]


class SentenceRecord(TypedDict):
    input_id: str
    raw_text: str
    sentences: Dict[EvidenceId, str]
    evidence_spans: Dict[EvidenceId, EvidenceSpanRecord]
    segmentation: SegmentationRecord
    provenance: NotRequired[Dict[str, Any]]


class PreconditionRecord(TypedDict):
    condition: str
    evidence_ids: List[EvidenceId]
    confidence: float


class EntryRecord(TypedDict):
    vector: str
    detail: str
    evidence_ids: List[EvidenceId]
    confidence: float


class VulnTypeRecord(TypedDict):
    type: str
    subtype: str
    evidence_ids: List[EvidenceId]
    confidence: float


class BehaviorRecord(TypedDict):
    action: str
    target: Optional[str]
    impact: Optional[str]
    evidence_ids: List[EvidenceId]
    confidence: float


class ImpactRecord(TypedDict):
    type: str
    detail: NotRequired[str]
    evidence_ids: List[EvidenceId]
    confidence: float


class RelationRecord(TypedDict):
    src: str
    type: Literal["enables", "characterized_by", "causes"]
    dst: str
    evidence_ids: List[EvidenceId]
    confidence: float


class ExtractionRecord(TypedDict):
    input_id: str
    preconditions: List[PreconditionRecord]
    entry: List[EntryRecord]
    vuln_type: List[VulnTypeRecord]
    behaviors: List[BehaviorRecord]
    relations: List[RelationRecord]
    impacts: List[ImpactRecord]
    _used_llm: NotRequired[bool]
    _validation_errors: NotRequired[List[str]]
    _provenance: NotRequired[Dict[str, Any]]


class GraphNodeRecord(TypedDict, total=False):
    id: str
    type: str
    text: str
    evidence_id: EvidenceId
    evidence_ids: List[EvidenceId]
    confidence: float


class GraphEdgeRecord(TypedDict, total=False):
    src: str
    dst: str
    type: str
    origin: str
    evidence_ids: List[EvidenceId]
    shared_evidence_ids: List[EvidenceId]
    confidence: float
    structural_score: float


class LocalGraphRecord(TypedDict):
    input_id: str
    graph_version: str
    nodes: List[GraphNodeRecord]
    edges: List[GraphEdgeRecord]
    stats: Dict[str, Any]
    warnings: List[str]


class MESRecord(TypedDict):
    input_id: str
    algorithm: str
    source_graph_version: Optional[str]
    parameters: Dict[str, Any]
    status: Literal["complete", "partial", "empty"]
    complete_core_chain: bool
    chain: List[str]
    chain_types: NotRequired[List[str]]
    structural_node_ids: List[str]
    evidence_ids: List[EvidenceId]
    nodes: List[GraphNodeRecord]
    edges: List[GraphEdgeRecord]
    compact_text: str
    selection_trace: Dict[str, Any]
    warnings: List[str]
    mes_sha256: str


class CandidateRecordItem(TypedDict):
    technique_id: TechniqueId
    score_fused: float
    score_text: float
    score_structure: float
    score_graph: NotRequired[float]
    rank: int


class CandidateRecord(TypedDict):
    input_id: str
    candidates: List[CandidateRecordItem]
    retrieval_metadata: Dict[str, Any]


class RerankedCandidateRecordItem(CandidateRecordItem):
    retrieval_rank: int
    retrieval_score: float
    llm_score: float
    final_score: float
    reason: str
    evidence_ids: List[EvidenceId]
    constraint_flags: List[str]
    rerank_rank: int


class RerankRecord(TypedDict):
    input_id: str
    candidates: List[RerankedCandidateRecordItem]
    rerank_metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Backward-compatible dataclasses retained from the original module
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRef:
    evidence_ids: List[EvidenceId]


@dataclass
class Behavior(EvidenceRef):
    action: str
    target: Optional[str] = None
    impact: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Relation(EvidenceRef):
    src: str
    rel: Literal[
        "enables",
        "leads_to",
        "supported_by",
        "has_precondition",
        "has_vulnerability",
    ]
    dst: str


@dataclass
class Extraction:
    input_id: str
    preconditions: List[Dict[str, Any]] = field(default_factory=list)
    entry: List[Dict[str, Any]] = field(default_factory=list)
    vuln_type: List[Dict[str, Any]] = field(default_factory=list)
    behaviors: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    impacts: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation infrastructure
# ---------------------------------------------------------------------------

class RecordValidationError(ValueError):
    """Raised when a pipeline record violates its stage contract."""

    def __init__(self, kind: str, errors: Sequence[str], input_id: Optional[str] = None):
        self.kind = kind
        self.errors = list(errors)
        self.input_id = input_id
        subject = f" for {input_id}" if input_id else ""
        preview = "; ".join(self.errors[:8])
        if len(self.errors) > 8:
            preview += f"; ... ({len(self.errors) - 8} more)"
        super().__init__(f"Invalid {kind} record{subject}: {preview}")


def _err(errors: List[str], path: str, code: str, message: str) -> None:
    errors.append(f"{path}: [{code}] {message}")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_nonempty_string(value: Any, path: str, errors: List[str]) -> Optional[str]:
    if not isinstance(value, str):
        _err(errors, path, "type", "must be a string")
        return None
    if not value.strip():
        _err(errors, path, "empty", "must not be empty")
        return None
    return value


def _validate_sha256(value: Any, path: str, errors: List[str]) -> Optional[str]:
    if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
        _err(errors, path, "sha256", "must be a lowercase 64-character SHA-256 hex string")
        return None
    return value


def _validate_probability(value: Any, path: str, errors: List[str]) -> Optional[float]:
    if not _is_number(value):
        _err(errors, path, "type", "must be a finite number")
        return None
    number = float(value)
    if number < 0.0 or number > 1.0:
        _err(errors, path, "range", "must be in [0, 1]")
        return None
    return number


def _evidence_sort_key(eid: str) -> Tuple[int, str]:
    match = EVIDENCE_ID_RE.fullmatch(str(eid))
    return (int(match.group(1)), str(eid)) if match else (10**12, str(eid))


def _validate_evidence_id_list(
    value: Any,
    path: str,
    errors: List[str],
    *,
    valid_ids: Optional[Set[str]] = None,
    allow_empty: bool = False,
) -> List[str]:
    if not isinstance(value, list):
        _err(errors, path, "type", "must be list[str]")
        return []
    ids: List[str] = []
    seen: Set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, str):
            _err(errors, item_path, "type", "must be a string")
            continue
        if not EVIDENCE_ID_RE.fullmatch(item):
            _err(errors, item_path, "format", "must match E1, E2, ...")
            continue
        if item in seen:
            _err(errors, item_path, "duplicate", f"duplicate evidence ID {item}")
            continue
        seen.add(item)
        ids.append(item)
        if valid_ids is not None and item not in valid_ids:
            _err(errors, item_path, "unknown", f"evidence ID {item} is not present in the sentence record")
    if not allow_empty and not ids:
        _err(errors, path, "empty", "must contain at least one evidence ID")
    if ids != sorted(ids, key=_evidence_sort_key):
        _err(errors, path, "order", "evidence IDs must be in natural E1, E2, ... order")
    return ids


def _validate_string_list(value: Any, path: str, errors: List[str]) -> List[str]:
    if not isinstance(value, list):
        _err(errors, path, "type", "must be list[str]")
        return []
    out: List[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            _err(errors, f"{path}[{index}]", "type", "must be a string")
        else:
            out.append(item)
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _input_id(record: Mapping[str, Any]) -> Optional[str]:
    value = record.get("input_id")
    return value if isinstance(value, str) and value.strip() else None


def _assert(kind: str, record: Mapping[str, Any], errors: Sequence[str]) -> None:
    if errors:
        raise RecordValidationError(kind, errors, _input_id(record))


# ---------------------------------------------------------------------------
# Sentence/evidence contract
# ---------------------------------------------------------------------------

def validate_sentence_record(record: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if not _is_mapping(record):
        return ["root: [type] must be an object"]

    _validate_nonempty_string(record.get("input_id"), "root.input_id", errors)
    raw_text = _validate_nonempty_string(record.get("raw_text"), "root.raw_text", errors)

    sentences_raw = record.get("sentences")
    sentences: Dict[str, str] = {}
    if not _is_mapping(sentences_raw):
        _err(errors, "root.sentences", "type", "must be an object mapping E-IDs to text")
    else:
        for key, value in sentences_raw.items():
            path = f"root.sentences.{key}"
            if not isinstance(key, str) or not EVIDENCE_ID_RE.fullmatch(key):
                _err(errors, path, "format", "key must match E1, E2, ...")
                continue
            text = _validate_nonempty_string(value, path, errors)
            if text is not None:
                sentences[key] = text
        expected = [f"E{i}" for i in range(1, len(sentences) + 1)]
        actual = sorted(sentences, key=_evidence_sort_key)
        if actual != expected:
            _err(errors, "root.sentences", "contiguous_ids", f"expected contiguous IDs {expected}, got {actual}")

    spans_raw = record.get("evidence_spans")
    spans: Dict[str, Mapping[str, Any]] = {}
    if not _is_mapping(spans_raw):
        _err(errors, "root.evidence_spans", "type", "must be an object")
    else:
        for eid, span in spans_raw.items():
            path = f"root.evidence_spans.{eid}"
            if not isinstance(eid, str) or not EVIDENCE_ID_RE.fullmatch(eid):
                _err(errors, path, "format", "key must match E1, E2, ...")
                continue
            if not _is_mapping(span):
                _err(errors, path, "type", "must be an object")
                continue
            spans[eid] = span
            start = span.get("start")
            end = span.get("end")
            if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                _err(errors, f"{path}.start", "range", "must be a non-negative integer")
            if not isinstance(end, int) or isinstance(end, bool) or end < 0:
                _err(errors, f"{path}.end", "range", "must be a non-negative integer")
            if isinstance(start, int) and isinstance(end, int) and end <= start:
                _err(errors, path, "span", "end must be greater than start")
            digest = _validate_sha256(span.get("text_sha256"), f"{path}.text_sha256", errors)
            text = sentences.get(eid)
            if text is not None and digest is not None and digest != _sha256_text(text):
                _err(errors, f"{path}.text_sha256", "hash_mismatch", "does not match evidence text")
            if raw_text is not None and text is not None and isinstance(start, int) and isinstance(end, int):
                if 0 <= start < end <= len(raw_text) and raw_text[start:end] != text:
                    _err(errors, path, "span_mismatch", "raw_text[start:end] does not equal evidence text")
                elif end > len(raw_text):
                    _err(errors, path, "span_range", "span exceeds raw_text length")

    if set(sentences) != set(spans):
        _err(
            errors,
            "root.evidence_spans",
            "id_mismatch",
            f"span IDs {sorted(spans, key=_evidence_sort_key)} do not match sentence IDs {sorted(sentences, key=_evidence_sort_key)}",
        )

    if raw_text is not None and sentences:
        reconstructed = " ".join(sentences[eid] for eid in sorted(sentences, key=_evidence_sort_key))
        if reconstructed != raw_text:
            _err(errors, "root.sentences", "reconstruction", "ordered evidence text does not reconstruct raw_text")

    segmentation = record.get("segmentation")
    if not _is_mapping(segmentation):
        _err(errors, "root.segmentation", "type", "must be an object")
    else:
        _validate_nonempty_string(segmentation.get("version"), "root.segmentation.version", errors)
        _validate_nonempty_string(segmentation.get("mode"), "root.segmentation.mode", errors)
        _validate_nonempty_string(segmentation.get("source_text_field"), "root.segmentation.source_text_field", errors)
        source_hash = _validate_sha256(
            segmentation.get("source_text_sha256"), "root.segmentation.source_text_sha256", errors
        )
        reconstruction_hash = _validate_sha256(
            segmentation.get("reconstruction_sha256"), "root.segmentation.reconstruction_sha256", errors
        )
        count = segmentation.get("evidence_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            _err(errors, "root.segmentation.evidence_count", "range", "must be a non-negative integer")
        elif count != len(sentences):
            _err(errors, "root.segmentation.evidence_count", "count_mismatch", f"expected {len(sentences)}, got {count}")
        if raw_text is not None and source_hash is not None and source_hash != _sha256_text(raw_text):
            _err(errors, "root.segmentation.source_text_sha256", "hash_mismatch", "does not match raw_text")
        if raw_text is not None and reconstruction_hash is not None and reconstruction_hash != _sha256_text(raw_text):
            _err(errors, "root.segmentation.reconstruction_sha256", "hash_mismatch", "does not match reconstructed text")
        if not _is_mapping(segmentation.get("parameters")):
            _err(errors, "root.segmentation.parameters", "type", "must be an object")

    if "provenance" in record and not _is_mapping(record.get("provenance")):
        _err(errors, "root.provenance", "type", "must be an object when present")
    return errors


def assert_valid_sentence_record(record: Mapping[str, Any]) -> None:
    _assert("sentences", record, validate_sentence_record(record))


# ---------------------------------------------------------------------------
# Extraction contract
# ---------------------------------------------------------------------------

_EXTRACTION_COLLECTIONS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "preconditions": (("condition",), ()),
    "entry": (("vector", "detail"), ()),
    "vuln_type": (("type", "subtype"), ()),
    "behaviors": (("action",), ("target", "impact")),
    "impacts": (("type",), ("detail",)),
}
_RELATION_REF_RE = re.compile(r"^(?:P|EN|VT|B|I)[1-9]\d*$")
_RELATION_PREFIX = {"P": "Precondition", "EN": "Entry", "VT": "VulnType", "B": "Behavior", "I": "Impact"}
_ALLOWED_RELATION_LAYERS: Set[Tuple[str, str, str]] = {
    ("Precondition", "enables", "Entry"),
    ("Precondition", "enables", "Behavior"),
    ("Entry", "characterized_by", "VulnType"),
    ("Entry", "enables", "Behavior"),
    ("VulnType", "enables", "Behavior"),
    ("Behavior", "causes", "Impact"),
}


def _relation_ref_type(ref: str) -> Optional[str]:
    match = re.match(r"^(EN|VT|P|B|I)", ref)
    return _RELATION_PREFIX.get(match.group(1)) if match else None


def validate_extraction_record(
    record: Mapping[str, Any],
    *,
    valid_evidence_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    errors: List[str] = []
    if not _is_mapping(record):
        return ["root: [type] must be an object"]
    _validate_nonempty_string(record.get("input_id"), "root.input_id", errors)
    valid_ids = set(valid_evidence_ids) if valid_evidence_ids is not None else None

    collection_sizes: Dict[str, int] = {}
    for field_name, (required_strings, optional_strings) in _EXTRACTION_COLLECTIONS.items():
        value = record.get(field_name)
        if not isinstance(value, list):
            _err(errors, f"root.{field_name}", "type", "must be a list")
            collection_sizes[field_name] = 0
            continue
        collection_sizes[field_name] = len(value)
        for index, item in enumerate(value):
            path = f"root.{field_name}[{index}]"
            if not _is_mapping(item):
                _err(errors, path, "type", "must be an object")
                continue
            for key in required_strings:
                _validate_nonempty_string(item.get(key), f"{path}.{key}", errors)
            for key in optional_strings:
                optional = item.get(key)
                if optional is not None and not isinstance(optional, str):
                    _err(errors, f"{path}.{key}", "type", "must be a string or null")
            _validate_evidence_id_list(
                item.get("evidence_ids"), f"{path}.evidence_ids", errors, valid_ids=valid_ids
            )
            _validate_probability(item.get("confidence"), f"{path}.confidence", errors)

    relations = record.get("relations")
    if not isinstance(relations, list):
        _err(errors, "root.relations", "type", "must be a list")
    else:
        valid_refs: Set[str] = set()
        valid_refs.update(f"P{i}" for i in range(1, collection_sizes.get("preconditions", 0) + 1))
        valid_refs.update(f"EN{i}" for i in range(1, collection_sizes.get("entry", 0) + 1))
        valid_refs.update(f"VT{i}" for i in range(1, collection_sizes.get("vuln_type", 0) + 1))
        valid_refs.update(f"B{i}" for i in range(1, collection_sizes.get("behaviors", 0) + 1))
        valid_refs.update(f"I{i}" for i in range(1, collection_sizes.get("impacts", 0) + 1))
        seen_relations: Set[Tuple[str, str, str]] = set()
        for index, relation in enumerate(relations):
            path = f"root.relations[{index}]"
            if not _is_mapping(relation):
                _err(errors, path, "type", "must be an object")
                continue
            src = _validate_nonempty_string(relation.get("src"), f"{path}.src", errors)
            dst = _validate_nonempty_string(relation.get("dst"), f"{path}.dst", errors)
            rel_type = _validate_nonempty_string(relation.get("type"), f"{path}.type", errors)
            if src is not None:
                if not _RELATION_REF_RE.fullmatch(src):
                    _err(errors, f"{path}.src", "format", "must be Pn, ENn, VTn, Bn, or In")
                elif src not in valid_refs:
                    _err(errors, f"{path}.src", "unknown", f"does not resolve to an extracted node: {src}")
            if dst is not None:
                if not _RELATION_REF_RE.fullmatch(dst):
                    _err(errors, f"{path}.dst", "format", "must be Pn, ENn, VTn, Bn, or In")
                elif dst not in valid_refs:
                    _err(errors, f"{path}.dst", "unknown", f"does not resolve to an extracted node: {dst}")
            if rel_type is not None and rel_type not in STRUCTURAL_EDGE_TYPES:
                _err(errors, f"{path}.type", "enum", f"must be one of {list(STRUCTURAL_EDGE_TYPES)}")
            if src and dst and rel_type:
                layer_tuple = (_relation_ref_type(src), rel_type, _relation_ref_type(dst))
                if layer_tuple not in _ALLOWED_RELATION_LAYERS:
                    _err(errors, path, "invalid_layer", f"unsupported relation layer {src} -{rel_type}-> {dst}")
                key = (src, rel_type, dst)
                if key in seen_relations:
                    _err(errors, path, "duplicate", f"duplicate relation {src} -{rel_type}-> {dst}")
                seen_relations.add(key)
            _validate_evidence_id_list(
                relation.get("evidence_ids"), f"{path}.evidence_ids", errors, valid_ids=valid_ids
            )
            _validate_probability(relation.get("confidence"), f"{path}.confidence", errors)

    if "_used_llm" in record and not isinstance(record.get("_used_llm"), bool):
        _err(errors, "root._used_llm", "type", "must be boolean when present")
    if "_validation_errors" in record:
        _validate_string_list(record.get("_validation_errors"), "root._validation_errors", errors)
    if "_provenance" in record and not _is_mapping(record.get("_provenance")):
        _err(errors, "root._provenance", "type", "must be an object when present")
    return errors


def assert_valid_extraction_record(
    record: Mapping[str, Any], *, valid_evidence_ids: Optional[Iterable[str]] = None
) -> None:
    _assert(
        "extraction",
        record,
        validate_extraction_record(record, valid_evidence_ids=valid_evidence_ids),
    )


# ---------------------------------------------------------------------------
# Local graph contract
# ---------------------------------------------------------------------------

def _validate_graph_components(
    nodes_raw: Any,
    edges_raw: Any,
    errors: List[str],
    *,
    path_prefix: str = "root",
) -> Tuple[Dict[str, Mapping[str, Any]], List[Mapping[str, Any]]]:
    node_map: Dict[str, Mapping[str, Any]] = {}
    evidence_nodes: Dict[str, str] = {}
    if not isinstance(nodes_raw, list):
        _err(errors, f"{path_prefix}.nodes", "type", "must be a list")
    else:
        for index, node in enumerate(nodes_raw):
            path = f"{path_prefix}.nodes[{index}]"
            if not _is_mapping(node):
                _err(errors, path, "type", "must be an object")
                continue
            node_id = _validate_nonempty_string(node.get("id"), f"{path}.id", errors)
            node_type = _validate_nonempty_string(node.get("type"), f"{path}.type", errors)
            _validate_nonempty_string(node.get("text"), f"{path}.text", errors)
            if node_id is not None:
                if node_id in node_map:
                    _err(errors, f"{path}.id", "duplicate", f"duplicate node ID {node_id}")
                node_map[node_id] = node
            if node_type is not None and node_type not in GRAPH_NODE_TYPES:
                _err(errors, f"{path}.type", "enum", f"must be one of {list(GRAPH_NODE_TYPES)}")
            if node_type == "CVE" and node_id and not node_id.startswith("CVE::"):
                _err(errors, f"{path}.id", "format", "CVE node ID must start with CVE::")
            elif node_type == "Evidence":
                eid = node.get("evidence_id")
                if not isinstance(eid, str) or not EVIDENCE_ID_RE.fullmatch(eid):
                    _err(errors, f"{path}.evidence_id", "format", "must match E1, E2, ...")
                else:
                    if eid in evidence_nodes:
                        _err(errors, f"{path}.evidence_id", "duplicate", f"duplicate Evidence node for {eid}")
                    evidence_nodes[eid] = node_id or ""
                if node_id and eid and node_id != f"EVIDENCE::{eid}":
                    _err(errors, f"{path}.id", "format", f"expected EVIDENCE::{eid}")
            elif node_type in STRUCTURAL_NODE_TYPES:
                _validate_evidence_id_list(
                    node.get("evidence_ids"), f"{path}.evidence_ids", errors, allow_empty=True
                )
                if "confidence" in node:
                    _validate_probability(node.get("confidence"), f"{path}.confidence", errors)

    edges: List[Mapping[str, Any]] = []
    seen_edges: Set[Tuple[str, str, str, str]] = set()
    if not isinstance(edges_raw, list):
        _err(errors, f"{path_prefix}.edges", "type", "must be a list")
    else:
        for index, edge in enumerate(edges_raw):
            path = f"{path_prefix}.edges[{index}]"
            if not _is_mapping(edge):
                _err(errors, path, "type", "must be an object")
                continue
            edges.append(edge)
            src = _validate_nonempty_string(edge.get("src"), f"{path}.src", errors)
            dst = _validate_nonempty_string(edge.get("dst"), f"{path}.dst", errors)
            edge_type = _validate_nonempty_string(edge.get("type"), f"{path}.type", errors)
            origin = _validate_nonempty_string(edge.get("origin"), f"{path}.origin", errors)
            if src is not None and src not in node_map:
                _err(errors, f"{path}.src", "dangling", f"source node {src} does not exist")
            if dst is not None and dst not in node_map:
                _err(errors, f"{path}.dst", "dangling", f"destination node {dst} does not exist")
            if edge_type is not None and edge_type not in GRAPH_EDGE_TYPES:
                _err(errors, f"{path}.type", "enum", f"must be one of {list(GRAPH_EDGE_TYPES)}")
            if src and dst and edge_type and origin:
                key = (src, edge_type, dst, origin)
                if key in seen_edges:
                    _err(errors, path, "duplicate", f"duplicate edge {src} -{edge_type}-> {dst} ({origin})")
                seen_edges.add(key)
            if edge_type == "mentions" and src and dst:
                if node_map.get(src, {}).get("type") != "CVE" or node_map.get(dst, {}).get("type") not in STRUCTURAL_NODE_TYPES:
                    _err(errors, path, "layer", "mentions must connect CVE -> structural node")
            elif edge_type == "supported_by" and src and dst:
                if node_map.get(src, {}).get("type") not in STRUCTURAL_NODE_TYPES or node_map.get(dst, {}).get("type") != "Evidence":
                    _err(errors, path, "layer", "supported_by must connect structural node -> Evidence")
            elif edge_type in STRUCTURAL_EDGE_TYPES and src and dst:
                src_type = node_map.get(src, {}).get("type")
                dst_type = node_map.get(dst, {}).get("type")
                if (src_type, edge_type, dst_type) not in _ALLOWED_RELATION_LAYERS:
                    _err(errors, path, "layer", f"invalid structural layer {src_type} -{edge_type}-> {dst_type}")
            if "evidence_ids" in edge:
                _validate_evidence_id_list(
                    edge.get("evidence_ids"), f"{path}.evidence_ids", errors, allow_empty=True
                )
            if "shared_evidence_ids" in edge:
                _validate_evidence_id_list(
                    edge.get("shared_evidence_ids"), f"{path}.shared_evidence_ids", errors, allow_empty=True
                )
            for score_field in (
                "confidence",
                "structural_score",
                "evidence_jaccard",
                "lexical_jaccard",
                "confidence_mean",
            ):
                if score_field in edge:
                    _validate_probability(edge.get(score_field), f"{path}.{score_field}", errors)

    # Cross-check structural-node evidence_ids against supported_by edges.
    support_pairs: Set[Tuple[str, str]] = set()
    for edge in edges:
        if edge.get("type") == "supported_by":
            dst_node = node_map.get(str(edge.get("dst")), {})
            eid = dst_node.get("evidence_id")
            if isinstance(eid, str):
                support_pairs.add((str(edge.get("src")), eid))
    for node_id, node in node_map.items():
        if node.get("type") not in STRUCTURAL_NODE_TYPES:
            continue
        for eid in node.get("evidence_ids") or []:
            if isinstance(eid, str) and (node_id, eid) not in support_pairs:
                _err(
                    errors,
                    f"{path_prefix}.nodes[{node_id}].evidence_ids",
                    "missing_edge",
                    f"evidence {eid} has no supported_by edge",
                )
    return node_map, edges


def validate_local_graph_record(record: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if not _is_mapping(record):
        return ["root: [type] must be an object"]
    input_id = _validate_nonempty_string(record.get("input_id"), "root.input_id", errors)
    _validate_nonempty_string(record.get("graph_version"), "root.graph_version", errors)
    node_map, edges = _validate_graph_components(record.get("nodes"), record.get("edges"), errors)

    cve_nodes = [node for node in node_map.values() if node.get("type") == "CVE"]
    if len(cve_nodes) != 1:
        _err(errors, "root.nodes", "cve_count", f"expected exactly one CVE node, got {len(cve_nodes)}")
    elif input_id is not None and cve_nodes[0].get("id") != f"CVE::{input_id}":
        _err(errors, "root.nodes", "cve_id", f"expected CVE::{input_id}")

    stats = record.get("stats")
    if not _is_mapping(stats):
        _err(errors, "root.stats", "type", "must be an object")
    else:
        node_count = stats.get("node_count")
        edge_count = stats.get("edge_count")
        if node_count != len(node_map):
            _err(errors, "root.stats.node_count", "count_mismatch", f"expected {len(node_map)}, got {node_count}")
        if edge_count != len(edges):
            _err(errors, "root.stats.edge_count", "count_mismatch", f"expected {len(edges)}, got {edge_count}")
        type_counts = stats.get("node_type_counts")
        if not _is_mapping(type_counts):
            _err(errors, "root.stats.node_type_counts", "type", "must be an object")
        else:
            for node_type in GRAPH_NODE_TYPES:
                actual = sum(node.get("type") == node_type for node in node_map.values())
                if type_counts.get(node_type) != actual:
                    _err(
                        errors,
                        f"root.stats.node_type_counts.{node_type}",
                        "count_mismatch",
                        f"expected {actual}, got {type_counts.get(node_type)}",
                    )
    _validate_string_list(record.get("warnings"), "root.warnings", errors)
    return errors


def assert_valid_local_graph_record(record: Mapping[str, Any]) -> None:
    _assert("local_graph", record, validate_local_graph_record(record))


# ---------------------------------------------------------------------------
# MES contract
# ---------------------------------------------------------------------------

def validate_mes_record(
    record: Mapping[str, Any],
    *,
    source_graph: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    errors: List[str] = []
    if not _is_mapping(record):
        return ["root: [type] must be an object"]
    input_id = _validate_nonempty_string(record.get("input_id"), "root.input_id", errors)
    _validate_nonempty_string(record.get("algorithm"), "root.algorithm", errors)
    if record.get("source_graph_version") is not None:
        _validate_nonempty_string(record.get("source_graph_version"), "root.source_graph_version", errors)
    if not _is_mapping(record.get("parameters")):
        _err(errors, "root.parameters", "type", "must be an object")

    status = record.get("status")
    if status not in MES_STATUSES:
        _err(errors, "root.status", "enum", f"must be one of {list(MES_STATUSES)}")
    complete = record.get("complete_core_chain")
    if not isinstance(complete, bool):
        _err(errors, "root.complete_core_chain", "type", "must be boolean")
    elif status == "complete" and not complete:
        _err(errors, "root.complete_core_chain", "status_mismatch", "must be true when status=complete")
    elif status != "complete" and complete:
        _err(errors, "root.complete_core_chain", "status_mismatch", "must be false unless status=complete")

    node_map, edges = _validate_graph_components(record.get("nodes"), record.get("edges"), errors)
    chain = _validate_string_list(record.get("chain"), "root.chain", errors)
    structural_ids = _validate_string_list(
        record.get("structural_node_ids"), "root.structural_node_ids", errors
    )
    if chain != structural_ids:
        _err(errors, "root.structural_node_ids", "chain_mismatch", "must exactly equal chain")
    for index, node_id in enumerate(structural_ids):
        node = node_map.get(node_id)
        if node is None:
            _err(errors, f"root.structural_node_ids[{index}]", "unknown", f"node {node_id} is absent from MES nodes")
        elif node.get("type") not in STRUCTURAL_NODE_TYPES:
            _err(errors, f"root.structural_node_ids[{index}]", "type", f"node {node_id} is not structural")

    chain_types = record.get("chain_types")
    if chain_types is not None:
        chain_type_list = _validate_string_list(chain_types, "root.chain_types", errors)
        expected_types = [str(node_map.get(node_id, {}).get("type", "")) for node_id in structural_ids]
        if chain_type_list != expected_types:
            _err(errors, "root.chain_types", "type_mismatch", f"expected {expected_types}, got {chain_type_list}")

    evidence_ids = _validate_evidence_id_list(
        record.get("evidence_ids"), "root.evidence_ids", errors, allow_empty=status == "empty"
    )
    evidence_in_nodes = {
        str(node.get("evidence_id"))
        for node in node_map.values()
        if node.get("type") == "Evidence" and isinstance(node.get("evidence_id"), str)
    }
    if set(evidence_ids) != evidence_in_nodes:
        _err(
            errors,
            "root.evidence_ids",
            "node_mismatch",
            f"expected Evidence-node IDs {sorted(evidence_in_nodes, key=_evidence_sort_key)}, got {evidence_ids}",
        )

    if status == "empty":
        if node_map or edges or structural_ids or evidence_ids:
            _err(errors, "root", "empty_status", "status=empty requires empty nodes, edges, chain, and evidence_ids")
        if record.get("compact_text") not in ("", None):
            _err(errors, "root.compact_text", "empty_status", "must be empty when status=empty")
    else:
        if not structural_ids:
            _err(errors, "root.structural_node_ids", "empty", "non-empty MES must contain structural nodes")
        cve_count = sum(node.get("type") == "CVE" for node in node_map.values())
        if cve_count != 1:
            _err(errors, "root.nodes", "cve_count", f"non-empty MES must contain exactly one CVE node, got {cve_count}")
        _validate_nonempty_string(record.get("compact_text"), "root.compact_text", errors)

    if complete:
        types = {node_map.get(node_id, {}).get("type") for node_id in structural_ids}
        missing = {"Entry", "Behavior", "Impact"} - types
        if missing:
            _err(errors, "root.structural_node_ids", "core_chain", f"complete MES is missing {sorted(missing)}")
        structural_edges = {
            (str(edge.get("src")), str(edge.get("type")), str(edge.get("dst")))
            for edge in edges
            if edge.get("type") in STRUCTURAL_EDGE_TYPES
        }
        entry_ids = [nid for nid in structural_ids if node_map.get(nid, {}).get("type") == "Entry"]
        behavior_ids = [nid for nid in structural_ids if node_map.get(nid, {}).get("type") == "Behavior"]
        impact_ids = [nid for nid in structural_ids if node_map.get(nid, {}).get("type") == "Impact"]
        if not any((entry, "enables", behavior) in structural_edges for entry in entry_ids for behavior in behavior_ids):
            _err(errors, "root.edges", "core_edge", "complete MES requires an Entry -enables-> Behavior edge")
        if not any((behavior, "causes", impact) in structural_edges for behavior in behavior_ids for impact in impact_ids):
            _err(errors, "root.edges", "core_edge", "complete MES requires a Behavior -causes-> Impact edge")

    if not _is_mapping(record.get("selection_trace")):
        _err(errors, "root.selection_trace", "type", "must be an object")
    _validate_string_list(record.get("warnings"), "root.warnings", errors)
    _validate_sha256(record.get("mes_sha256"), "root.mes_sha256", errors)

    if source_graph is not None:
        source_errors = validate_local_graph_record(source_graph)
        if source_errors:
            _err(errors, "source_graph", "invalid", f"source graph has {len(source_errors)} validation errors")
        if input_id is not None and source_graph.get("input_id") != input_id:
            _err(errors, "root.input_id", "source_mismatch", "does not match source graph input_id")
        if record.get("source_graph_version") != source_graph.get("graph_version"):
            _err(errors, "root.source_graph_version", "source_mismatch", "does not match source graph version")
        source_nodes = {
            str(node.get("id")): node
            for node in source_graph.get("nodes") or []
            if _is_mapping(node) and isinstance(node.get("id"), str)
        }
        source_edges = {
            (
                str(edge.get("src")),
                str(edge.get("type")),
                str(edge.get("dst")),
                str(edge.get("origin")),
            )
            for edge in source_graph.get("edges") or []
            if _is_mapping(edge)
        }
        for node_id, node in node_map.items():
            if node_id not in source_nodes:
                _err(errors, f"root.nodes[{node_id}]", "not_subgraph", "node is absent from source graph")
            elif dict(node) != dict(source_nodes[node_id]):
                _err(errors, f"root.nodes[{node_id}]", "modified_node", "node differs from source graph")
        for index, edge in enumerate(edges):
            key = (
                str(edge.get("src")),
                str(edge.get("type")),
                str(edge.get("dst")),
                str(edge.get("origin")),
            )
            if key not in source_edges:
                _err(errors, f"root.edges[{index}]", "not_subgraph", "edge is absent from source graph")
    return errors


def assert_valid_mes_record(
    record: Mapping[str, Any], *, source_graph: Optional[Mapping[str, Any]] = None
) -> None:
    _assert("mes", record, validate_mes_record(record, source_graph=source_graph))


# ---------------------------------------------------------------------------
# Candidate-retrieval contract
# ---------------------------------------------------------------------------

def _validate_technique_id(value: Any, path: str, errors: List[str]) -> Optional[str]:
    if not isinstance(value, str) or not TECHNIQUE_ID_RE.fullmatch(value):
        _err(errors, path, "format", "must be an ATT&CK technique ID such as T1190 or T1055.001")
        return None
    return value


def _validate_ranked_candidates(
    candidates: Any,
    errors: List[str],
    *,
    reranked: bool,
    beta: Optional[float] = None,
) -> List[Mapping[str, Any]]:
    if not isinstance(candidates, list):
        _err(errors, "root.candidates", "type", "must be a list")
        return []
    seen_ids: Set[str] = set()
    seen_ranks: Set[int] = set()
    out: List[Mapping[str, Any]] = []
    rank_field = "rerank_rank" if reranked else "rank"
    for index, candidate in enumerate(candidates):
        path = f"root.candidates[{index}]"
        if not _is_mapping(candidate):
            _err(errors, path, "type", "must be an object")
            continue
        out.append(candidate)
        technique_id = _validate_technique_id(candidate.get("technique_id"), f"{path}.technique_id", errors)
        if technique_id is not None:
            if technique_id in seen_ids:
                _err(errors, f"{path}.technique_id", "duplicate", f"duplicate candidate {technique_id}")
            seen_ids.add(technique_id)
        for score_field in ("score_fused", "score_text", "score_structure"):
            _validate_probability(candidate.get(score_field), f"{path}.{score_field}", errors)
        if "score_graph" in candidate:
            graph_score = _validate_probability(candidate.get("score_graph"), f"{path}.score_graph", errors)
            structure_score = candidate.get("score_structure")
            if graph_score is not None and _is_number(structure_score) and abs(graph_score - float(structure_score)) > 1e-10:
                _err(errors, f"{path}.score_graph", "alias_mismatch", "must equal score_structure")
        rank = candidate.get(rank_field)
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            _err(errors, f"{path}.{rank_field}", "range", "must be a positive integer")
        else:
            if rank in seen_ranks:
                _err(errors, f"{path}.{rank_field}", "duplicate", f"duplicate rank {rank}")
            seen_ranks.add(rank)
        if not reranked:
            expected_rank = index + 1
            if rank != expected_rank:
                _err(errors, f"{path}.rank", "order", f"expected rank {expected_rank} at list position {index}")
            continue

        retrieval_rank = candidate.get("retrieval_rank")
        if not isinstance(retrieval_rank, int) or isinstance(retrieval_rank, bool) or retrieval_rank < 1:
            _err(errors, f"{path}.retrieval_rank", "range", "must be a positive integer")
        retrieval_score = _validate_probability(
            candidate.get("retrieval_score"), f"{path}.retrieval_score", errors
        )
        llm_score = _validate_probability(candidate.get("llm_score"), f"{path}.llm_score", errors)
        final_score = _validate_probability(candidate.get("final_score"), f"{path}.final_score", errors)
        _validate_nonempty_string(candidate.get("reason"), f"{path}.reason", errors)
        _validate_evidence_id_list(
            candidate.get("evidence_ids"), f"{path}.evidence_ids", errors, allow_empty=True
        )
        _validate_string_list(candidate.get("constraint_flags"), f"{path}.constraint_flags", errors)
        if candidate.get("rerank_rank") != index + 1:
            _err(errors, f"{path}.rerank_rank", "order", f"expected rerank_rank {index + 1}")
        if retrieval_score is not None and _is_number(candidate.get("score_fused")):
            if abs(retrieval_score - float(candidate["score_fused"])) > 1e-10:
                _err(errors, f"{path}.retrieval_score", "score_mismatch", "must equal score_fused")
        if beta is not None and retrieval_score is not None and llm_score is not None and final_score is not None:
            expected = beta * retrieval_score + (1.0 - beta) * llm_score
            if abs(final_score - expected) > 5e-10:
                _err(
                    errors,
                    f"{path}.final_score",
                    "formula",
                    f"expected beta*retrieval+(1-beta)*llm = {expected:.12f}, got {final_score:.12f}",
                )
    expected_ranks = set(range(1, len(out) + 1))
    if seen_ranks != expected_ranks:
        _err(errors, "root.candidates", "rank_set", f"expected ranks {sorted(expected_ranks)}, got {sorted(seen_ranks)}")
    return out


def validate_candidate_record(record: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if not _is_mapping(record):
        return ["root: [type] must be an object"]
    _validate_nonempty_string(record.get("input_id"), "root.input_id", errors)
    candidates = _validate_ranked_candidates(record.get("candidates"), errors, reranked=False)
    metadata = record.get("retrieval_metadata")
    if not _is_mapping(metadata):
        _err(errors, "root.retrieval_metadata", "type", "must be an object")
    else:
        _validate_nonempty_string(metadata.get("version"), "root.retrieval_metadata.version", errors)
        alpha = _validate_probability(metadata.get("alpha"), "root.retrieval_metadata.alpha", errors)
        _validate_nonempty_string(
            metadata.get("score_normalization"), "root.retrieval_metadata.score_normalization", errors
        )
        _validate_nonempty_string(
            metadata.get("text_query_source"), "root.retrieval_metadata.text_query_source", errors
        )
        _validate_sha256(
            metadata.get("text_query_sha256"), "root.retrieval_metadata.text_query_sha256", errors
        )
        _validate_sha256(
            metadata.get("structure_query_sha256"), "root.retrieval_metadata.structure_query_sha256", errors
        )
        if not isinstance(metadata.get("structure_query_empty"), bool):
            _err(errors, "root.retrieval_metadata.structure_query_empty", "type", "must be boolean")
        if not isinstance(metadata.get("complete_core_chain"), bool):
            _err(errors, "root.retrieval_metadata.complete_core_chain", "type", "must be boolean")
        if metadata.get("mes_sha256") is not None:
            _validate_sha256(metadata.get("mes_sha256"), "root.retrieval_metadata.mes_sha256", errors)
        if not isinstance(metadata.get("parent_normalization"), bool):
            _err(errors, "root.retrieval_metadata.parent_normalization", "type", "must be boolean")
        # Verify the documented fusion equation when alpha is available.
        if alpha is not None:
            for index, candidate in enumerate(candidates):
                if all(_is_number(candidate.get(key)) for key in ("score_text", "score_structure", "score_fused")):
                    expected = alpha * float(candidate["score_text"]) + (1.0 - alpha) * float(candidate["score_structure"])
                    if abs(float(candidate["score_fused"]) - expected) > 5e-10:
                        _err(
                            errors,
                            f"root.candidates[{index}].score_fused",
                            "formula",
                            f"expected alpha*text+(1-alpha)*structure = {expected:.12f}",
                        )
    return errors


def assert_valid_candidate_record(record: Mapping[str, Any]) -> None:
    _assert("candidates", record, validate_candidate_record(record))


# ---------------------------------------------------------------------------
# Reranking contract
# ---------------------------------------------------------------------------

def validate_rerank_record(
    record: Mapping[str, Any],
    *,
    source_candidates: Optional[Mapping[str, Any]] = None,
    valid_evidence_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    errors: List[str] = []
    if not _is_mapping(record):
        return ["root: [type] must be an object"]
    input_id = _validate_nonempty_string(record.get("input_id"), "root.input_id", errors)
    metadata = record.get("rerank_metadata")
    beta: Optional[float] = None
    mode: Optional[str] = None
    if not _is_mapping(metadata):
        _err(errors, "root.rerank_metadata", "type", "must be an object")
    else:
        _validate_nonempty_string(metadata.get("version"), "root.rerank_metadata.version", errors)
        _validate_nonempty_string(metadata.get("prompt_version"), "root.rerank_metadata.prompt_version", errors)
        mode_value = metadata.get("mode")
        if mode_value not in RERANK_MODES:
            _err(errors, "root.rerank_metadata.mode", "enum", f"must be one of {list(RERANK_MODES)}")
        else:
            mode = str(mode_value)
        _validate_nonempty_string(metadata.get("model_requested"), "root.rerank_metadata.model_requested", errors)
        _validate_probability(metadata.get("temperature"), "root.rerank_metadata.temperature", errors)
        if not isinstance(metadata.get("seed"), int) or isinstance(metadata.get("seed"), bool):
            _err(errors, "root.rerank_metadata.seed", "type", "must be an integer")
        beta = _validate_probability(metadata.get("beta"), "root.rerank_metadata.beta", errors)
        _validate_sha256(
            metadata.get("candidate_set_sha256"), "root.rerank_metadata.candidate_set_sha256", errors
        )
        _validate_sha256(metadata.get("prompt_sha256"), "root.rerank_metadata.prompt_sha256", errors)
        if metadata.get("mes_sha256") is not None:
            _validate_sha256(metadata.get("mes_sha256"), "root.rerank_metadata.mes_sha256", errors)
        if not _is_mapping(metadata.get("parse_stats")):
            _err(errors, "root.rerank_metadata.parse_stats", "type", "must be an object")

    candidates = _validate_ranked_candidates(
        record.get("candidates"), errors, reranked=True, beta=beta
    )
    valid_ids = set(valid_evidence_ids) if valid_evidence_ids is not None else None
    for index, candidate in enumerate(candidates):
        evidence_ids = candidate.get("evidence_ids")
        if isinstance(evidence_ids, list) and valid_ids is not None:
            for eid in evidence_ids:
                if isinstance(eid, str) and eid not in valid_ids:
                    _err(errors, f"root.candidates[{index}].evidence_ids", "unknown", f"unknown evidence ID {eid}")
        if mode == "generic" and evidence_ids:
            _err(errors, f"root.candidates[{index}].evidence_ids", "mode", "generic mode must not cite evidence IDs")

    if source_candidates is not None:
        source_errors = validate_candidate_record(source_candidates)
        if source_errors:
            _err(errors, "source_candidates", "invalid", f"source candidate record has {len(source_errors)} errors")
        if input_id is not None and source_candidates.get("input_id") != input_id:
            _err(errors, "root.input_id", "source_mismatch", "does not match source candidates input_id")
        source_list = source_candidates.get("candidates") or []
        source_ids = [item.get("technique_id") for item in source_list if _is_mapping(item)]
        output_by_retrieval_rank = sorted(
            (item for item in candidates if isinstance(item.get("retrieval_rank"), int)),
            key=lambda item: int(item["retrieval_rank"]),
        )
        output_ids = [item.get("technique_id") for item in output_by_retrieval_rank]
        if output_ids != source_ids[: len(output_ids)]:
            _err(errors, "root.candidates", "candidate_set", "reranking changed the source candidate IDs/order")
    return errors


def assert_valid_rerank_record(
    record: Mapping[str, Any],
    *,
    source_candidates: Optional[Mapping[str, Any]] = None,
    valid_evidence_ids: Optional[Iterable[str]] = None,
) -> None:
    _assert(
        "reranking",
        record,
        validate_rerank_record(
            record,
            source_candidates=source_candidates,
            valid_evidence_ids=valid_evidence_ids,
        ),
    )


# ---------------------------------------------------------------------------
# Generic dispatch and compatibility helpers
# ---------------------------------------------------------------------------

VALIDATORS = {
    "sentences": validate_sentence_record,
    "extraction": validate_extraction_record,
    "local_graph": validate_local_graph_record,
    "mes": validate_mes_record,
    "candidates": validate_candidate_record,
    "reranking": validate_rerank_record,
}


def validate_record(kind: RecordKind, record: Mapping[str, Any], **kwargs: Any) -> List[str]:
    """Validate one record using the named stage contract."""
    try:
        validator = VALIDATORS[kind]
    except KeyError as exc:  # defensive for dynamically supplied strings
        raise ValueError(f"Unknown record kind: {kind}") from exc
    return validator(record, **kwargs)


def assert_valid_record(kind: RecordKind, record: Mapping[str, Any], **kwargs: Any) -> None:
    errors = validate_record(kind, record, **kwargs)
    _assert(kind, record, errors)


def validate_evidence_ids(extraction: Dict[str, Any], valid_ids: Set[str]) -> List[str]:
    """Compatibility validator used by :mod:`pgt.extract`.

    This deliberately checks every nested ``evidence_ids`` field, including
    provenance extensions, while preserving the original return type
    (``list[str]``).  Stage-specific validation should use
    :func:`validate_extraction_record` for stronger structural checks.
    """
    errors: List[str] = []

    def check_obj(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            if "evidence_ids" in obj:
                eids = obj["evidence_ids"]
                if not isinstance(eids, list) or not all(isinstance(x, str) for x in eids):
                    errors.append(f"{path}.evidence_ids not list[str]")
                else:
                    bad_format = [x for x in eids if not EVIDENCE_ID_RE.fullmatch(x)]
                    if bad_format:
                        errors.append(f"{path}.evidence_ids contains malformed ids: {bad_format}")
                    bad = [x for x in eids if x not in valid_ids]
                    if bad:
                        errors.append(f"{path}.evidence_ids contains invalid ids: {bad}")
                    if len(eids) != len(set(eids)):
                        errors.append(f"{path}.evidence_ids contains duplicates")
            for key, value in obj.items():
                check_obj(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                check_obj(value, f"{path}[{index}]")

    check_obj(extraction, "root")
    return errors


__all__ = [
    "Behavior",
    "CandidateRecord",
    "CandidateRecordItem",
    "EvidenceId",
    "EvidenceRef",
    "Extraction",
    "ExtractionRecord",
    "GRAPH_EDGE_TYPES",
    "GRAPH_NODE_TYPES",
    "LocalGraphRecord",
    "MESRecord",
    "MES_STATUSES",
    "RecordKind",
    "RecordValidationError",
    "Relation",
    "RERANK_MODES",
    "RerankRecord",
    "RerankedCandidateRecordItem",
    "SCHEMA_VERSIONS",
    "STRUCTURAL_EDGE_TYPES",
    "STRUCTURAL_NODE_TYPES",
    "SentenceRecord",
    "TechniqueId",
    "assert_valid_candidate_record",
    "assert_valid_extraction_record",
    "assert_valid_local_graph_record",
    "assert_valid_mes_record",
    "assert_valid_record",
    "assert_valid_rerank_record",
    "assert_valid_sentence_record",
    "validate_candidate_record",
    "validate_evidence_ids",
    "validate_extraction_record",
    "validate_local_graph_record",
    "validate_mes_record",
    "validate_record",
    "validate_rerank_record",
    "validate_sentence_record",
]
