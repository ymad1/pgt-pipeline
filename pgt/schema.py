
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any

EvidenceId = str

@dataclass
class EvidenceRef:
    evidence_ids: List[EvidenceId]

@dataclass
class Behavior(EvidenceRef):
    action: str
    target: Optional[str] = None
    impact: Optional[str] = None
    # you can add: actor/tool/object, etc.
    confidence: Optional[float] = None

@dataclass
class Relation(EvidenceRef):
    src: str
    rel: Literal["enables","leads_to","supported_by","has_precondition","has_vulnerability"]
    dst: str

@dataclass
class Extraction:
    input_id: str
    preconditions: List[Dict[str, Any]] = field(default_factory=list)  # each must include evidence_ids
    entry: List[Dict[str, Any]] = field(default_factory=list)          # each must include evidence_ids
    vuln_type: List[Dict[str, Any]] = field(default_factory=list)      # each must include evidence_ids
    behaviors: List[Dict[str, Any]] = field(default_factory=list)      # Behavior-like dicts
    relations: List[Dict[str, Any]] = field(default_factory=list)      # Relation-like dicts
    impacts: List[Dict[str, Any]] = field(default_factory=list)        # each must include evidence_ids

def validate_evidence_ids(extraction: Dict[str, Any], valid_ids: set[str]) -> List[str]:
    """Return a list of errors (empty if ok)."""
    errors: List[str] = []
    def check_obj(obj: Any, path: str):
        if isinstance(obj, dict):
            if "evidence_ids" in obj:
                eids = obj["evidence_ids"]
                if not isinstance(eids, list) or not all(isinstance(x, str) for x in eids):
                    errors.append(f"{path}.evidence_ids not list[str]")
                else:
                    bad = [x for x in eids if x not in valid_ids]
                    if bad:
                        errors.append(f"{path}.evidence_ids contains invalid ids: {bad}")
            for k,v in obj.items():
                check_obj(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i,v in enumerate(obj):
                check_obj(v, f"{path}[{i}]")
    check_obj(extraction, "root")
    return errors
