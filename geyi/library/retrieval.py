"""Exact library recall for Strategy 0."""

from __future__ import annotations

from typing import Any, Dict, List

from geyi.contract.model import SemanticContract
from geyi.library.index import normalize_key, search_library_index


def recall_exact_signature(contract: SemanticContract, index: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return exact op/signature matches.

    This is deliberately conservative: an op or alias must match exactly after
    normalization, and the contract shape/dtype counts must satisfy the hotset
    signature. Similar names are not returned as semantic equivalents.
    """

    if not contract.intents:
        return []
    intent = contract.intents[0]
    op_candidates = exact_op_candidates(intent.kind, intent.subkind)
    results = []
    for op in op_candidates:
        for item in search_library_index(index, op):
            signature = item["contract_signature"]
            if signature_matches_contract(signature, contract):
                result = dict(item)
                result["match_type"] = "exact_signature"
                result["strategy"] = "library"
                result["contract_hash"] = contract.contract_hash
                result["evidence"].append(
                    {
                        "kind": "contract",
                        "claim": "contract intent %s.%s matches hotset signature exactly"
                        % (intent.kind, intent.subkind),
                        "contract_hash": contract.contract_hash,
                    }
                )
                results.append(result)
    return dedupe_results(results)


def exact_op_candidates(kind: str, subkind: str) -> List[str]:
    candidates = [subkind]
    if kind == "reduce" and subkind == "row_sum":
        candidates.extend(["reduce_sum", "sum"])
    if kind == "transpose" and subkind == "2d_contiguous":
        candidates.append("transpose")
    if kind == "copy":
        candidates.append("copy")
    return candidates


def signature_matches_contract(signature: Dict[str, Any], contract: SemanticContract) -> bool:
    if not contract.intents:
        return False
    intent = contract.intents[0]
    sig_intent = signature.get("intent") or {}
    if sig_intent.get("kind") and str(sig_intent["kind"]) != intent.kind:
        return False
    sig_subkind = sig_intent.get("subkind")
    aliases = [normalize_key(item) for item in sig_intent.get("aliases", [])]
    if sig_subkind and normalize_key(sig_subkind) != normalize_key(intent.subkind):
        if normalize_key(intent.subkind) not in aliases:
            return False

    inputs = signature.get("inputs", [])
    outputs = signature.get("outputs", [])
    if inputs and len(inputs) != len(intent.inputs):
        return False
    if outputs and len(outputs) != len(intent.outputs):
        return False

    return tensors_match(inputs, intent.inputs, contract) and tensors_match(outputs, intent.outputs, contract)


def tensors_match(specs: List[Dict[str, Any]], names: List[str], contract: SemanticContract) -> bool:
    for spec, name in zip(specs, names):
        tensor = contract.tensors.get(name)
        if tensor is None:
            return False
        if "rank" in spec and len(tensor.shape) != int(spec["rank"]):
            return False
        dtypes = spec.get("dtypes") or spec.get("dtype")
        if isinstance(dtypes, str):
            dtypes = [dtypes]
        if dtypes and tensor.dtype not in {str(item) for item in dtypes}:
            return False
        access = spec.get("access")
        if access and tensor.access != str(access):
            return False
    return True


def dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in results:
        key = (item.get("op"), tuple(item.get("source_paths", [])))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return sorted(deduped, key=lambda item: (item["rank"], item["op"]))
