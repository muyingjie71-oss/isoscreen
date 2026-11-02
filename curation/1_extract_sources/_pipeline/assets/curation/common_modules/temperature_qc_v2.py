"""Conservative internal temperature consistency audit; never replace measurements.

Call ``screen(records)`` with dictionaries containing Study_ID,
Solute_Identity, Solvent_Audit_ID, T_K, gamma_inf, canonical_record_id,
and Effective_Conditions in pressure/phase/standard-state/method/fixed order.
Series use maximal mutually compatible condition declarations observed across
all temperatures of the same study and molecular system. A partially reported
record joins a series only when exactly one observed signature is compatible.
This assignment is confined to QC; reported and effective metadata are intact.

The robust line is a QC reference, not a physical truth test or ML target.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import argparse
import csv
import hashlib
import json
import math

VERSION = "2.1"
CONDITION_FIELDS = ("pressure", "phase", "standard_state", "method", "fixed")
POLICY = {
    "review_factor": 3.0,
    "severe_factor": 10.0,
    "minimum_temperatures": 3,
    "peer_loo_max_factor": 1.5,
    "reference_prediction_stability_factor": 2.0,
    "three_temperature_extreme_factor": 100.0,
    "three_temperature_peer_gamma_spread_factor": 1.5,
    "maximum_endpoint_extrapolation_fraction": 1.0,
    "fit": "Theil-Sen median pair slope and median intercept in ln(gamma) versus 1/T; leave-one-out",
    "grouping": "Study + full solute identity + audited solvent identity + uniquely compatible maximal observed condition signature across temperatures",
    "condition_assignment": "Only QC series metadata may be completed from a unique compatible signature; unknown declarations never bridge conflicting methods/pressures/phases/standard/fixed conditions",
    "no_iteration": "At most one uniquely isolated point may be excluded from one series",
    "three_point_protection": "3-temperature exclusion additionally needs factor>=100 and peer gamma spread<=1.5; this is a modelling exclusion, not a confirmed source error",
    "source_values": "Input values are never modified, imputed, or replaced with trend predictions",
}


def _stable(prefix, parts):
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _factor(log_difference):
    return math.exp(min(700.0, abs(log_difference)))


def _numeric_key(value):
    # Remove floating serialisation noise only; retain nominal 0.01/0.1 K differences.
    return format(float(value), ".12g")


def _conditions(record):
    conditions = record.get("Effective_Conditions", ["", "", "", "", ""])
    if isinstance(conditions, str):
        conditions = json.loads(conditions)
    if isinstance(conditions, dict):
        conditions = [conditions.get("pressure", conditions.get("pressure_kPa", "")),
                      conditions.get("phase", ""), conditions.get("standard_state", ""),
                      conditions.get("method", ""), conditions.get("fixed", conditions.get("fixed_conditions", ""))]
    if len(conditions) != 5:
        raise ValueError("Effective_Conditions must have five ordered entries")
    # Caller owns scientifically justified metadata normalisation. Do not fuzzy-merge.
    return tuple("" if value is None else str(value).strip() for value in conditions)


def _study_known(record):
    if "Study_Confirmed" in record:
        value = record["Study_Confirmed"]
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "是")
        return bool(value)
    study = str(record.get("Study_ID", "")).strip()
    return study.startswith(("DOI:", "BIB:", "TITLE:", "RAW_BIB_", "10."))


def _compatible(first, second):
    return all(not a or not b or a == b for a, b in zip(first, second))


def maximal_signatures(patterns):
    """Union declarations within maximal compatible cliques, never transitively.

    Unique observed patterns are compressed before exact clique enumeration.
    A blank declaration can occur in several cliques but cannot connect two
    conflicting known declarations into one signature.
    """
    observed = sorted(set(patterns))
    if not observed:
        return []
    adjacency = {i: {j for j in range(len(observed))
                     if i != j and _compatible(observed[i], observed[j])}
                 for i in range(len(observed))}
    signatures = set()

    def visit(included, potential, seen):
        if not potential and not seen:
            merged = []
            for field in range(len(CONDITION_FIELDS)):
                values = {observed[node][field] for node in included if observed[node][field]}
                if len(values) > 1:
                    raise AssertionError("Conflicting condition declarations entered a QC series")
                merged.append(next(iter(values), ""))
            signatures.add(tuple(merged))
            return
        pivot = max(potential | seen, key=lambda node: len(potential & adjacency[node])) if potential | seen else None
        for node in sorted(potential - (adjacency[pivot] if pivot is not None else set())):
            visit(included | {node}, potential & adjacency[node], seen & adjacency[node])
            potential.remove(node)
            seen.add(node)

    visit(set(), set(adjacency), set())
    return sorted(signatures)


def _assign_condition_series(records):
    base_groups = defaultdict(list)
    for row in records:
        base = (str(row.get("Study_ID", "")), str(row.get("Solute_Identity", "")),
                str(row.get("Solvent_Audit_ID", "")))
        base_groups[base].append(row)
    groups, assignments = defaultdict(list), {}
    for base, rows in base_groups.items():
        # An explicitly unresolved effective assignment contributes no metadata
        # evidence. It also cannot be rescued through cross-temperature inference.
        signatures = maximal_signatures(_conditions(row) for row in rows
                                        if not _true(row.get("Effective_Conditions_Ambiguous", False)))
        for row in rows:
            rid = row["canonical_record_id"]
            reported_effective = _conditions(row)
            compatible = [signature for signature in signatures if _compatible(reported_effective, signature)]
            explicit_ambiguity = _true(row.get("Effective_Conditions_Ambiguous", False))
            unique = len(compatible) == 1 and not explicit_ambiguity
            series_conditions = compatible[0] if unique else reported_effective
            inferred = [CONDITION_FIELDS[index] for index, value in enumerate(reported_effective)
                        if not value and series_conditions[index]] if unique else []
            inference_evidence = {
                field: sorted(peer["canonical_record_id"] for peer in rows
                              if not _true(peer.get("Effective_Conditions_Ambiguous", False))
                              and _compatible(_conditions(peer), series_conditions)
                              and _conditions(peer)[index] == series_conditions[index])
                for index, field in enumerate(CONDITION_FIELDS) if field in inferred
            }
            assignment = ("EXACT_OBSERVED_SIGNATURE" if not inferred else "UNIQUE_COMPATIBLE_CROSS_T_SIGNATURE") if unique else "AMBIGUOUS_OR_UNRESOLVED"
            reason = "EFFECTIVE_CONDITIONS_UNRESOLVED" if explicit_ambiguity else (
                "MISSING_CONDITION_WITH_MULTIPLE_COMPATIBLE_SIGNATURES" if len(compatible) > 1 else
                "NO_CONFIRMED_COMPATIBLE_CONDITION_SIGNATURE") if not unique else ""
            assignments[rid] = {
                "QC_Series_Conditions": list(series_conditions),
                "QC_Condition_Assignment": assignment,
                "QC_Inferred_Condition_Fields": inferred,
                "QC_Condition_Evidence_Record_IDs": inference_evidence,
                "QC_Compatible_Signature_Count": len(compatible),
                "QC_Compatible_Signatures": [list(signature) for signature in compatible],
                "QC_Condition_Assignment_Basis": "Observed effective declarations from the same original study, full solute identity and audited solvent identity across all temperatures; no source metadata modified",
                "QC_Condition_Unresolved_Reason": reason,
            }
            # Unresolved declarations never join a fitted group, even if their
            # literal missing-value pattern happens to equal a known signature.
            groups[base + series_conditions + ("ASSIGNED" if unique else "UNRESOLVED",)].append(row)
    return groups, assignments


def fit(records):
    xy = [(1.0 / float(row["T_K"]), math.log(float(row["gamma_inf"]))) for row in records]
    slopes = [(y2-y1)/(x2-x1) for index, (x1,y1) in enumerate(xy)
              for x2,y2 in xy[index+1:] if x2 != x1]
    if not slopes:
        return None
    slope = median(slopes)
    intercept = median(y - slope*x for x,y in xy)
    return intercept, slope


def predict(model, temperature):
    return model[0] + model[1]/float(temperature)


def loo(records):
    diagnostics = {}
    for row in records:
        rid = row["canonical_record_id"]
        model = fit([peer for peer in records if peer["canonical_record_id"] != rid])
        if model is None:
            continue
        prediction = predict(model, row["T_K"])
        residual = math.log(float(row["gamma_inf"])) - prediction
        diagnostics[rid] = {"A": model[0], "B": model[1], "ln_pred": prediction,
                            "residual": residual, "F": _factor(residual)}
    return diagnostics


def _isolated(row, records):
    rid = row["canonical_record_id"]
    peers = [peer for peer in records if peer["canonical_record_id"] != rid]
    model = fit(peers)
    if model is None:
        return None
    temperature = float(row["T_K"])
    peer_temperatures = [float(peer["T_K"]) for peer in peers]
    span = max(peer_temperatures)-min(peer_temperatures)
    outside = max(min(peer_temperatures)-temperature, temperature-max(peer_temperatures), 0)
    if span <= 0 or outside/span > POLICY["maximum_endpoint_extrapolation_fraction"]:
        return None
    prediction = predict(model, temperature)
    factor = _factor(math.log(float(row["gamma_inf"]))-prediction)
    evidence = {"F": factor, "A": model[0], "B": model[1], "ln_pred": prediction,
                "residual": math.log(float(row["gamma_inf"]))-prediction,
                "extrapolation": outside > 0, "extrapolation_fraction": outside/span,
                "peer_max_F": None, "peer_spread_F": None, "stability_F": None}
    if len(records) == 3:
        peer_values = [math.log(float(peer["gamma_inf"])) for peer in peers]
        spread = max(peer_values)-min(peer_values)
        if factor < POLICY["three_temperature_extreme_factor"] or spread > math.log(POLICY["three_temperature_peer_gamma_spread_factor"]):
            return None
        return dict(evidence, mode="HIGH_SUSPICION_3T", peer_spread_F=math.exp(spread))
    if factor < POLICY["severe_factor"]:
        return None
    peer_diagnostics = loo(peers)
    peer_max = max(item["F"] for item in peer_diagnostics.values())
    if peer_max > POLICY["peer_loo_max_factor"]:
        return None
    peer_predictions = [predict(fit([other for other in peers if other["canonical_record_id"] != peer["canonical_record_id"]]), temperature) for peer in peers]
    stability = _factor(max(peer_predictions)-min(peer_predictions))
    if stability > POLICY["reference_prediction_stability_factor"]:
        return None
    return dict(evidence, mode="ISOLATED_TEMP_INCONSISTENCY", peer_max_F=peer_max, stability_F=stability)


def screen(records):
    """Return results/removals/series/stats without mutating any input record.

    EXCLUDE_QC is modelling eligibility only. REVIEW and NOT_EVALUATED remain
    retained by temperature QC; separate structural/source eligibility applies.
    """
    records = list(records)
    ids = [str(row["canonical_record_id"]) for row in records]
    if any(not rid for rid in ids) or len(ids) != len(set(ids)):
        raise ValueError("canonical_record_id must be nonempty and unique")
    groups, assignments = _assign_condition_series(records)
    results, removals, series = {}, {}, []
    for key, rows in sorted(groups.items()):
        base, conditions = key[:3], key[3:8]
        sid = _stable("TS2_", key)
        valid = all(_valid_numeric(row) for row in rows)
        sorted_rows = sorted(rows, key=lambda row: (float(row["T_K"]) if _valid_numeric(row) else 0.0, str(row["canonical_record_id"])))
        n_temperatures = len({_numeric_key(row["T_K"]) for row in rows}) if valid else 0
        reason = ""
        if not valid:
            reason = "NONPOSITIVE_OR_NONFINITE_T_OR_GAMMA"
        elif not all(_study_known(row) for row in rows):
            reason = "ORIGINAL_STUDY_UNRESOLVED"
        elif not base[1] or not base[2]:
            reason = "MOLECULAR_IDENTITY_UNRESOLVED"
        elif key[8] == "UNRESOLVED":
            reason = assignments[rows[0]["canonical_record_id"]]["QC_Condition_Unresolved_Reason"]
        elif n_temperatures < POLICY["minimum_temperatures"]:
            reason = "FEWER_THAN_THREE_DISTINCT_TEMPERATURES"
        elif n_temperatures != len(rows):
            reason = "REPEATED_TEMPERATURE_MEASUREMENT_AMBIGUITY"
        eligible = not reason
        initial = loo(sorted_rows) if eligible else {}
        candidates = []
        if eligible:
            for row in sorted_rows:
                if initial[row["canonical_record_id"]]["F"] >= POLICY["severe_factor"]:
                    evidence = _isolated(row, sorted_rows)
                    if evidence is not None:
                        candidates.append((row, evidence))
        # No iterative deletions; multiple candidates mean review, not exclusion.
        accepted = candidates if len(candidates) == 1 else []
        removed = {row["canonical_record_id"]: evidence for row,evidence in accepted}
        peers = [row for row in sorted_rows if row["canonical_record_id"] not in removed]
        post = loo(peers) if removed and len(peers) >= 3 else initial if not removed else {}
        for row in sorted_rows:
            rid = row["canonical_record_id"]
            evidence = removed.get(rid)
            diagnostic = evidence if evidence is not None else post.get(rid)
            status = "NOT_EVALUATED"
            record_reason = (assignments[rid]["QC_Condition_Unresolved_Reason"]
                             if key[8] == "UNRESOLVED" and reason in
                             ("EFFECTIVE_CONDITIONS_UNRESOLVED", "MISSING_CONDITION_WITH_MULTIPLE_COMPATIBLE_SIGNATURES",
                              "NO_CONFIRMED_COMPATIBLE_CONDITION_SIGNATURE") else reason)
            if evidence:
                status = "EXCLUDE_QC"
                record_reason = evidence["mode"]
                removals[rid] = {"status": status, "reason": record_reason, "series_id": sid, "evidence": evidence}
            elif diagnostic:
                status = "PASS" if diagnostic["F"] < POLICY["review_factor"] else "REVIEW"
                record_reason = "WITHIN_THREEFOLD_TREND_REFERENCE" if status == "PASS" else "TREND_INCONSISTENT_BUT_ISOLATION_NOT_ESTABLISHED"
            elif removed:
                record_reason = "ONLY_TWO_REFERENCE_POINTS_AFTER_3T_EXCLUSION"
            results[rid] = {"canonical_record_id": rid, "series_id": sid, "status": status,
                            "reason": record_reason, "n_points": len(rows), "n_temperatures": n_temperatures,
                            "initial": initial.get(rid), "reference": diagnostic,
                            "exclusion_mode": evidence["mode"] if evidence else "",
                            "source_value_changed": False, **assignments[rid]}
        series.append({"series_id": sid, "key": list(key[:8]), "QC_Series_Conditions": list(conditions),
                       "QC_Condition_Group_Assigned": key[8] == "ASSIGNED", "record_ids": [row["canonical_record_id"] for row in sorted_rows],
                       "eligible": eligible, "not_evaluated_reason": reason,
                       "n_temperatures": n_temperatures, "excluded_ids": list(removed),
                       "initial_max_F": max((item["F"] for item in initial.values()), default=None),
                       "candidate_exclusions": len(candidates),
                       "status_counts": dict(Counter(results[row["canonical_record_id"]]["status"] for row in rows))})
    stats = {"version": VERSION, "policy": POLICY, "input_points": len(records), "series_count": len(series),
             "eligible_series": sum(item["eligible"] for item in series),
             "eligible_points": sum(len(item["record_ids"]) for item in series if item["eligible"]),
             "status_counts": dict(Counter(item["status"] for item in results.values())),
             "initial_F_ge_3": sum(item["initial"] is not None and item["initial"]["F"] >= 3 for item in results.values()),
             "initial_F_ge_10": sum(item["initial"] is not None and item["initial"]["F"] >= 10 for item in results.values()),
             "exclusion_modes": dict(Counter(item["exclusion_mode"] for item in results.values() if item["exclusion_mode"])),
             "confirmed_source_errors": 0, "source_values_changed": 0,
             "condition_assignment_counts": dict(Counter(item["QC_Condition_Assignment"] for item in results.values())),
             "not_evaluated_reasons": dict(Counter(item["reason"] for item in results.values() if item["status"] == "NOT_EVALUATED"))}
    return {"results": results, "removals": removals, "series": series, "stats": stats}


def _true(value):
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "是")
    return bool(value)


def _valid_numeric(row):
    try:
        return all(math.isfinite(float(row[key])) and float(row[key]) > 0 for key in ("T_K", "gamma_inf"))
    except (TypeError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    audit = screen(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"temperature_qc_results.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output/"temperature_qc_summary.json").write_text(json.dumps(audit["stats"], ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ("canonical_record_id", "series_id", "status", "reason", "n_points", "n_temperatures", "exclusion_mode", "F_initial", "F_reference",
               "QC_Series_Conditions", "QC_Condition_Assignment", "QC_Inferred_Condition_Fields", "QC_Condition_Evidence_Record_IDs",
               "QC_Compatible_Signature_Count", "QC_Compatible_Signatures", "QC_Condition_Assignment_Basis")
    with (args.output/"temperature_qc_records.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for item in audit["results"].values():
            row = {key: item[key] for key in columns if key in item}
            row = {key: json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value for key,value in row.items()}
            row.update(F_initial=(item["initial"] or {}).get("F", ""), F_reference=(item["reference"] or {}).get("F", ""))
            writer.writerow(row)
    print(json.dumps(audit["stats"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
