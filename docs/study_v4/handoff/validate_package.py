"""Validate handoff artifacts only; no models, tasks, GPU jobs or network calls."""
from pathlib import Path
import hashlib
import io
import json
import re
import sys
import unittest

import jsonschema
import yaml

BASE = Path(__file__).resolve().parent
REPORT = {"scope": "specification_package_only", "experiments_executed": False,
          "runtime_validated": False, "checks": [], "unresolved_execute_inputs": []}


def check(name, passed, detail=None):
    REPORT["checks"].append({"name": name, "passed": bool(passed), "detail": detail})


def strict_json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key: " + k)
            result[k] = v
        return result
    def reject(value):
        raise ValueError("nonfinite JSON token: " + value)
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject)


def unresolved(value, prefix=""):
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in unresolved(v, prefix + "." + str(k))]
    if isinstance(value, list):
        return [p for k, v in enumerate(value) for p in unresolved(v, prefix + "." + str(k))]
    return []


def main():
    docpath = BASE / "agents_scaling_local_future_state_reversal_experiment_spec_v4_0.md"
    doc = docpath.read_text()
    headings = re.findall(r"^## (\d+)\. ", doc, re.M)
    check("twelve_top_level_sections", headings == [str(i) for i in range(1, 13)], headings)
    check("code_fences_balanced", sum(line.startswith("```") for line in doc.splitlines()) % 2 == 0)
    check("no_internal_web_citation_markers", not re.search(r"turn\d+(?:search|view|fetch)\d+|cite", doc))
    check("standalone_spec_substantial", len(doc.split()) >= 20000, len(doc.split()))
    check("no_runtime_or_result_claim", "no empirical results or completed runtime implementation are claimed" in doc)
    check("all_primary_family_names", all(("**" + x + " —") in doc for x in "AORCGM"))
    check("RLM_depth_control_and_complete_attempt_definition", "RLM_D0" in doc and "RLM_D1" in doc and "complete episodes" in doc)
    configs = {}
    for path in sorted((BASE / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        configs[path.stem] = cfg
        check("parse_yaml_" + path.name, isinstance(cfg, dict))
        REPORT["unresolved_execute_inputs"].extend(path.name + ":" + x for x in unresolved(cfg))
    schemas = {}
    for path in sorted((BASE / "schemas").glob("*.json")):
        s = strict_json(path)
        jsonschema.Draft202012Validator.check_schema(s)
        schemas[path.name] = s
        check("valid_schema_" + path.name, True)
    jsonschema.validate({"approach": "test", "evidence": [], "alternatives_considered": [],
                         "failure_checks": [], "final_answer": "42", "confidence": 0.5},
                        schemas["candidate.schema.json"])
    check("candidate_valid_fixture", True)
    base_candidate = {"approach": "test", "evidence": [], "alternatives_considered": [],
                      "failure_checks": [], "final_answer": "42", "confidence": 0.5}
    for label, update in [("outside_probability", {"confidence": 1.2}), ("unexpected_field", {"gold": "42"})]:
        try:
            jsonschema.validate({**base_candidate, **update}, schemas["candidate.schema.json"])
            check("reject_candidate_" + label, False)
        except jsonschema.ValidationError:
            check("reject_candidate_" + label, True)
    m = configs["experiment_matrix"]
    h = configs["hypotheses"]
    check("six_family_Holm", h["primary_multiplicity"]["families_in_fixed_order"] == ["A", "O", "R", "C", "G", "M"])
    check("N_grid_includes_hub_and_N1", m["modules"]["static_membership"]["N_roles"] == [1, 2, 3, 5, 9]
          and m["modules"]["static_membership"]["hub_included_in_N"])
    check("required_neutral_count_grid", m["modules"]["static_membership"]["root_framing_cell"] == "00")
    check("informed_primary_independent", m["framing"]["main_N5_IND_VOTE_cell"] == "11")
    check("RLM_full_depth_comparison_frontier", m["modules"]["rlm_frontier"]["conditions"] == ["RLM_D1", "RLM_D2"]
          and m["modules"]["rlm_frontier"]["budget_multipliers"] == [1, 2, 4, 8])
    check("G_freezes_before_confirmation", h["families"]["G"]["fit_or_tune_on_confirmation_labels"] is False
          and h["families"]["G"]["primary_bootstrap_refits_predictors"] is False)
    check("HPC_values_explicit_unresolved", configs["hpc_profile"]["required_execute_inputs"]["gpu_model"] is None)
    n = configs["neural_readouts"]
    sec8 = re.search(r"^## 8\. .*?(?=^## 9\. )", doc, re.M | re.S).group(0).strip() + "\n"
    check("neural_config_matches_current_section", hashlib.sha256(sec8.encode()).hexdigest() == n["source_section8_sha256"])
    check("controlled_consumer_count", 48 * 400 + 48 * 200 + 28 * 200 + 3 * 200 + 0.8 * 200 == 35160)
    check("template_cannot_claim_ready_runtime", configs["hpc_profile"]["execution_implemented"] is False)
    for path in sorted((BASE / "prompts").glob("*.json")):
        strict_json(path)
        check("valid_prompt_json_" + path.name, True)
    suite = unittest.defaultTestLoader.discover(str(BASE / "tests"))
    stream = io.StringIO()
    results = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    check("reference_metric_tests", results.wasSuccessful(), {"tests_run": results.testsRun, "log": stream.getvalue()})
    manifest = BASE / "MANIFEST.json"
    if manifest.exists():
        records = strict_json(manifest)["files"]
        mismatches = [r["path"] for r in records if not (BASE / r["path"]).is_file()
                      or hashlib.sha256((BASE / r["path"]).read_bytes()).hexdigest() != r["sha256"]]
        check("manifest_checksums", not mismatches, mismatches)
    REPORT["passed"] = all(x["passed"] for x in REPORT["checks"])
    REPORT["check_count"] = len(REPORT["checks"])
    REPORT["unresolved_count"] = len(REPORT["unresolved_execute_inputs"])
    result = json.dumps(REPORT, indent=2)
    if "--write-report" in sys.argv:
        (BASE / "VALIDATION_REPORT.json").write_text(result + "\n")
    print(result)
    return 0 if REPORT["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
