"""Operator-facing contracts for the authoritative schema-5 v1.2 runbook."""

from __future__ import annotations

from pathlib import Path


RUNBOOK = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "SCHEMA5_V12_RECOVERY_RUNBOOK.md"
)
RELEASE_GUIDE = RUNBOOK.with_name("SCHEMA5_RELEASE.md")
RETIRED_R1_CHAIN = RUNBOOK.with_name("SCHEMA5_RECOVERY_CHAIN.md")


def test_local_operator_blocks_begin_in_one_fail_fast_bash() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    fixed = text[text.index("## Fixed identities") :]
    first_block = fixed[fixed.index("```bash") + len("```bash") :]
    first_block = first_block[: first_block.index("```")]

    assert "same dedicated Bash process" in text
    assert "do not paste a later mutation or submission block" in text
    assert first_block.lstrip().startswith("set -euo pipefail")
    assert first_block.index("set -euo pipefail") < first_block.index("repo=")
    assert "export PATH=/usr/bin:/bin" in first_block
    assert "GIT_NO_REPLACE_OBJECTS=1" in first_block
    assert "SBATCH_*|SACCT_*|SCONTROL_*|SQUEUE_*" in first_block
    assert first_block.index("export PATH=/usr/bin:/bin") < first_block.index(
        'dev_python="$(realpath'
    )
    assert "for-each-ref --format='%(refname)' refs/replace" in text


def test_recovery_chain_documents_transactional_failfast_stage_observers() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "publishes all 43 jobs" in text
    assert "Chain schema 11" in text
    assert "one fail-fast `afterany` observer" in text
    assert "STAGE_SCHEDULER_EVIDENCE.json" in text
    assert "STAGE_SENTINEL_COMPLETE.json" in text
    assert (
        "recovery_chain_stage_sentinels/schema5-v1.2-r4/gNNNN/<stage>/"
        in text
    )
    assert "explicitly has no repair authority" in text
    assert "aggregate sentinel remains the sole causal" in text
    assert "without waiting for the\nlong-running email-acknowledgement branch" in text


def test_recovery_chain_documents_prelaunch_bootstrap_resurrection_gate() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    section = text[
        text.index("## Render and submit the superseding chain")
        : text.index("Each stage observer authenticates")
    ]
    assert 'status == "awaiting_bootstrap_watchdog"' in section
    assert "BOOTSTRAP_GENERATION_PROVENANCE.json" in section
    assert "bootstrap-bundle" in section
    assert "prepare-isolated-bootstrap-drill" in section
    assert "--isolated-drill-chain-manifest" in section
    assert "--isolated-drill-submission-receipt" in section
    assert "authorized_keys.bootstrap.line" in section
    assert "bootstrap-deployment-evidence" in section
    assert "bootstrap-drill-evidence" in section
    assert "bootstrap-attestation" in section
    assert "schema5-bootstrap-watchdog drill-status" in section
    assert "schema5-bootstrap-watchdog drill-repair" in section
    assert 'scancel -- "${isolated_job_ids[@]}"' in section
    assert "--installed-release-root $bootstrap_vm_release" in section
    assert "sudo systemctl enable --now $bootstrap_timer" in section
    assert ": \"${BOOTSTRAP_VM_LOGIN:?" in section
    assert ": \"${BOOTSTRAP_CLUSTER_HOST:?" in section
    assert ": \"${BOOTSTRAP_CLUSTER_KNOWN_HOSTS:?" in section
    assert ": \"${PRODUCTION_WATCHDOG_PUBLIC_KEY:?" in section
    assert "READY`, `ARM_INTENT`, and\n`ARMED`" in section
    assert section.index("submit_result=") < section.index(
        "prepare-isolated-bootstrap-drill"
    ) < section.index(
        "bootstrap-bundle"
    ) < section.index(
        "bootstrap-deployment-evidence"
    ) < section.index(
        "bootstrap-drill-evidence"
    ) < section.index(
        "bootstrap-attestation"
    ) < section.index(
        '"$sealed_python" -I "$renderer" release-root'
    )
    assert "at most 1,200\nseconds" in section
    assert "descending from the validated immutable initial\n`LAUNCHED`" in section
    assert "no direct scientific admission" in section
    assert "does not transfer\nbootstrap authority" in section
    assert "aggregate `failure_sentinel`" in text
    assert "other 42 jobs" in text
    assert "RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_HANDOFF_COMPLETE.json" in text
    assert "schema5-v1.2-r4-bootstrap-watchdog-handoff-v1" in text
    assert "shared\nsubmission/repair lock" in text
    assert "Only after that sealed\nmarker exists" in text


def test_partial_materialization_is_quarantined_before_suffix_repair() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    section = text[
        text.index("If the sentinel classifies `release_materialize`")
        : text.index("## Email acknowledgement and production")
    ]

    quarantine = '"$sealed_python" -I "$renderer" quarantine-materialization'
    repair = '"$sealed_python" -I "$renderer" repair-chain'
    occurrences = [
        index
        for index in range(len(section))
        if section.startswith(quarantine, index)
    ]

    assert len(occurrences) == 3
    assert occurrences[-1] < section.index(repair)
    assert section.count("--chain-manifest \"$chain_manifest\" --apply") == 3
    assert "already_quarantined_and_sealed" in section
    assert "The first command is non-mutating." in section


def test_email_acknowledgement_uses_the_generation_scoped_v2_challenge() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    section = text[
        text.index("## Email acknowledgement and production")
        : text.index("## Controlled capacity transition")
    ]

    assert 'email_challenges/CURRENT.json' in section
    assert 'request="$(jq -er \'.request\' "$active_challenge")"' in section
    assert 'output="$(jq -er \'.acknowledgement\' "$active_challenge")"' in section
    assert 'ack_tool="$(jq -er \'.acknowledgement_tool.path\' "$request")"' in section
    assert '--request "$recovery/readiness/email_ack_request.json"' not in section
    assert (
        '--output "$recovery/readiness/email_acknowledgement.json"'
        not in section
    )
    assert section.index("--token \"$email_token\"") < section.index(
        "--token \"$email_token\" --apply"
    )


def test_materialization_pilot_uses_only_the_exact_tagged_script() -> None:
    text = RELEASE_GUIDE.read_text(encoding="utf-8")
    section = text[
        text.index("## Required two-prefix pilot")
        : text.index("## Materialize")
    ]

    assert (
        'schema5_pilot_script="$schema5_checkout/scripts/'
        'run_schema5_materialization_pilot.py"' in section
    )
    assert "python scripts/run_schema5_materialization_pilot.py" not in section
    assert section.count(
        '"$schema5_dev_python" -I "$schema5_pilot_script"'
    ) >= 6
    assert '.submit_command == ["sbatch", $path]' in section
    assert 'sbatch "$schema5_pilot_sbatch"' not in section
    assert (
        '"$schema5_dev_python" -I "$schema5_pilot_script" submit-sbatch'
        in section
    )
    assert '--pilot-root "$schema5_pilot"' in section
    assert '--sbatch-receipt "$schema5_pilot_receipt"' in section
    assert "--scheduler-user mabdel03" in section
    assert "--visibility-timeout 900" in section
    assert '"${schema5_pilot_submit_cmd[@]}" --apply' in section
    assert (
        "jq -er '.job_id | select(type == \"string\" and "
        'test("^[0-9]+$"))\'' in section
    )
    assert 'squeue -h -j "$schema5_pilot_job_id" -o "%i"' in section
    assert 'sacct -X -n -P -j "$schema5_pilot_job_id"' in section
    assert "--format=JobIDRaw,State,ExitCode" in section
    assert (
        '"$schema5_pilot_job_id|COMPLETED|0:0"' in section
    )
    assert "transactional scheduler submission boundary" in section
    assert "manual scheduler boundary" not in section
    assert "schema5_pilot_sacct_deadline=$((SECONDS + 900))" in section
    assert "SECONDS > schema5_pilot_sacct_deadline" in section
    assert 'case "${#schema5_pilot_sacct_rows[@]}" in' in section
    assert "An empty result is the only retryable state." in section
    assert (
        "expected exactly one top-level pilot sacct row, got "
        '${#schema5_pilot_sacct_rows[@]}' in section
    )
    assert (
        "unexpected top-level pilot terminal accounting: "
        "$schema5_pilot_terminal" in section
    )
    assert (
        "timed out waiting for top-level pilot sacct accounting"
        in section
    )
    assert section.index('squeue -h -j "$schema5_pilot_job_id"') < (
        section.index("schema5_pilot_sacct_deadline=$((SECONDS + 900))")
    )
    assert section.index(
        'case "${#schema5_pilot_sacct_rows[@]}" in'
    ) < (
        section.index(
            '"$schema5_dev_python" -I "$schema5_pilot_script" '
            "accept-scheduler"
        )
    )
    assert section.count(
        '"$schema5_dev_python" -I "$schema5_pilot_script" accept-scheduler'
    ) == 3
    assert "PILOT_SCHEDULER_ACCEPTED.json" in section
    assert "accepted` and `already_accepted`" in section
    assert "Do not add `--apply` to that interactive command." in section
    assert "raw\nstdout/stderr" in section
    assert "reparses\nthose raw records" in section
    assert section.count("--ownership-policy") == 2
    assert section.count("--integrity-normalization-policy") == 2
    assert section.count(
        '--durable-git-release-marker "$schema5_recovery/'
        'DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R4_COMPLETE.json"'
    ) == 2
    assert "`already_submitted`" in section
    assert "reports `adopted`" in section
    assert "fail closed\nwithout a second submission" in section
    assert (
        '--prior-quarantine-seal "$schema5_pilot_quarantine_seal"'
        in section
    )
    assert "retaining all earlier\n`--prior-quarantine-seal` arguments" in section
    assert "ignores only those exact terminal\naccounting rows" in section
    assert "unsealed historical job remains fatal" in section
    assert "PILOT_SUBMISSION_ACCEPTED.json" in section
    assert "publishes acceptance without resubmitting" in section
    assert "deliberately ineligible to suppress any scheduler" in section


def test_chain_requires_scheduler_attested_materialization_pilot() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    section = text[
        text.index("## Required two-prefix materialization pilot")
        : text.index("## Render and submit the superseding chain")
    ]

    assert "PILOT_COMPLETE.json" in section
    assert "PILOT_SCHEDULER_ACCEPTED.json" in section
    assert "`accept-scheduler`" in section
    assert "both pilot markers verify" in section
    assert "Interactive\n`run --apply`" in section
    assert "schema-5 pilot verifier report" in section
    assert "both independent policy hashes" in section
    assert "environment_ownership_policy.v1.json" in text
    assert "environment_integrity_normalization_policy.v1.json" in text
    assert 'sbatch "$pilot_sbatch"' not in section
    assert '"$dev_python" -I "$pilot_script" submit-sbatch' in section
    assert '--pilot-root "$pilot_root"' in section
    assert '--sbatch-receipt "$pilot_receipt"' in section
    assert "--scheduler-user mabdel03" in section
    assert "--visibility-timeout 900" in section
    assert '"${pilot_submit_cmd[@]}" --apply' in section
    assert (
        "jq -er '.job_id | select(type == \"string\" and "
        'test("^[0-9]+$"))\'' in section
    )
    assert 'squeue -h -j "$pilot_job_id" -o "%i"' in section
    assert (
        'sacct -X -n -P -j "$pilot_job_id" '
        "--format=JobIDRaw,State,ExitCode" in section
    )
    assert '"$pilot_job_id|COMPLETED|0:0"' in section
    assert "pilot_sacct_deadline=$((SECONDS + 900))" in section
    assert "SECONDS > pilot_sacct_deadline" in section
    assert 'case "${#pilot_sacct_rows[@]}" in' in section
    assert (
        "test -z \"$pilot_sacct_row\" || "
        'pilot_sacct_rows+=("$pilot_sacct_row")' in section
    )
    assert (
        "expected exactly one top-level pilot sacct row, got "
        '${#pilot_sacct_rows[@]}' in section
    )
    assert (
        "unexpected top-level pilot terminal accounting: $pilot_terminal"
        in section
    )
    assert (
        "timed out waiting for top-level pilot sacct accounting"
        in section
    )
    assert "An empty result alone is retried" in section
    assert "absence through the 900-second\ndeadline fails" in section
    assert section.index('squeue -h -j "$pilot_job_id"') < section.index(
        "pilot_sacct_deadline=$((SECONDS + 900))"
    )
    assert section.index('case "${#pilot_sacct_rows[@]}" in') < section.index(
        '"$dev_python" -I "$pilot_script" accept-scheduler'
    )
    assert "`already_submitted`" in section
    assert "returning `adopted`" in section
    assert "all fail closed without resubmission" in section
    assert '--prior-quarantine-seal "$pilot_quarantine_seal"' in section
    assert "Accumulate every earlier seal on later retries." in section
    assert "Any unsealed, mismatched, or newly duplicated job remains" in section
    assert "must adopt that exact terminal job without a second `sbatch`" in section


def test_materialization_pilot_transient_quarantine_is_documented() -> None:
    text = RELEASE_GUIDE.read_text(encoding="utf-8")
    section = text[
        text.index("## Required two-prefix pilot")
        : text.index("## Materialize")
    ]

    assert section.count(
        '"$schema5_dev_python" -I "$schema5_pilot_script" quarantine'
    ) == 3
    assert "already_quarantined_and_sealed" in section
    assert "FAILED" in section
    assert "OUT_OF_MEMORY" in section
    assert "requires a superseding release" in section
    assert (
        '--prior-quarantine-seal "$schema5_pilot_quarantine_seal"'
        in section
    )
    assert "Never discover these markers with a glob" in section


def test_chain_renderer_receives_canonical_python_paths() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert (
        'dev_python="$(realpath -e '
        '/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python)"'
        in text
    )
    render_section = text[
        text.index("## Render and submit the superseding chain")
        : text.index("## Email acknowledgement and production")
    ]
    assert 'sealed_python="$(realpath -e "$sealed_python")"' in render_section
    assert '--dev-python "$dev_python"' in render_section
    assert '--partition mit_normal --slurm-user "$slurm_user"' in render_section
    assert "RECOVERY_CHAIN_SCHEMA5_V1_2_R4_SUBMISSION.json" in render_section
    assert "(.jobs | length) == 43" in render_section
    assert '" Requeue=0 "' in render_section
    assert '" Comment=$comment "' in render_section


def test_production_documents_the_full_fail_closed_admission_ramp() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    production = text[
        text.index("## Email acknowledgement and production")
        : text.index("## Controlled capacity transition")
    ]
    transition = text[text.index("## Controlled capacity transition") :]

    assert "`24 -> 96 -> 192 -> 384`" in production
    assert "| 24 | 96 | one hour" in production
    assert "| 96 | 192 | six hours" in production
    assert "| 192 | 384 | twelve hours" in production
    assert "eight hours" in production
    assert "thirteen hours" in production
    assert "nineteen hours" in production
    assert "full-capacity throughput qualification" in production
    assert "There is no\ngeneration-1 96-cell fallback" in production
    assert "safe ceiling of 96" not in production
    assert "768 cells and 15,360 QIDs" in production
    assert "at least 201,994 trusted QIDs/day" in production
    assert "scientific work is never\nrouted to preemptible capacity" in production
    assert "attest-client-capacity" in transition
    assert "no-requeue client canary" in transition
    assert "Do not hand-author any of its evidence files or invoke `sbatch`" in transition
    assert "build-client-capacity dry-run" in transition
    assert "build-client-capacity prepare" in transition
    assert "build-client-capacity submit" in transition
    assert "build-client-capacity apply" in transition
    assert 'client_canary_submission="$(sbatch --parsable' not in transition
    assert 'sbatch --parsable "$client_canary_sbatch"' not in transition
    assert "--canary-job-id" not in transition
    assert "marker-first submission intent and attempt" in transition
    assert "complete `squeue` plus `sacct` truth" in transition
    assert "accepted-submission marker last" in transition
    assert "source-verified absence observations" in transition
    assert "recursively binds their sealed retry authorization" in transition
    assert "candidate-bearing observation permanently forbids" in transition
    assert "derives the completed canary ID only from" in transition
    assert "four rehashed evidence identities" in transition
    assert "submission_pending) sleep 60" in transition
    assert "adopted|already_submitted) break" in transition
    assert "publishes\n`CLIENT_CAPACITY_COMPLETE.json` last" in transition
    assert "post-transition readiness" in transition


def test_launch_requires_protected_capacity_watchdog_and_qualification() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    prerequisites = text[
        text.index(
            "## Protected capacity before render; external watchdog at stage 19"
        )
        : text.index("## Render and submit the superseding chain")
    ]
    render = text[
        text.index("## Render and submit the superseding chain")
        : text.index("## Email acknowledgement and production")
    ]

    assert "PROTECTED_CAPACITY_COMPLETE.json" in prerequisites
    assert "WATCHDOG_READY.json" in prerequisites
    assert "EXTERNAL_WATCHDOG_KILL_DRILL_COMPLETE.json" in prerequisites
    assert "PreemptMode=OFF" in prerequisites
    assert "448 jobs" in prerequisites
    assert "1,572,864 MiB" in prerequisites
    assert "forced-command-only" in prerequisites
    assert "exact 300-second timer" in prerequisites
    assert "at least 60 seconds apart" in prerequisites
    assert "`throughput_qualification` stage 18" in render
    assert "THROUGHPUT_QUALIFICATION_COMPLETE.json" in render
    assert "does not wait for or adopt a pre-existing qualification marker" in render
    assert "`controller_drill` stage 19" in render
    assert "`production_resume` names both stages 18 and\n"
    "19 as exact `afterok` dependencies" in render
    assert "Watchdog\ndrill/readiness markers are\nstage-19 outputs" in render


def test_protected_capacity_uses_marker_last_effective_fleet_materializer() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    section = text[
        text.index(
            "## Protected capacity before render; external watchdog at stage 19"
        )
        : text.index(
            "After the submitted chain has completed `schema5_initialize`"
        )
    ]

    assert "materialize_schema5_effective_fleet.py" in section
    assert (
        "schema5-v1.2-r4-effective-fleet-materialization-v1"
        in section
    )
    assert "EFFECTIVE_FLEET_COMPLETE.json" in section
    assert "schema5_fleet.effective.v1.json" in section
    assert section.count(
        '"$sealed_python" -I "$effective_fleet_tool" materialize'
    ) == 2
    assert '"${effective_fleet_args[@]}" --apply' in section
    assert "--format runbook-nul" in section
    assert "test \"${#effective_fleet_inputs[@]}\" -eq 10" in section
    assert "Never construct\nthe effective contract with `jq`" in section
    assert "zero\ndelta to every serving profile before qualification" in section
    assert "base = effective = 22/24 and zero\nadditive replicas/GPUs" in section
    assert "278/384 selected cells, a 106-cell\n# shortfall" in section
    assert section.count(
        "--effective-fleet-contract \"$effective_fleet_contract\""
    ) == 3
    assert section.count(
        "--effective-fleet-contract-sha256 "
        '"$effective_fleet_contract_sha256"'
    ) == 2
    assert section.count(
        "--additive-overlay-contract \"$additive_overlay_contract\""
    ) == 3
    assert section.count("--capacity-generation 1") == 3
    assert section.count(
        "--static-feasibility-certificate \"$static_capacity_certificate\""
    ) == 2
    assert section.count(
        "--source-tree-sha256 \"$source_tree_sha256\""
    ) == 2
    assert (
        "--dispatcher-source-sha256 \"$dispatcher_source_sha256\""
        in section
    )
    assert "--qualification-runner-source-sha256" in section
    assert '"$qualification_runner_source_sha256"' in section
    publisher_verify = section[
        section.index(
            '"$sealed_python" -I "$protected_publisher" verify'
        )
        : section.index(
            'test -f "$protected_capacity"'
        )
    ]
    assert publisher_verify.count(
        "--qualification-runner-source-sha256"
    ) == 1
    assert (
        "Review exactly 448 canary job elements: 384 clients,\n"
        "# 22 active servers, three warm-turnover allocations, and 39 held "
        "reserve elements."
    ) in section
    assert (
        "`PROTECTED_CAPACITY_COMPLETE.json` is a schema-4"
        in section
    )


def test_durable_release_and_watchdog_commands_are_exact_and_fail_closed() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    freeze = text[
        text.index("## Freeze and test the r4 source")
        : text.index("## Required two-prefix materialization pilot")
    ]
    watchdog = text[
        text.index("After the submitted chain has completed `schema5_initialize`")
        : text.index("## Render and submit the superseding chain")
    ]

    assert 'git -C "$repo" push --atomic "$durable_remote"' in freeze
    assert "publish_schema5_durable_git_release.py" in freeze
    assert freeze.count("--remote-commit-ref \"$durable_commit_ref\"") == 2
    assert "--durable-git-release-marker \"$durable_marker\"" in freeze
    assert "build_schema5_watchdog_deployment.py" in watchdog
    assert "deployment-evidence" in watchdog
    assert "acknowledge-liveness" in watchdog
    assert "--confirm-email-received" in watchdog
    assert "liveness-evidence" in watchdog
    assert "--installed-release-root $watchdog_vm_release" in watchdog
    assert "--service-heartbeat" in watchdog
    assert "Result=success" in watchdog
    assert "ExecMainStatus=0" in watchdog
    assert "Do not manually publish `WATCHDOG_READY.json`" in watchdog


def test_r4_uses_fresh_operational_paths_and_binds_sealed_r2_r3_lineage() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    release = RELEASE_GUIDE.read_text(encoding="utf-8")

    assert "# Schema-5 v1.2-r4 recovery and production runbook" in text
    assert "tag=sweep-recovery-schema5-v1.2-r4" in text
    assert "chain namespace is `schema5-v1.2-r4`" in text
    assert "durable_commit_ref=refs/heads/schema5-v1.2-r4" in text
    assert (
        "schema5-v1.2-r4-client-placement-capacity-generation-v1"
        in text
    )
    assert (
        "schema5-v1.2-r2-client-placement-capacity-generation-v1"
        not in text
    )
    assert (
        'durable_marker="$recovery/'
        'DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R4_COMPLETE.json"' in text
    )
    for path in (
        "materialization_pilot_source_checkout_v1_2_r4",
        "materialization_pilots/schema5-v1.2-r4",
        "slurm_canaries/schema5-v1.2-r4",
        "jobs/schema5-v1.2-r4-materialization-pilot.sbatch",
        "logs/materialization-pilot-r4",
        "protected_capacity/schema5-v1.2-r4",
        "recovery_chain_stage_sentinels/schema5-v1.2-r4",
    ):
        assert path in text

    lineage = text[
        text.index("## Immutable r2 and r3 to r4 lineage")
        : text.index("## Before-tag gates")
    ]
    assert "refs/heads/schema5-v1.2-r2" in lineage
    assert "DURABLE_GIT_RELEASE_COMPLETE.json" in lineage
    assert "requires_superseding_release" in lineage
    assert "Never rename the partial tree" in lineage
    assert "canary_failures/schema5-v1.2-r2/CANARY_FAILURE_SEALED.json" in lineage
    assert "schema5-v1.2-r2-partial-canary-failure-seal-v1" in lineage
    assert "`retry_in_place=false`" in lineage
    assert "exact seal path, raw SHA-256, size, and\n`seal_id`" in lineage
    assert "reverify\nthat binding at submission" in lineage
    assert "Operational artifact basenames are r4-specific" in lineage
    assert "chain_namespace=schema5-v1.2-r4" in lineage
    assert "sweep-recovery-schema5-v1.2-r3" in lineage
    assert "acd723ba9a99d88e77f7d752268bc31205c3a808" in lineage
    assert "fa87b283974afc1fa48fcb22b61e6ae7eec4bb36" in lineage
    assert "prelaunch_failures/schema5-v1.2-r3" in lineage
    assert "schema5-v1.2-r3-prelaunch-failure-seal-binding-v1" in lineage
    assert "verify-prelaunch-failure" in lineage

    assert "## Superseding operational lineage" in release
    assert "sweep-recovery-schema5-v1.2-r4" in release
    assert "canary_failures/schema5-v1.2-r2/CANARY_FAILURE_SEALED.json" in release
    assert "jobs/schema5-v1.2-r4-materialization-pilot.sbatch" in release
    assert "logs/materialization-pilot-r4" in release


def test_r3_failure_producer_independently_verifies_before_toolchain_use() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    section = text[
        text.index("Produce the two r3 prelaunch-failure envelopes")
        : text.index("Run the genuine isolated schema-4 composite Slurm canary")
    ]

    assert (
        section.count(
            '"$dev_python" -I -B "$r4_sealer" record-prelaunch-attempt'
        )
        == 2
    )
    assert '"$dev_python" -I -B "$r4_sealer" seal-prelaunch-failure' in section
    assert '"$dev_python" -I -B "$r4_sealer" \\\n' in section
    assert '"$r4_probe_python" -I -B "$r3_pilot"' in section
    assert '"$r4_probe_python" -I -B "$r4_sealer"' not in section
    assert "imports\nthe tagged r4 provisioner in-process" in section


def test_r2_canary_failure_sealing_commands_are_exact_and_idempotent() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    lineage = text[
        text.index("## Immutable r2 and r3 to r4 lineage")
        : text.index("## Before-tag gates")
    ]

    assert (
        'r2_release_checkout="$recovery/'
        'materialization_pilot_source_checkout_v1_2_r2"' in text
    )
    assert (
        'r2_zero_mutation="$recovery/r1_acceptance/'
        'ZERO_RESULT_MUTATION_RECEIPT.json"' in text
    )
    assert (
        'r2_snapshot_marker="$recovery/pre_repair/SNAPSHOT_COMPLETE.json"'
        in text
    )
    assert '"$dev_python" -I "$r2_sealer" seal-canary-failure' in lineage
    assert '--tree "$r2_canary_tree"' in lineage
    assert '--evidence-root "$r2_canary_failure_root"' in lineage
    assert '--release-checkout "$r2_release_checkout"' in lineage
    assert "--error-classification deterministic_missing_subprocess_capture" in lineage
    assert lineage.count('--mutation-evidence "$r2_zero_mutation"') == 1
    assert lineage.count('--mutation-evidence "$r2_snapshot_marker"') == 1
    assert lineage.count('"${r2_canary_seal_cmd[@]}"') == 3
    assert lineage.count('"${r2_canary_seal_cmd[@]}" --apply') == 2
    assert "transaction canary job 18889366 was cancelled before start" in lineage
    assert ".known_scheduler_identity.job_ids == [\"18889366\"]" in lineage
    assert "`already_sealed`" in lineage


def test_r4_docs_use_sealed_toolchain_cache_and_current_schema_contracts() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    release = RELEASE_GUIDE.read_text(encoding="utf-8")
    toolchain = RUNBOOK.with_name("SCHEMA5_R4_CONDA_TOOLCHAIN.md").read_text(
        encoding="utf-8"
    )
    historical_r3_probe = (
        'conda-runtime-identity --conda-executable "$recorded_shared_conda"'
    )
    assert runbook.count(historical_r3_probe) == 1
    active_runbook = runbook.replace(historical_r3_probe, "")
    active = "\n".join((active_runbook, release, toolchain))

    assert "--conda-executable" not in active
    assert active.count("--conda-toolchain-root") >= 5
    assert active.count("--source-package-cache") >= 5
    assert "--expected-package-cache-seed-input-json" in release
    assert "source_package_cache=/orcd/home/002/mabdel03/.conda/pkgs" in runbook
    assert (
        'conda_provisioner="$pilot_checkout/scripts/'
        'provision_schema5_conda_toolchain.py"' in runbook
    )
    assert "Pilot schema 5" in active
    assert "materialization schema 5" in active
    assert "Chain schema 11" in runbook
    assert "prerequisite-evidence schema 7" in runbook
    assert "verify-prelaunch-failure" in runbook
    assert "r3_prelaunch_failure_root" in runbook
    assert 'install -d -m 0755 -- "$conda_toolchain_namespace"' in runbook
    assert 'install -d -m 0755 -- "$namespace"' in toolchain
    assert '$repo/scripts/provision_schema5_conda_toolchain.py' not in toolchain
    assert "arbitrary Conda executable" in toolchain
    assert (
        "materialization_pilot_source_checkout_v1_2_r4" in toolchain
    )


def test_release_gates_follow_executable_tag_order() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    before = text[
        text.index("## Before-tag gates")
        : text.index("## Freeze and test the r4 source")
    ]
    after = text[
        text.index("## Post-tag and pre-render gates")
        : text.index("## Render and submit the superseding chain")
    ]

    assert text.index("## Before-tag gates") < text.index(
        "## Freeze and test the r4 source"
    )
    assert text.index("## Freeze and test the r4 source") < text.index(
        "## Post-tag and pre-render gates"
    )
    assert text.index("## Post-tag and pre-render gates") < text.index(
        "## Render and submit the superseding chain"
    )
    assert "clean full test suite" in before
    assert "deterministic r2 canary failure seal" in before
    assert "PROTECTED_CAPACITY_COMPLETE.json" not in before
    assert "external watchdog" not in before
    assert "PROTECTED_CAPACITY_COMPLETE.json" in after
    assert "bootstrap-watchdog VM login" in after
    assert "production external-watchdog deployment" in after
    assert "after `schema5_initialize`" in after
    assert "not render prerequisites" in after
    assert "complete `squeue` plus `sacct`" in after
    assert "`WATCHDOG_READY.json` remains a stage-19 output" in after


def test_retired_r1_record_preserves_historical_r2_successor() -> None:
    text = RETIRED_R1_CHAIN.read_text(encoding="utf-8")

    assert "Immutable historical record" in text
    assert "requires the v1.2-r2\n> superseding release" in text
    assert "requires the v1.2-r4" not in text


def test_throughput_runbook_separates_health_soak_from_loaded_saturation() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    qualification = text[
        text.index("The preproduction qualification exercises")
        : text.index("At 48 continuous production hours")
    ]

    assert "`health_soak_384_seconds` window of at least 7,200 seconds" in qualification
    assert "health soak is not represented as loaded steady-state" in qualification
    assert "at least two\nobservations at exactly 384 active/pending clients" in qualification
    assert "positive trusted-QID progress" in qualification
    assert "`loaded_384_seconds`" in qualification
    assert "`loaded_384_useful_qids`" in qualification
    assert "`loaded_384_observation_count`" in qualification
