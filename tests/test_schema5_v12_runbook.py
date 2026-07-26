"""Operator-facing contracts for the authoritative schema-5 v1.2 runbook."""

from __future__ import annotations

from pathlib import Path


RUNBOOK = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "SCHEMA5_V12_RECOVERY_RUNBOOK.md"
)
RELEASE_GUIDE = RUNBOOK.with_name("SCHEMA5_RELEASE.md")


def test_recovery_chain_documents_transactional_failfast_stage_observers() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "publishes all 43 jobs" in text
    assert "Chain schema 9" in text
    assert "one fail-fast `afterany` observer" in text
    assert "STAGE_SCHEDULER_EVIDENCE.json" in text
    assert "STAGE_SENTINEL_COMPLETE.json" in text
    assert (
        "recovery_chain_stage_sentinels/schema5-v1.2-r2/gNNNN/<stage>/"
        in text
    )
    assert "explicitly has no repair authority" in text
    assert "aggregate sentinel remains the sole causal" in text
    assert "without waiting for the\nlong-running email-acknowledgement branch" in text


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
        'DURABLE_GIT_RELEASE_COMPLETE.json"'
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
    assert "schema-4 verifier report" in section
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
    assert "RECOVERY_CHAIN_SCHEMA5_V1_2_R2_SUBMISSION.json" in render_section
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
    assert "Watchdog drill/readiness markers are\nstage-19 outputs" in render


def test_durable_release_and_watchdog_commands_are_exact_and_fail_closed() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    freeze = text[
        text.index("## Freeze and test the r2 source")
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
