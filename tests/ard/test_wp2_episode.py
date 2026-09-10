"""ARD36 WP2: the episode loop against the fake vLLM server (spec §7, §12.3, §19).

Every claim here is asserted on the **recorded request bodies** — what the server actually
received — not on the plan objects: the plan is tested in ``test_wp2_protocols.py``, and a
plan that is right while the prompt is wrong is exactly the failure this file exists to
catch.
"""

from __future__ import annotations

import json

from agents_scaling.ard import contract as K
from agents_scaling.ard import protocols as P
from agents_scaling.ard import vote as V
from agents_scaling.ard.episode import PHASE_LEADER, PHASE_WORKER, fresh_bank
from tests.ard import wp2_support as W

NO_PACKETS = "(no messages this round)"


def update_bodies(harness, episode) -> dict[tuple[int, int], str]:
    """Recorded prompt text per ``(round, agent slot)``, matched by exact content."""
    recorded = set(harness.prompts())
    out: dict[tuple[int, int], str] = {}
    for decision in episode.decisions:
        content = "\n".join(str(m["content"]) for m in decision.messages)
        assert content in recorded, f"round {decision.round_} slot {decision.agent_slot} was never sent"
        out[(decision.round_, decision.agent_slot)] = content
    return out


def packet_block(prompt: str) -> str:
    return prompt.split("Messages from other agents:", 1)[1].strip()


# --------------------------------------------------------------------------- PRIVATE


def test_private_sends_an_empty_packet_in_every_round(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.PRIVATE)
        bodies = update_bodies(h, episode)
    assert len(bodies) == K.N_AGENTS * K.UPDATE_ROUNDS
    for (round_, slot), prompt in bodies.items():
        assert packet_block(prompt) == NO_PACKETS
        assert '"from":"agent-' not in prompt
        assert "Task: " in prompt and "Your original report:" in prompt


# --------------------------------------------------------------------------- PEER


def test_peer_reads_only_the_previous_round(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.PEER)
        bodies = update_bodies(h, episode)
    for result in episode.rounds:
        previous = (
            episode.root_state if result.round_ == 1 else episode.rounds[result.round_ - 2].state
        )
        current_evidence = {
            d.report.evidence for d in result.decisions if d.report.evidence
        }
        for decision in result.decisions:
            block = packet_block(bodies[(decision.round_, decision.agent_slot)])
            assert len(block.splitlines()) == K.N_AGENTS - 1
            for sender in range(K.N_AGENTS):
                if sender == decision.agent_slot:
                    continue
                report = previous.latest(sender).report
                if report.evidence:
                    assert report.evidence in block          # the round t-1 report is there
            own = previous.latest(decision.agent_slot).report
            if own.evidence:
                assert block.count(own.evidence) == 0        # never its own packet
            for evidence in current_evidence:
                # a round-t sibling's *current* report can never appear in a round-t prompt
                if evidence not in {previous.latest(s).report.evidence for s in range(K.N_AGENTS)}:
                    assert evidence not in block


def test_peer_packet_bytes_carry_no_confidence(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.PEER)
        bodies = update_bodies(h, episode)
    for prompt in bodies.values():
        for line in packet_block(prompt).splitlines():
            body = json.loads(line)
            assert "confidence" not in body
            assert set(body) == {"from", "final_answer", "evidence", "truncated"}


# --------------------------------------------------------------------------- no early stop


def test_a_unanimous_team_still_spends_every_decision(tmp_path) -> None:
    """Spec §7: no early stopping on unanimity; unanimous error is not absorbing."""
    with W.harness(tmp_path, answer_pool=("42",)) as h:
        episode = h.run(K.Protocol.PEER)
        n_prompts = len(h.prompts())
    for result in episode.rounds:
        assert result.class_counts == {"42": K.N_AGENTS}      # unanimous from round 0 on
    assert len(episode.decisions) == K.N_AGENTS * K.UPDATE_ROUNDS
    assert n_prompts == K.N_AGENTS + K.N_AGENTS * K.UPDATE_ROUNDS   # roots + 3 x 5


# --------------------------------------------------------------------------- coverage


def test_archive_coverage_never_falls(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.PEER)
    coverage = [frozenset(episode.root_state.archive_classes())]
    coverage += [r.archive_classes for r in episode.rounds]
    for earlier, later in zip(coverage, coverage[1:]):
        assert earlier <= later
        assert len(earlier) <= len(later)


def test_transition_counts_reconcile_with_the_correct_count(tmp_path) -> None:
    """Plan §5.8: ``C_{t+1} - C_t == #wrong->correct - #correct->wrong``, invalidity included."""
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.PEER)
    correct_class = "42"

    def classes(state):
        return [state.canonical_class(state.latest(s).report) for s in range(K.N_AGENTS)]

    states = [episode.root_state] + [r.state for r in episode.rounds]
    for prev, nxt in zip(states, states[1:]):
        before, after = classes(prev), classes(nxt)
        c_before = sum(1 for c in before if c == correct_class)
        c_after = sum(1 for c in after if c == correct_class)
        w2c = sum(1 for b, a in zip(before, after) if b != correct_class and a == correct_class)
        c2w = sum(1 for b, a in zip(before, after) if b == correct_class and a != correct_class)
        assert c_after - c_before == w2c - c2w
        # invalid reports are states too: every slot lands in exactly one transition class
        assert len(before) == len(after) == K.N_AGENTS


# --------------------------------------------------------------------------- BALANCED


def test_balanced_hides_multiplicity_in_the_bytes(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.BALANCED)
        bodies = update_bodies(h, episode)
    for result in episode.rounds:
        previous = (
            episode.root_state if result.round_ == 1 else episode.rounds[result.round_ - 2].state
        )
        for decision in result.decisions:
            block = packet_block(bodies[(decision.round_, decision.agent_slot)])
            lines = block.splitlines()
            peers = [s for s in range(K.N_AGENTS) if s != decision.agent_slot]
            classes = [previous.canonical_class(previous.latest(s).report) for s in peers]
            keyed = {c for c in classes if c}
            unkeyed = [c for c in classes if not c]
            # one packet per canonical class, plus ONE typed entry for every unparsed peer
            expected = len(keyed) + (1 if unkeyed else 0)
            assert len(lines) == expected
            assert len(lines) <= len(peers)
            assert result.collapsed_unparsed[decision.agent_slot] == len(unkeyed)
            for line in lines:
                body = json.loads(line)
                assert set(body) in ({"status"}, {"from", "final_answer", "evidence", "truncated"})


# --------------------------------------------------------------------------- HUB5


def test_hub5_is_one_leader_then_four_workers(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.HUB5)
        prompts = h.prompts()
        bodies = update_bodies(h, episode)
    leader = episode.leader
    assert leader is not None
    for result in episode.rounds:
        phases = [(d.agent_slot, d.phase) for d in result.decisions]
        assert phases[0] == (leader, PHASE_LEADER)
        assert {slot for slot, _ in phases[1:]} == set(range(K.N_AGENTS)) - {leader}
        assert all(phase == PHASE_WORKER for _, phase in phases[1:])
        # the leader's request physically precedes every worker's request
        leader_at = prompts.index(bodies[(result.round_, leader)])
        for slot in range(K.N_AGENTS):
            if slot != leader:
                assert prompts.index(bodies[(result.round_, slot)]) > leader_at
        # and every worker reads exactly the current leader report
        leader_report = result.state.latest(leader).report
        leader_block = packet_block(bodies[(result.round_, leader)])
        assert len(leader_block.splitlines()) == K.N_AGENTS - 1   # the leader reads four
        for slot in range(K.N_AGENTS):
            if slot == leader:
                continue
            block = packet_block(bodies[(result.round_, slot)])
            assert len(block.splitlines()) == 1
            if leader_report.evidence:
                assert leader_report.evidence in block


# --------------------------------------------------------------------------- GATE


def test_gate_composition_on_the_recorded_bodies(tmp_path) -> None:
    """Spec §12.3: one empty packet; the other four identical to the paired PEER episode.

    Byte-identity is asserted at round 1, where the two arms still share the root state —
    after that the protected agent's answer legitimately diverges, which is the
    intervention.  The arbitrary-round claim is proved on the plan in
    ``test_wp2_protocols.py::test_gate_packets_are_byte_identical_to_the_paired_peer_episode``.
    """
    protected = 2
    with W.harness(tmp_path) as h:
        peer = h.run(K.Protocol.PEER)
        peer_bodies = update_bodies(h, peer)
        gate = h.run(K.Protocol.GATE, choose_protected=lambda state, round_: protected)
        gate_bodies = update_bodies(h, gate)

    for result in gate.rounds:
        empty = [
            slot
            for slot in range(K.N_AGENTS)
            if packet_block(gate_bodies[(result.round_, slot)]) == NO_PACKETS
        ]
        assert empty == [protected]
        assert result.protected_slot == protected
    for slot in range(K.N_AGENTS):
        if slot == protected:
            continue
        assert packet_block(gate_bodies[(1, slot)]) == packet_block(peer_bodies[(1, slot)])
        # the protected agent still transmits its round-0 report to everyone else
        assert peer.root_state.latest(protected).report.evidence in packet_block(gate_bodies[(1, slot)])
    # the protected agent's own prompt is the PRIVATE construction, byte for byte
    agent = gate.root_state.agents[protected]
    assert gate_bodies[(1, protected)] == P.render_update_prompt(
        gate.spec.task_text, agent.root.report, agent.latest.report, ()
    )[0]["content"]


def test_random_protection_protects_exactly_one_agent_per_round(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.RANDOM_PROTECT)
        bodies = update_bodies(h, episode)
    for result in episode.rounds:
        empty = [
            slot
            for slot in range(K.N_AGENTS)
            if packet_block(bodies[(result.round_, slot)]) == NO_PACKETS
        ]
        assert empty == [result.protected_slot]


# --------------------------------------------------------------------------- FRESH


def test_fresh_is_task_only_and_accumulates_twenty_independent_draws(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.FRESH)
        bodies = update_bodies(h, episode)
    root_prompt = P.render_root_prompt(episode.spec.task_text)[0]["content"]
    for prompt in bodies.values():
        assert prompt == root_prompt          # no previous answers, no peer messages (§7)
    ids = {ref.decision_id for ref in episode.roots}
    ids |= {r.state.latest(s).decision_id for r in episode.rounds for s in range(K.N_AGENTS)}
    assert len(ids) == K.N_AGENTS * (K.UPDATE_ROUNDS + 1) == 20

    bank = fresh_bank(episode)
    assert len(bank) == 20
    votes = V.fresh_prefix_votes(bank, source_id=episode.spec.source_id, study_seed=W.STUDY_SEED)
    assert sorted(votes) == sorted(K.FRESH_PREFIX_K)
    assert votes[5].pool_candidate_ids == tuple(c.candidate_id for c in bank[:5])


def test_fresh_draw_ordinals_follow_the_frozen_bank_layout(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.FRESH)
    for round_ in range(1, K.UPDATE_ROUNDS + 1):
        for slot in range(K.N_AGENTS):
            key = P.update_seed_key(
                source_id=episode.spec.source_id,
                split=episode.spec.split,
                model_cell=episode.spec.model_cell,
                protocol=K.Protocol.FRESH,
                agent_slot=slot,
                round_=round_,
            )
            assert key.namespace == K.NS_STATELESS_BANK
            assert key.purpose == K.PURPOSE_ROOT
            assert key.step_slot == K.fresh_draw_index(round_, slot)
            assert key.step_slot >= K.FRESH_DRAW_BASE      # disjoint from the auxiliary bank


# --------------------------------------------------------------------------- identity


def test_roots_alias_across_protocols(tmp_path) -> None:
    """Plan §5.1: one root record per (item, draw), shared by every protocol."""
    with W.harness(tmp_path) as h:
        private = h.run(K.Protocol.PRIVATE)
        after_first = h.server.chat_count
        peer = h.run(K.Protocol.PEER)
        after_second = h.server.chat_count
    assert [r.decision_id for r in private.roots] == [r.decision_id for r in peer.roots]
    # the second episode paid for its fifteen updates only; the five roots were aliased
    assert after_second - after_first == K.N_AGENTS * K.UPDATE_ROUNDS
    # the two arms are genuinely different request sets after round 0
    assert {d.decision_id for d in private.decisions}.isdisjoint(
        {d.decision_id for d in peer.decisions}
    )


def test_states_at_t0_are_shared_but_diverge_after_the_intervention(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        private = h.run(K.Protocol.PRIVATE)
        peer = h.run(K.Protocol.PEER)
    assert private.root_state.state_ids() == peer.root_state.state_ids()
    assert private.rounds[0].state.state_ids() != peer.rounds[0].state.state_ids()


def test_a_short_horizon_is_still_a_complete_episode(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.PRIVATE, rounds=1)
    assert len(episode.decisions) == K.N_AGENTS
    assert len(episode.rounds) == 1


# --------------------------------------------------------------------------- invalidity


def _mixed_report_mode(prompt: str) -> str:
    """Deterministic pure function of the prompt: every third distinct prompt is unparseable."""
    import hashlib

    return "invalid" if int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % 3 == 0 else "valid"


def test_balanced_collapses_unparsed_peers_into_one_entry(tmp_path) -> None:
    with W.harness(tmp_path, report_mode=_mixed_report_mode) as h:
        episode = h.run(K.Protocol.BALANCED)
        bodies = update_bodies(h, episode)
    seen_bucket = False
    for result in episode.rounds:
        previous = (
            episode.root_state if result.round_ == 1 else episode.rounds[result.round_ - 2].state
        )
        for decision in result.decisions:
            lines = packet_block(bodies[(decision.round_, decision.agent_slot)]).splitlines()
            buckets = [line for line in lines if json.loads(line).get("status")]
            peers = [s for s in range(K.N_AGENTS) if s != decision.agent_slot]
            unkeyed = [s for s in peers if not previous.canonical_class(previous.latest(s).report)]
            assert len(buckets) == (1 if unkeyed else 0)     # never two, never per-peer
            assert result.collapsed_unparsed[decision.agent_slot] == len(unkeyed)
            if unkeyed:
                seen_bucket = True
                assert buckets[0] == '{"status":"unparsed_peer"}'
                assert str(len(unkeyed)) not in buckets[0]   # the count is not in the bytes
    assert seen_bucket, "the fixture never produced an unparsed peer"


def test_an_all_invalid_team_fails_selection_without_stopping_the_episode(tmp_path) -> None:
    with W.harness(tmp_path, report_mode="invalid") as h:
        episode = h.run(K.Protocol.PEER)
    assert len(episode.decisions) == K.N_AGENTS * K.UPDATE_ROUNDS
    assert episode.root_selections["root"].selection_failed is True
    for result in episode.rounds:
        assert result.class_counts == {}
        assert result.archive_classes == frozenset()
        current = result.selections["current"]
        assert current.selection_failed is True
        assert current.planned_count == K.N_AGENTS and current.valid_count == 0
        assert V.plurality_expected_correctness(current, {}) == 0.0
