# Re-review of the fix pass (afe3093..1791e63), 2026-09-24 — verdict NOT CLEAR at 1791e63

Blocker 1 (B3 grace kept tools, parsed tool calls) is FIXED by the lead in b5043a2 (contract §10.28).

## MAJOR (fix before main; 7 before the pilot)
2. B9 regression — core.py:377-379, 231-244; httpd.py:431-442, 465-485. The reply is cached only when mark_undelivered
   fires, asynchronously (keeper task / after the failed write), after budget.settle released the run's FIFO lock; an
   identical retry of a disconnected in-flight call wakes first, misses the cache and mints a second grant. Probe (real
   core, fake upstream 1.5 s, real ProxyServer; client 1 streams then disconnects, identical retry 0.3 s later): 2 THINK
   calls, duplicates_served=0. Fix: detect client EOF promptly while awaiting (a pending reader.read() beside the
   keepalive loop), flag the in-flight call "client gone" keyed (conv_key, prompt_sha); a new identical request that
   finds an in-flight or finished flagged call waits for it and is served its reply; cache synchronously in _served
   before budget.settle when the flag is already set.
3. 402/401 marks the run infrastructure_stop even after it recovers — runner.py:443-464 fed by core.py:809-811.
   _journal_failure sets record.error="infrastructure_stop: …" on any call's 402/401 and nothing clears it; the runner
   reads record.error, not the last event, so a run whose sub-agent hit a transient upstream 402 and then delivered an
   answer is rewritten to transport_failed and reruns (unlimited attempts). Contradicts review item #11 (only the last
   event decides). Fix: decide from the last event, or a core flag cleared by any later served reply. Draining the
   stage is fine; discarding a delivered answer is not.
4. Hitting the ledger ceiling burns every remaining cell's reruns — runner.py:82, 390-395; core.py:481-488. At the $
   ceiling / unresolved-exposure limit every call gets 503 ledger_refused; TRANSPORT_ERROR_CODES makes each run
   transport_failed and run_spec reruns as a2, a3; each remaining cell ends at attempt 3 and _rerun_pending rejects it
   forever. The plan says ledger exposure "pauses admissions". Fix: treat ledger_refused like the 402 infrastructure
   stop (drain, no in-job rerun, no rerun consumed), or check ledger.fits / unresolved exposure before admitting a cell.
5. The drain overruns Slurm's 900 s warning — runner.py:291-311 + httpd.close + the CLI after launch_stage. Worst case
   800 s drain + ≤60 s terminating + 90 s shielded wait + 60 s wait_idle + reconcile > 900 s. The 90 s wait also cancels
   _await_settled (up to 300 s). Cells left `running` become `interrupted` on resume and burn an attempt; open runs never
   closed; in-flight SENT reservations charged max on resume. Fix: drain wait ≈ 600 s, bound the settle wait by the drain
   deadline, close cancelled attempts as killed_by_drain.
6. Calls from a harness that already exited still go upstream — core.py:362-396 (no closed/exited check after the lock);
   runner.py:482-503. Queued sub-agent calls are served after the harness is gone, even after close_run, each granted up
   to C−spent; completion_total / realized_share inflated for timeout rows; late saves rewrite a finalised graded record
   and could overwrite the next attempt's runs/ file. Fix: core.harness_exited(run_key) called by the runner on process
   exit; chat() refuses without minting a grant if set or the run is closed; block store.save after close.
7. Canary Gate 1 reconciliation "mismatch" — reconcile.py:284. (a) Tokens: reconcile sums /generation tokens_completion
   (OpenRouter's NORMALISED count) against journaled native tokens: 4323 journaled vs 2388 metadata, identical cost
   $0.0132775. Use native_tokens_completion. (b) Dollars: key delta $0.1139 vs ledger $0.2531 because /key was read the
   same second the stage finished; the lead confirmed the key caught up to exactly $0.253105 within ~45 s. Re-poll /key
   after a settle delay (poll until two reads ≥ 30 s apart agree, bounded, e.g. 5 min) before declaring a mismatch.

## MINOR
8. upstream.py:429-440 + core.py:481-488: a ledger_refused / LedgerError on transport attempt ≥ 2 (after 429s or a
   charged-max CACHE_HIT) releases the token grant (correct) but the earlier attempts' phases and charged-max $ never
   reach the call journal or record.cost_usd; the PRE_SEND_ERROR_CODES comment is false for attempt ≥ 2. Attach the
   phases to the ProxyError and journal them.
9. core.py:655-660: _terminal does not handle internal_error (LedgerError) like ledger_refused (budget.calls not
   decremented, nothing journaled).
10. httpd.py:431-442: "delivered" = write+drain succeeded, and the first write after the peer's FIN succeeds, so which
   late answers count as delivered is timing-dependent. Check reader EOF before and after the write.
11. plan.py:305-328: the write-once manifest records only the first launch; a filtered first launch pre-registers a
   subset and the later full launch raises. Always write plan_stage's full list; check incoming specs are a subset.
12. tables.py:175: run_key_of still uses the old formula without the §10.22 partition (only synthetic_plan uses it).
13. runner.py:434-436, 505-518: _close_failed skips _await_settled; closes while a call may be in flight.
