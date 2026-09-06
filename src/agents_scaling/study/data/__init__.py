"""Data adapters, salted-hash splits and the public/protected exports (WP1).

Public consumers (generate, judge_best, cells) import only :mod:`.public`;
:mod:`.protected` may be imported by ``agents_scaling.study.evaluation`` only (§3.2, §10.6).
"""
