"""Public selectors of the study package (WP3; ``seal.py`` is WP5).

* :mod:`agents_scaling.study.selection.normalize` — outcome-blind vote keys (§5.3):
  MC letter, exact-answer normalization frozen on dev, AST-canonical / exact-source code
  identity (amendment S1, E3').
* :mod:`agents_scaling.study.selection.vote` — plurality VOTE with multiplicity and blind
  ties over ``metrics_reference.plurality_vote`` (§4.4, §5.3).
* :mod:`agents_scaling.study.selection.judge_best` — the strict JUDGE_BEST record parser
  and the pointwise max-score selection over ``metrics_reference.judge_best`` (§4.4).

Nothing here imports evaluation code or protected data; every input is a public
candidate field or the study seed.
"""
