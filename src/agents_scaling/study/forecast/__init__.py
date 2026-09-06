"""FINAL_HANDOFF_REPORT compiler and the tier-2b shadow-forecast cell (N2; spec §8.7).

Modules
* :mod:`report`   — ``compile_report``: the sealed item + selection → one frozen report text
  with byte spans, the task-only anchor and the 8,192-recipient-token evidence cap.
* :mod:`manifest` — the trusted forecast manifest, the rendered forecast request (single
  user message, thinking off), the request spec/seed and the byte→token anchor mapping for
  the neural capture stage.
* :mod:`shadow`   — strict forecast parsing and the on-disk forecast / report-render files.
* :mod:`run`      — the resumable CPU cell CLI (``python -m agents_scaling.study.forecast.run``).

Nothing here reads ``data/protected``; every input is a sealed artifact or the public export.
"""

from agents_scaling.study.forecast.manifest import forecast_manifest, forecast_spec, render_forecast_request, report_render
from agents_scaling.study.forecast.report import Report, compile_report
from agents_scaling.study.forecast.shadow import Forecast, ParsedForecast, parse_forecast

__all__ = [
    "Forecast",
    "ParsedForecast",
    "Report",
    "compile_report",
    "forecast_manifest",
    "forecast_spec",
    "parse_forecast",
    "render_forecast_request",
    "report_render",
]
