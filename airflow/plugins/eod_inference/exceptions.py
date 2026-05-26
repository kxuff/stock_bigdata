from __future__ import annotations


class PipelineValidationError(ValueError):
    pass


class NoNewEodData(RuntimeError):
    pass
