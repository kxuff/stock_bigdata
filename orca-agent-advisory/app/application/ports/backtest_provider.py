from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class BacktestRequest:
    symbols: list[str]
    start_date: str | None
    end_date: str | None
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestProviderResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    trades_summary: dict[str, Any] = field(default_factory=dict)
    equity_curve_sampled: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BacktestProvider(Protocol):
    def is_available(self) -> bool: ...

    def run_backtest(self, request: BacktestRequest) -> BacktestProviderResult: ...
