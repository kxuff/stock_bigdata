from typing import Any, Protocol


class StreamingAlertProvider(Protocol):
    def get_latest_alerts(self, symbols: list[str], severities: list[str], limit: int, lookback_minutes: int) -> list[dict[str, Any]]: ...

    def get_active_symbol_alerts(self, symbol: str, lookback_minutes: int) -> list[dict[str, Any]]: ...
