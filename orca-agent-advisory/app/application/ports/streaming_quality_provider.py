from typing import Any, Protocol


class StreamingQualityProvider(Protocol):
    def find_quality_incidents(self, symbols: list[str], lookback_minutes: int, limit: int) -> list[dict[str, Any]]: ...

    def compare_streaming_to_batch_features(self, symbols: list[str], as_of_date: str | None) -> list[dict[str, Any]]: ...
