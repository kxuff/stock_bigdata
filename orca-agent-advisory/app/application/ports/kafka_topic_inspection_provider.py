from typing import Any, Protocol


class KafkaTopicInspectionProvider(Protocol):
    def inspect_topics(self, topics: list[str] | None = None) -> list[dict[str, Any]]: ...
