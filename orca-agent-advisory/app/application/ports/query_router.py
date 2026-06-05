from typing import Protocol

from app.schemas.agent import AgentQueryRequest, RoutedAgentQuery


class QueryRouter(Protocol):
    def route(self, request: AgentQueryRequest) -> RoutedAgentQuery: ...
