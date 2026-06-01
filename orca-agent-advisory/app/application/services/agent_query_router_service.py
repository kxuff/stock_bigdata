import re
from dataclasses import dataclass
from typing import Protocol

from app.schemas.agent import AgentQueryRequest, RoutedAgentQuery, SuggestedAction
from app.schemas.enums import AgentRoute


class AgentRoutePlanner(Protocol):
    def plan(self, request: AgentQueryRequest) -> RoutedAgentQuery: ...


@dataclass
class AgentQueryRouterService:
    planner: AgentRoutePlanner | None = None

    def route(self, request: AgentQueryRequest) -> RoutedAgentQuery:
        try:
            if self.planner is not None:
                return self._validate(self.planner.plan(request), request)
        except Exception:  # noqa: BLE001 - router fallback must fail closed.
            pass
        return self._fallback(request)

    def _validate(self, plan: RoutedAgentQuery, request: AgentQueryRequest) -> RoutedAgentQuery:
        symbols = _unique_symbols([*plan.symbols, *_context_symbols(request), *_extract_symbols(request.message)])
        route = plan.route
        message = plan.message
        needs_clarification = plan.needs_clarification

        if len(symbols) > 1 and route == AgentRoute.SINGLE_SYMBOL_ADVISORY:
            route = AgentRoute.SYMBOL_COMPARISON
            message = "Multiple symbols detected; routing comparison workflow."
        elif route == AgentRoute.SINGLE_SYMBOL_ADVISORY and not symbols:
            route = AgentRoute.CLARIFICATION
            needs_clarification = True
            message = "Which stock symbol should ORCA analyze?"
        elif route in {AgentRoute.SYMBOL_COMPARISON, AgentRoute.WATCHLIST_REVIEW} and not symbols:
            route = AgentRoute.CLARIFICATION
            needs_clarification = True
            message = "Which symbols should ORCA compare or review?"
        elif route == AgentRoute.UNIVERSE_SCREEN and not (symbols or request.context.universe):
            message = message or "Screening latest ORCA prediction universe."

        return RoutedAgentQuery(
            route=route,
            confidence=plan.confidence,
            symbols=symbols,
            needs_clarification=needs_clarification or route == AgentRoute.CLARIFICATION,
            message=message,
            suggested_actions=plan.suggested_actions or _suggested_actions(route, symbols),
        )

    def _fallback(self, request: AgentQueryRequest) -> RoutedAgentQuery:
        symbols = _unique_symbols([*_context_symbols(request), *_extract_symbols(request.message)])
        if len(symbols) > 1:
            route = AgentRoute.SYMBOL_COMPARISON
            message = "Routing to comparison for detected symbols."
        elif len(symbols) == 1:
            route = AgentRoute.SINGLE_SYMBOL_ADVISORY
            message = f"Routing to single-symbol advisory for {symbols[0]}."
        else:
            route = AgentRoute.CLARIFICATION
            message = "Ask about one stock symbol, multiple symbols, watchlist, screener, market brief, or data diagnostics."
        return RoutedAgentQuery(
            route=route,
            confidence=0.35 if symbols else 0.0,
            symbols=symbols,
            needs_clarification=route == AgentRoute.CLARIFICATION,
            message=message,
            suggested_actions=_suggested_actions(route, symbols),
        )


def _context_symbols(request: AgentQueryRequest) -> list[str]:
    return [s for s in [request.context.symbol, *request.context.symbols, *request.context.watchlist] if s]


def _extract_symbols(message: str) -> list[str]:
    ignored = {"I", "A", "THE", "AND", "OR", "FOR", "API", "SQL", "HTML", "CSS", "JSON", "CSV"}
    return [symbol for symbol in re.findall(r"(?<![A-Za-z0-9])([A-Z]{1,5}(?:-[A-Z])?)(?![A-Za-z0-9])", message) if symbol not in ignored]


def _normalize_symbol(value: str | None) -> str | None:
    if not value:
        return None
    symbol = value.strip().upper().replace(".", "-")
    return symbol if re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", symbol) else None


def _unique_symbols(values: list[str]) -> list[str]:
    symbols: list[str] = []
    for value in values:
        symbol = _normalize_symbol(value)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _suggested_actions(route: AgentRoute, symbols: list[str]) -> list[SuggestedAction]:
    actions = [SuggestedAction(label="Ask for single-symbol advisory", route=AgentRoute.SINGLE_SYMBOL_ADVISORY)]
    actions.extend(SuggestedAction(label=f"Analyze {symbol}", route=AgentRoute.SINGLE_SYMBOL_ADVISORY, symbol=symbol) for symbol in symbols[:3])
    if route != AgentRoute.SYMBOL_COMPARISON:
        actions.append(SuggestedAction(label="Compare symbols", route=AgentRoute.SYMBOL_COMPARISON, symbols=symbols[:5]))
    return actions
