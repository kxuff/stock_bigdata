from enum import StrEnum


class Recommendation(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    WATCH = "WATCH"


class PortfolioAction(StrEnum):
    INCREASE_WEIGHT = "INCREASE_WEIGHT"
    DECREASE_WEIGHT = "DECREASE_WEIGHT"
    MAINTAIN_WEIGHT = "MAINTAIN_WEIGHT"
    EXIT = "EXIT"
    CASH_BUFFER = "CASH_BUFFER"


class RiskLabel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SentimentLabel(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"
    UNAVAILABLE = "UNAVAILABLE"


class ValuationLabel(StrEnum):
    UNDERVALUED = "UNDERVALUED"
    FAIRLY_VALUED = "FAIRLY_VALUED"
    OVERVALUED = "OVERVALUED"
    UNKNOWN = "UNKNOWN"


class AgentStatus(StrEnum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class DecisionMode(StrEnum):
    SINGLE_SYMBOL_ADVISORY = "single_symbol_advisory"
    PORTFOLIO_RECOMMENDATION = "portfolio_recommendation"


class ConflictLevel(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ReviewReason(StrEnum):
    HIGH_RISK = "HIGH_RISK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    STALE_DATA = "STALE_DATA"
    MISSING_REQUIRED_TOOL_RESULT = "MISSING_REQUIRED_TOOL_RESULT"
    PORTFOLIO_CONSTRAINT_VIOLATION = "PORTFOLIO_CONSTRAINT_VIOLATION"
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
    COMPLIANCE_REVIEW = "COMPLIANCE_REVIEW"
    DATA_QUALITY = "DATA_QUALITY"


class RiskTolerance(StrEnum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class InvestmentHorizon(StrEnum):
    INTRADAY = "INTRADAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"


class ToolStatus(StrEnum):
    SUCCESS = "SUCCESS"
    UNAVAILABLE = "UNAVAILABLE"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"


class SignalStance(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class FactorWeight(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
