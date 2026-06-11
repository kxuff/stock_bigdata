import json
import sys
import types

import pytest

sys.modules.setdefault("yfinance", types.SimpleNamespace(Ticker=lambda *_args, **_kwargs: None))

from eod_inference import agent_context


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"label": "positive", "sentiment_score": 0.7}).encode("utf-8")


def test_score_sentiment_requires_finbert_url(monkeypatch):
    monkeypatch.delenv("FINBERT_API_URL", raising=False)

    with pytest.raises(RuntimeError, match="FINBERT_API_URL is required"):
        agent_context._score_sentiment_with_finbert("Apple shares rise")


def test_score_sentiment_throws_when_finbert_unreachable(monkeypatch):
    monkeypatch.setenv("FINBERT_API_URL", "http://127.0.0.1:9")

    def fail(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(agent_context.urlrequest, "urlopen", fail)

    with pytest.raises(RuntimeError, match="FinBERT API request failed"):
        agent_context._score_sentiment_with_finbert("Apple shares rise")


def test_score_sentiment_uses_finbert_response(monkeypatch):
    monkeypatch.setenv("FINBERT_API_URL", "http://finbert.local")
    monkeypatch.setattr(agent_context.urlrequest, "urlopen", lambda *_args, **_kwargs: _Response())

    result = agent_context._score_sentiment_with_finbert("Apple shares rise")

    assert result == {"label": "positive", "sentiment_score": 0.7}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Apple launches new chips", True),
        ("AAPL shares rise", True),
        ("Broad market rallies", False),
    ],
)
def test_relevance_filter_uses_symbol_and_company_alias(text, expected):
    aliases = {"AAPL", "apple", "apple inc"}

    assert agent_context._is_relevant_news(text, aliases) is expected


def test_merged_news_combines_yfinance_and_finnhub_with_dedupe(monkeypatch):
    monkeypatch.setattr(
        agent_context,
        "_news_items",
        lambda symbol: [
            {
                "title": "Apple launches AI features",
                "url": "https://example.com/apple-ai",
                "providerPublishTime": 1781136000,
                "data_source": "yfinance",
            }
        ],
    )
    monkeypatch.setattr(
        agent_context,
        "_finnhub_news_items",
        lambda symbol, target_date, *, lookback_days: [
            {
                "headline": "Apple launches AI features",
                "url": "https://example.com/apple-ai",
                "datetime": 1781136000,
                "data_source": "finnhub",
            },
            {
                "headline": "AAPL supplier demand improves",
                "url": "https://example.com/aapl-supplier",
                "datetime": 1781049600,
                "data_source": "finnhub",
            },
        ],
    )

    merged = agent_context._merged_news_items("AAPL", agent_context.parse_date("2026-06-10"), lookback_days=14)

    assert len(merged) == 2
    assert {item["data_source"] for item in merged} == {"yfinance", "finnhub"}
    assert {item.get("url") for item in merged} == {
        "https://example.com/apple-ai",
        "https://example.com/aapl-supplier",
    }


def test_sentiment_source_refs_include_available_news_sources():
    refs = agent_context._sentiment_source_refs(
        "AAPL",
        [
            {"source": "yfinance"},
            {"source": "finnhub"},
        ],
    )

    assert refs == [
        "yfinance.news:AAPL",
        "finnhub.company_news:AAPL",
        "finbert:ProsusAI/finbert",
    ]
