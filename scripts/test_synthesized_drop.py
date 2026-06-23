import sys, os
sys.path.append(os.getcwd())
from unittest.mock import patch
from src.tools.financial_tools import search_web

# Fake Tavily response with a mix of clean + blacklisted sources
mock_response_with_blacklisted = {
    "answer": "Apple stock is performing well according to multiple sources.",
    "results": [
        {"title": "Reuters: Apple Earnings", "url": "https://www.reuters.com/apple-earnings", "content": "Apple reported strong earnings..."},
        {"title": "Facebook post about AAPL", "url": "https://www.facebook.com/someuser/posts/123", "content": "I think apple stock is going up..."},
    ],
}

# Fake Tavily response with only clean sources
mock_response_clean = {
    "answer": "Apple stock is performing well according to multiple sources.",
    "results": [
        {"title": "Reuters: Apple Earnings", "url": "https://www.reuters.com/apple-earnings", "content": "Apple reported strong earnings..."},
        {"title": "CNBC: Apple Stock", "url": "https://www.cnbc.com/quotes/AAPL", "content": "Apple shares rose today..."},
    ],
}

with patch("src.tools.financial_tools.TavilyClient") as MockClient:
    # Test 1: blacklisted source present -> synthesized should be dropped
    MockClient.return_value.search.return_value = mock_response_with_blacklisted
    result = search_web("test query")
    assert "Apple stock is performing well" not in result["result"], "FAIL: synthesized answer was not dropped despite blacklisted source"
    assert "facebook.com" not in result["result"], "FAIL: blacklisted source leaked into sources list"
    assert "reuters.com" in result["result"], "FAIL: clean source missing from sources list"
    print("PASS: synthesized dropped + blacklisted source excluded when blacklist hit")

    # Test 2: no blacklisted source -> synthesized should be preserved
    MockClient.return_value.search.return_value = mock_response_clean
    result = search_web("test query")
    assert "Apple stock is performing well" in result["result"], "FAIL: synthesized answer was dropped despite clean sources"
    assert "cnbc.com" in result["result"], "FAIL: clean source missing from sources list"
    print("PASS: synthesized preserved when no blacklist hit")