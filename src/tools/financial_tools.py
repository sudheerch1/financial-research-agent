import boto3
import json
import yfinance as yf
from src.agent.state import AgentState

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def get_stock_price(ticker: str) -> dict:
    """Gets current and historical stock price for a ticker symbol."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        current = stock.info.get("currentPrice", "N/A")
        return {
            "tool": "get_stock_price",
            "ticker": ticker,
            "current_price": current,
            "52w_high": round(hist["High"].max(), 2),
            "52w_low": round(hist["Low"].min(), 2),
            "avg_volume": int(hist["Volume"].mean()),
        }
    except Exception as e:
        return {"tool": "get_stock_price", "ticker": ticker, "error": str(e)}


def get_financial_ratios(ticker: str) -> dict:
    """Gets key financial ratios for fundamental analysis."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "tool": "get_financial_ratios",
            "ticker": ticker,
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "pb_ratio": info.get("priceToBook", "N/A"),
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "roe": info.get("returnOnEquity", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "revenue_growth": info.get("revenueGrowth", "N/A"),
        }
    except Exception as e:
        return {"tool": "get_financial_ratios", "ticker": ticker, "error": str(e)}


def get_earnings_history(ticker: str) -> dict:
    """Gets quarterly earnings history and EPS trend."""
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.quarterly_earnings
        if earnings is not None and not earnings.empty:
            recent = earnings.head(4).to_dict()
        else:
            recent = {}
        return {
            "tool": "get_earnings_history",
            "ticker": ticker,
            "quarterly_earnings": recent,
        }
    except Exception as e:
        return {"tool": "get_earnings_history", "ticker": ticker, "error": str(e)}


def get_company_news(ticker: str) -> dict:
    """Gets recent news headlines for a company."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5] if stock.news else []
        headlines = [
            {
                "title": n.get("content", {}).get("title", ""),
                "summary": n.get("content", {}).get("summary", ""),
            }
            for n in news
        ]
        return {
            "tool": "get_company_news",
            "ticker": ticker,
            "headlines": headlines,
        }
    except Exception as e:
        return {"tool": "get_company_news", "ticker": ticker, "error": str(e)}


def get_competitor_comparison(ticker: str) -> dict:
    """Compares key metrics against sector peers."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        market_cap = info.get("marketCap", "N/A")
        return {
            "tool": "get_competitor_comparison",
            "ticker": ticker,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "note": "Full peer comparison requires premium data source",
        }
    except Exception as e:
        return {"tool": "get_competitor_comparison", "ticker": ticker, "error": str(e)}


def search_web(query: str) -> dict:
    """Searches for macro/news context using Bedrock's knowledge."""
    # Stub — replace with Tavily API in Week 2
    return {
        "tool": "search_web",
        "query": query,
        "result": f"[STUB] Web search results for: {query}",
    }


def validate_data_consistency(state: AgentState) -> dict:
    """Cross-checks data consistency across all gathered sources."""
    issues = []

    rag = state.get("rag_results", "")
    tool_results = state.get("tool_results", [])

    if not rag or "[STUB]" in rag:
        issues.append("RAG results are stubbed — no SEC filing data available")

    if not tool_results:
        issues.append("No tool results gathered yet")

    return {
        "tool": "validate_data_consistency",
        "issues_found": len(issues) > 0,
        "issues": issues,
        "sources_checked": ["rag_results", "tool_results"],
    }


# Tool registry — used by the ReAct loop
TOOLS = {
    "get_stock_price": get_stock_price,
    "get_financial_ratios": get_financial_ratios,
    "get_earnings_history": get_earnings_history,
    "get_company_news": get_company_news,
    "get_competitor_comparison": get_competitor_comparison,
    "search_web": search_web,
    "validate_data_consistency": validate_data_consistency,
}