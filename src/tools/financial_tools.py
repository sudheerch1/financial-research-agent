import boto3
import json
import yfinance as yf
from src.agent.state import AgentState
import os
from tavily import TavilyClient
from urllib.parse import urlparse
import re

# Domains excluded from financial research sourcing — social platforms and
# user-generated content sites are not appropriate sources for material
# financial claims.
BLACKLISTED_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "quora.com",
    "threads.net",
}

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

def _is_blacklisted(url: str) -> bool:
    """Checks if a URL's domain (or its parent domain) is blacklisted."""
    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace("www.", "")
        return any(domain == bd or domain.endswith(f".{bd}") for bd in BLACKLISTED_DOMAINS)
    except Exception:
        return False

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
            "52w_high": float(round(hist["High"].max(), 2)),
            "52w_low": float(round(hist["Low"].min(), 2)),
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
            "data_note": "revenue_growth reflects the most recent trailing period available from Yahoo Finance — not necessarily the same fiscal year(s) referenced in the research question. Do not directly compare this figure to a specific historical fiscal year's growth rate without confirming the time periods overlap.",
        }
    except Exception as e:
        return {"tool": "get_financial_ratios", "ticker": ticker, "error": str(e)}


def get_earnings_history(ticker: str) -> dict:
    """Gets the most recent quarterly earnings (Net Income) available from yfinance.
    Note: yfinance only exposes the last ~4-5 quarters — NOT historical quarters
    matching an arbitrary requested fiscal year range."""
    try:
        stock = yf.Ticker(ticker)
        income = stock.quarterly_income_stmt
        if income is not None and not income.empty:
            if "Net Income" in income.index:
                series = income.loc["Net Income"].head(4)
                series.index = series.index.astype(str)
                net_income = series.to_dict()
            else:
                df = income.head(4)
                df.columns = df.columns.astype(str)
                net_income = df.to_dict()
        else:
            net_income = {}
        return {
            "tool": "get_earnings_history",
            "ticker": ticker,
            "quarterly_earnings": net_income,
            "data_note": "Most recent quarters available via yfinance — may not align with a historical period requested in the research question.",
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


def _extract_years(text: str) -> list[str]:
    """Extract 4-digit year-like tokens (20xx) from text."""
    return re.findall(r"\b20\d{2}\b", text)


def search_web(query: str, in_scope_years: list[str] | None = None) -> dict:
    """Searches for macro/news context using live web search (Tavily).

    in_scope_years: optional list of fiscal year strings (e.g. ["2022","2023","2024"]).
    When provided, results whose content is dominated by out-of-scope years are
    filtered out before being returned.
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            max_results=5,
            search_depth="basic",
            include_answer=True,
        )

        raw_results = response.get("results", [])
        filtered_results = [r for r in raw_results if not _is_blacklisted(r["url"])]
        any_blacklisted = len(filtered_results) < len(raw_results)

        any_out_of_scope = False
        if in_scope_years:
            scope_set = set(in_scope_years)
            scoped_results = []
            for r in filtered_results:
                years_found = _extract_years(r.get("content", "") + " " + r.get("title", ""))
                out_of_scope_count = sum(1 for y in years_found if y not in scope_set)
                in_scope_count = sum(1 for y in years_found if y in scope_set)
                if years_found and out_of_scope_count > in_scope_count:
                    any_out_of_scope = True
                    continue
                scoped_results.append(r)
            filtered_results = scoped_results

        synthesized = "" if (any_blacklisted or any_out_of_scope) else response.get("answer", "")

        sources = "\n".join(
            f"- {r['title']} ({r['url']}): {r['content'][:200]}"
            for r in filtered_results
        )
        result_text = f"{synthesized}\n\nSources:\n{sources}" if synthesized else sources

        return {
            "tool": "search_web",
            "query": query,
            "result": result_text,
            "data_note": "Live web search via Tavily. Reflects current/recent web content, not historical filings — may not align with the fiscal period under discussion.",
        }
    except Exception as e:
        return {
            "tool": "search_web",
            "query": query,
            "result": f"[ERROR] Web search failed: {e}",
            "data_note": "Web search unavailable for this call.",
        }

        
def check_data_completeness(state: AgentState) -> dict:
    """Cross-checks data consistency across all gathered sources."""
    issues = []

    rag = state.get("rag_results", "")
    tool_results = state.get("tool_results", [])

    if not rag or "[STUB]" in rag:
        issues.append("RAG results are stubbed — no SEC filing data available")

    if not tool_results:
        issues.append("No tool results gathered yet")

    return {
        "tool": "check_data_completeness",
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
    "check_data_completeness": check_data_completeness,
}