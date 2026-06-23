import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.financial_tools import _is_blacklisted

test_urls = [
    "https://www.facebook.com/somepost",
    "https://twitter.com/someuser/status/123",
    "https://x.com/someuser",
    "https://www.reddit.com/r/stocks/comments/abc",
    "https://www.reuters.com/business/apple-news",
    "https://www.linkedin.com/posts/ceo-update",
    "https://medium.com/@analyst/apple-deep-dive",
    "https://www.sec.gov/cgi-bin/browse-edgar",
]

for url in test_urls:
    print(f"{_is_blacklisted(url)}  —  {url}")