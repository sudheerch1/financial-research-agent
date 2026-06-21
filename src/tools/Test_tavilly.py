from dotenv import load_dotenv
load_dotenv()
from src.tools.financial_tools import search_web

result = search_web("Apple Vision Pro recent news")
print(result)