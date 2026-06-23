import sys, os
sys.path.append(os.getcwd())
from src.tools.financial_tools import search_web

# result = search_web("Apple stock news today")
# print(result["result"])

result = search_web("Apple stock reddit discussion sentiment")
print(result["result"])