from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command
from checkpoint_test import graph

config = {"configurable": {"thread_id": "checkpoint-test-1"}}
for step in graph.stream(Command(resume="hello"), config=config, stream_mode="updates"):
    print(step)