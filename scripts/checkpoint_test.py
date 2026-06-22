from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph_checkpoint_aws import DynamoDBSaver


class TestState(TypedDict):
    value: str


def node_pause(state: TestState) -> dict:
    decision = interrupt({"message": "waiting for input"})
    return {"value": f"resumed with: {decision}"}


builder = StateGraph(TestState)
builder.add_node("node_pause", node_pause)
builder.add_edge(START, "node_pause")
builder.add_edge("node_pause", END)

checkpointer = DynamoDBSaver(
    table_name="financial-research-agent-checkpoints",
    region_name="us-east-1",
    ttl_seconds=86400 * 7,
)
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "checkpoint-test-1"}}
    for step in graph.stream({"value": "start"}, config=config, stream_mode="updates"):
        print(step)
    print("Process ending now — no resume happening in this run.")