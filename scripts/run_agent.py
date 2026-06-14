import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import agent_graph


def run(question: str):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")

    config = {"configurable": {"thread_id": "test-run-1"}}

    for step in agent_graph.stream(
        {"question": question},
        config=config,
        stream_mode="updates"
    ):
        for node_name, output in step.items():
            print(f"✅ {node_name}")
            if not output:                    # add this line
                continue                      # add this line            
            if "research_goals" in output:
                print(f"   Goals: {output['research_goals']}")
            if "confidence_score" in output:
                print(f"   Confidence: {output['confidence_score']}")
            if "goals_met" in output:
                print(f"   Goals met: {output['goals_met']}")
            if "conflict_detected" in output:
                print(f"   Conflict: {output['conflict_detected']}")
            if "final_report" in output:
                print(f"\n{'='*60}")
                print("FINAL REPORT:")
                print(f"{'='*60}")
                print(output["final_report"])


if __name__ == "__main__":
    run("What is Apple's revenue trend over the last 3 years?")