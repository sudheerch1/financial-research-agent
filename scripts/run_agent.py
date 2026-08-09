import sys
import os
import uuid
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "financial-research-agent"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.types import Command
from src.agent.graph import agent_graph


def run(question: str):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}\n")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph_input = {"question": question}

    while True:
        interrupted = False

        for step in agent_graph.stream(
            graph_input,
            config=config,
            stream_mode="updates"
        ):
            if "__interrupt__" in step:
                payload = step["__interrupt__"][0].value
                print(f"\n⚠️  HUMAN REVIEW REQUIRED")
                print(f"   Conflict: {payload['conflict_details']}")
                print(f"   Options: {payload['options']}")

                decision = input("   Your decision (sec_primary/balanced/disputed): ").strip()
                graph_input = Command(resume=decision)
                interrupted = True
                break  # exit inner loop, restart stream() in resume mode

            for node_name, output in step.items():
                print(f"✅ {node_name}")
                if not output:
                    continue
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

        if not interrupted:
            break


if __name__ == "__main__":
    #run("What is the status of Apple's partnership with OpenAI?")
    #run("What is Apple's revenue trend over the last 3 years?")
    #run("What's the latest news on Apple's AI strategy?")
    
    #run("What is Microsoft's revenue trend over the last 3 years?")
    # run("What is the status of Apple's partnership with OpenAI?")
    # run("What was Apple's fiscal year 2025 revenue, and how does its current stock price compare to its 52-week high?")
    run("What was Apple's total net sales for fiscal year 2025, and how does Apple's gross margin compare to Microsoft and Amazon?")
 
    # run("What was Apple's total net sales for fiscal year 2025?")