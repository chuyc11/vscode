import argparse
import logging
import sys
from dotenv import load_dotenv

from agent_a_retriever import search_real_news
from orchestrator import run_pipeline
import pprint

def main():
    parser = argparse.ArgumentParser(description="Run intelligence analysis pipeline")
    parser.add_argument("--query", "-q", type=str, default="OpenAI model Nvidia stock", help="Search query for the analysis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    load_dotenv()
    query = args.query

    print(f"Starting live mission with query: {query}")

    facts_list = search_real_news(query, 5)
    if not facts_list:
        print("\033[91m任务终止：未能从互联网获取到任何有效事实数据\033[0m")
        sys.exit(0)
    facts = {f.fact_id: f for f in facts_list}

    print(f"Retrieved {len(facts)} facts:")
    for fact in facts.values():
        print(f"- {fact.content[:100]}...")

    state = run_pipeline(query, facts)

    print("\n=== Final State ===")
    print(f"Status: {state.status}")
    print(f"Final Decision: {state.final_decision}")
    print(f"Briefing Mode: {state.briefing_mode}")

    if state.graph:
        print("\n=== ClaimGraph ===")
        pprint.pprint(state.graph.model_dump())
    else:
        print("No ClaimGraph generated.")

    if state.briefing:
        print("\n=== Briefing ===")
        print(state.briefing)
    else:
        print("No briefing generated.")

    if state.errors:
        print("\n=== Errors ===")
        for i, error in enumerate(state.errors, 1):
            print(f"Error {i}: {error}")
            if hasattr(error, 'errors'):
                print("Pydantic errors:")
                pprint.pprint(error.errors())

if __name__ == '__main__':
    main()
