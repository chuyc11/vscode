"""CLI entry point for the A-J intelligence pipeline.

Usage:
    python main.py "NVIDIA stock earnings"
    python main.py --help
"""

import argparse
import logging
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from dotenv import load_dotenv

from agent_a_retriever import search_real_news
from orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="A-J 终极情报分析引擎")
    parser.add_argument("query", help="分析主题")
    parser.add_argument("-n", "--max-results", type=int, default=10, help="最大检索条数 (默认 10)")
    parser.add_argument("-d", "--doc-dir", default="", help="本地内参文档目录")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    load_dotenv()

    print(f"🔍 正在搜索: {args.query}")
    facts_list = search_real_news(args.query, args.max_results)

    if not facts_list:
        print("❌ 未能获取到任何事实数据")
        sys.exit(1)

    facts = {f.fact_id: f for f in facts_list}
    print(f"✅ 检索到 {len(facts)} 条事实数据\n")

    for i, f in enumerate(facts_list, 1):
        print(f"  [{i}] (相关性:{f.relevance_score:.2f} 可信度:{f.credibility_score:.2f}) {f.content[:100]}...")

    print(f"\n🧠 开始推理分析...")
    state = run_pipeline(args.query, facts, doc_dir=args.doc_dir)

    print(f"\n{'='*60}")
    print(f"状态: {state.status}")
    print(f"决策: {state.final_decision.value if state.final_decision else 'N/A'}")
    print(f"迭代轮次: {state.iteration_count}")
    print(f"{'='*60}")

    if state.briefing:
        print(f"\n📋 情报简报:\n")
        print(state.briefing)

    if state.errors:
        print(f"\n⚠️ 错误:")
        for e in state.errors:
            print(f"  - {e}")

    if state.all_attack_findings:
        print(f"\n⚔️ 攻击发现 ({len(state.all_attack_findings)} 条):")
        for f in state.all_attack_findings:
            print(f"  [{f.severity.value}] {f.attack_type.value}: {f.description}")


if __name__ == "__main__":
    main()
