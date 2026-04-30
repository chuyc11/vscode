import streamlit as st
import logging
from dotenv import load_dotenv

from agent_a_retriever import search_real_news
from orchestrator import run_pipeline
from schema import FinalDecision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

st.set_page_config(page_title="A-J 终极情报矩阵控制台", page_icon="🧠", layout="wide")

st.title("🧠 A-J 终极情报矩阵控制台")
st.markdown("**终极情报分析引擎 - Alpha 版** | 由 A-J 节点驱动")

with st.sidebar:
    st.header("控制面板")
    query = st.text_input("输入课题 (query)", "OpenAI model Nvidia stock")
    st.markdown("---")
    start_button = st.button("🚀 启动矩阵", type="primary")

if start_button:
    load_dotenv()

    with st.spinner("🔍 A节点正在搜索并评估情报..."):
        facts_list = search_real_news(query, 5)

    if not facts_list:
        st.error("❌ 任务终止：未能从互联网获取到任何有效事实数据")
        st.stop()

    facts = {f.fact_id: f for f in facts_list}
    st.success(f"✅ 检索到 {len(facts)} 条事实数据（已通过相关性筛选）")

    with st.spinner("🧠 D-E-G 节点正在迭代分析..."):
        state = run_pipeline(query, facts)

    st.success("🎯 推理完成！")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 分析概况")
        st.metric("状态", state.status)
        st.metric("最终决策", str(state.final_decision.value) if state.final_decision else "N/A")
        st.metric("迭代轮次", state.iteration_count)

        if state.briefing:
            st.subheader("📋 情报简报")
            st.markdown(state.briefing)
        else:
            st.warning("⚠️ 无简报生成")

    with col2:
        with st.expander("🔍 事实数据"):
            for fact in facts.values():
                st.write(f"**相关性:** {fact.relevance_score} | **可信度:** {fact.credibility_score}")
                st.write(fact.content[:200])
                if fact.summary:
                    st.caption(f"摘要: {fact.summary}")
                st.divider()

        with st.expander("🔧 ClaimGraph JSON"):
            if state.graph:
                st.json(state.graph.model_dump())
            else:
                st.write("无 ClaimGraph 数据")

        if state.all_attack_findings:
            with st.expander("⚔️ 攻击发现"):
                for f in state.all_attack_findings:
                    st.write(f"**[{f.severity.value}]** {f.attack_type.value}: {f.description}")
                    if f.evidence_quote:
                        st.caption(f"证据: {f.evidence_quote}")

        if state.errors:
            with st.expander("🐛 错误详情"):
                for i, error in enumerate(state.errors, 1):
                    st.write(f"**Error {i}:** {error}")
