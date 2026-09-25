import streamlit as st
from research_crew import run_research_crew

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Single-Agent AI Research Assistant")
st.caption("Powered by CrewAI, Groq (openai/gpt-oss-120b), and DuckDuckGo")

# Safe secret retrieval
groq_api_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None

if not groq_api_key:
    st.warning("⚠️ `GROQ_API_KEY` not found in Streamlit Secrets.")
    st.info("Please set `GROQ_API_KEY` under 'Advanced settings' -> 'Secrets' in your Streamlit Cloud dashboard.")
    st.stop()

# User Input
topic = st.text_input(
    "Enter Research Topic:",
    placeholder="e.g., Breakthroughs in quantum computing hardware"
)

if st.button("Generate Research Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner("Agent is gathering web data and synthesizing findings..."):
            try:
                report = run_research_crew(topic=topic, groq_api_key=groq_api_key)
                st.success("Research Complete!")
                st.markdown("---")
                st.markdown(report)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
