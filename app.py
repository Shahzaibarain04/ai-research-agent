import streamlit as st
from research_crew import run_research_crew

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Single-Agent AI Research Assistant")
st.caption("Powered by CrewAI, Groq (openai/gpt-oss-120b), and DuckDuckGo")

# Safe secret extraction to handle unconfigured initial deployment states
groq_api_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None

if not groq_api_key:
    st.warning("⚠️ `GROQ_API_KEY` not found in Streamlit Secrets.")
    st.info("Please configure `GROQ_API_KEY` under 'Advanced settings' -> 'Secrets' in your Streamlit Cloud dashboard.")
    st.stop()

# User Input Interface
topic = st.text_input(
    "Enter Research Topic:",
    placeholder="e.g., Recent breakthroughs in quantum computing hardware"
)

if st.button("Generate Research Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        with st.spinner("Agent is querying DuckDuckGo and compiling the report..."):
            try:
                report = run_research_crew(topic=topic, groq_api_key=groq_api_key)
                st.success("Research Complete!")
                st.markdown("---")
                st.markdown(report)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
