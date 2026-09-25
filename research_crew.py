from crewai import Agent, Task, Crew, LLM
from search_tool import duckduckgo_search_tool

def run_research_crew(topic: str, groq_api_key: str) -> str:
    """Configures and runs the CrewAI research agent using Groq's OpenAI-compatible endpoint."""
    
    # Configure LLM using CrewAI's native OpenAI adapter pointed at Groq
    # Double-prefix pattern ensures compatibility without litellm
    llm = LLM(
        model="openai/openai/gpt-oss-120b",
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.3
    )

    researcher = Agent(
        role="Lead AI Research Analyst",
        goal=f"Conduct accurate and deep web research on '{topic}'.",
        backstory=(
            "You are a skilled technical researcher. You excel at searching the web, "
            "evaluating information sources, and compiling structured, clear summaries."
        ),
        tools=[duckduckgo_search_tool],
        llm=llm,
        verbose=True
    )

    research_task = Task(
        description=(
            f"Perform web research on the topic: '{topic}'.\n"
            "Use the search tool to collect accurate facts, news, and key details.\n"
            "Synthesize your findings into a comprehensive Markdown research report featuring:\n"
            "- Executive Summary\n"
            "- Key Discoveries & Analysis\n"
            "- Conclusion & Future Outlook"
        ),
        expected_output="A full markdown report with key findings and structured headings.",
        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )

    result = crew.kickoff()
    return str(result)
