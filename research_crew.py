from crewai import Agent, Task, Crew, LLM
from search_tool import duckduckgo_search_tool

def run_research_crew(topic: str, groq_api_key: str) -> str:
    """Configures and runs the CrewAI research agent using Groq via OpenAI-compatible endpoint."""
    
    # Prefix with 'openai/' to use CrewAI's native OpenAI SDK, 
    # but direct base_url to Groq's endpoint
    llm = LLM(
        model="openai/gpt-oss-120b",
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.3
    )

    researcher = Agent(
        role="Lead AI Research Specialist",
        goal=f"Perform detailed web research on '{topic}' and compile findings.",
        backstory=(
            "You are an expert intelligence analyst proficient in searching the web, "
            "verifying details, and structuring insights clearly."
        ),
        tools=[duckduckgo_search_tool],
        llm=llm,
        verbose=True
    )

    research_task = Task(
        description=(
            f"Search the web for accurate information regarding: '{topic}'.\n"
            "Collect key information, developments, and facts.\n"
            "Format your findings into a clean Markdown research report with key sections: "
            "Executive Summary, Detailed Analysis, and Key Takeaways."
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
