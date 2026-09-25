from crewai.tools import tool
from duckduckgo_search import DDGS

@tool("DuckDuckGo Search")
def duckduckgo_search_tool(query: str) -> str:
    """Searches the web using DuckDuckGo and returns top results with titles and snippets."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Error executing web search: {str(e)}"
