from crewai.tools import tool
from duckduckgo_search import DDGS

@tool("DuckDuckGo Search")
def duckduckgo_search_tool(query: str) -> str:
    """Searches the web using DuckDuckGo and returns top results with titles and snippets."""
    results = []
    try:
        with DDGS() as ddgs:
            # Notice query passed as positional argument
            search_results = list(ddgs.text(query, max_results=5))
            
            if not search_results:
                return f"No results found for query: '{query}'."
                
            for r in search_results:
                title = r.get("title", "No Title")
                url = r.get("href", r.get("link", ""))
                snippet = r.get("body", r.get("snippet", ""))
                results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")
                
        return "\n---\n".join(results)
    except Exception as e:
        return f"Error executing web search: {str(e)}"
