from crewai.tools import tool
from duckduckgo_search import DDGS

@tool("DuckDuckGo Search")
def duckduckgo_search_tool(query: str) -> str:
    """Searches the web using DuckDuckGo and returns top results with titles and snippets."""
    results = []
    try:
        # DDGS context manager handles connection lifecycle automatically
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=5))
            
            if not search_results:
                return f"No web search results found for query: '{query}'."
                
            for r in search_results:
                title = r.get("title", "No Title")
                url = r.get("href", r.get("link", ""))
                body = r.get("body", r.get("snippet", ""))
                results.append(f"Title: {title}\nURL: {url}\nSnippet: {body}\n")
                
        return "\n---\n".join(results)
    except Exception as e:
        return f"Error executing DuckDuckGo web search: {str(e)}"
