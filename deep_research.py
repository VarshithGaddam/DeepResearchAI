import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# API keys (hardcoded as provided)
TAVILY_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyDIoC5h2uluEihqkaoDHIBZTFC1wbVHWzk"

# Set environment variables
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Define the State structure for LangGraph
class ResearchState(TypedDict):
    query: str
    research_data: List[str]
    draft_answer: str
    final_answer: str

# Initialize Tavily Search Tool
tavily_tool = TavilySearchResults(max_results=5)

# Define tools
tools = [
    Tool(
        name="TavilySearch",
        func=tavily_tool.invoke,
        description="Search the web using Tavily for current information"
    )
]

# Initialize LLM with Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Research Agent
research_prompt = ChatPromptTemplate.from_template("""
You are a meticulous research agent. Your task is to gather comprehensive, accurate information 
relevant to the user's query using available tools. Focus on collecting raw data and facts 
without generating conclusions.

Query: {query}

Return a list of relevant findings from your research.
""")
research_agent = research_prompt | llm.bind_tools(tools)

# Answer Drafting Agent
drafting_prompt = ChatPromptTemplate.from_template("""
You are an expert answer drafting agent. Your task is to analyze the research data provided 
and create a clear, concise, and well-structured response to the original query.

Query: {query}
Research Data: {research_data}

Provide a draft answer based on the information given.
""")
drafting_agent = drafting_prompt | llm

# Define Agent Nodes
def research_node(state: ResearchState) -> ResearchState:
    result = research_agent.invoke({"query": state["query"]})
    research_data = [str(item) for item in result.content] if isinstance(result.content, list) else [result.content]
    return {"research_data": research_data}

def drafting_node(state: ResearchState) -> ResearchState:
    draft = drafting_agent.invoke({
        "query": state["query"],
        "research_data": "\n".join(state["research_data"])
    })
    return {"draft_answer": draft.content}

def review_node(state: ResearchState) -> ResearchState:
    final_answer = state["draft_answer"]
    return {"final_answer": final_answer}

# Build Workflow Graph
workflow = StateGraph(ResearchState)
workflow.add_node("research", research_node)
workflow.add_node("drafting", drafting_node)
workflow.add_node("review", review_node)
workflow.add_edge("research", "drafting")
workflow.add_edge("drafting", "review")
workflow.add_edge("review", END)
workflow.set_entry_point("research")
graph = workflow.compile()

# Main execution function
def run_deep_research(query: str) -> str:
    try:
        initial_state = ResearchState(
            query=query,
            research_data=[],
            draft_answer="",
            final_answer=""
        )
        result = graph.invoke(initial_state)
        return result["final_answer"]
    except Exception as e:
        return f"Error occurred during research: {str(e)}"
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Load API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if keys are provided
if not TAVILY_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Please set TAVILY_API_KEY and GOOGLE_API_KEY environment variables.")

# Set environment variables (optional redundancy for runtime)
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Define the State structure for LangGraph
class ResearchState(TypedDict):
    query: str
    research_data: List[str]
    draft_answer: str
    final_answer: str

# Initialize Tavily Search Tool
tavily_tool = TavilySearchResults(max_results=5)

# Define tools
tools = [
    Tool(
        name="TavilySearch",
        func=tavily_tool.invoke,
        description="Search the web using Tavily for current information"
    )
]

# Initialize LLM with Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Research Agent
research_prompt = ChatPromptTemplate.from_template("""
You are a meticulous research agent. Your task is to gather comprehensive, accurate information 
relevant to the user's query using available tools. Focus on collecting raw data and facts 
without generating conclusions.

Query: {query}

Return a list of relevant findings from your research.
""")
research_agent = research_prompt | llm.bind_tools(tools)

# Answer Drafting Agent
drafting_prompt = ChatPromptTemplate.from_template("""
You are an expert answer drafting agent. Your task is to analyze the research data provided 
and create a clear, concise, and well-structured response to the original query.

Query: {query}
Research Data: {research_data}

Provide a draft answer based on the information given.
""")
drafting_agent = drafting_prompt | llm

# Define Agent Nodes
def research_node(state: ResearchState) -> ResearchState:
    result = research_agent.invoke({"query": state["query"]})
    research_data = [str(item) for item in result.content] if isinstance(result.content, list) else [result.content]
    return {"research_data": research_data}

def drafting_node(state: ResearchState) -> ResearchState:
    draft = drafting_agent.invoke({
        "query": state["query"],
        "research_data": "\n".join(state["research_data"])
    })
    return {"draft_answer": draft.content}

def review_node(state: ResearchState) -> ResearchState:
    final_answer = state["draft_answer"]
    return {"final_answer": final_answer}

# Build Workflow Graph
workflow = StateGraph(ResearchState)
workflow.add_node("research", research_node)
workflow.add_node("drafting", drafting_node)
workflow.add_node("review", review_node)
workflow.add_edge("research", "drafting")
workflow.add_edge("drafting", "review")
workflow.add_edge("review", END)
workflow.set_entry_point("research")
graph = workflow.compile()

# Main execution function
def run_deep_research(query: str) -> str:
    try:
        initial_state = ResearchState(
            query=query,
            research_data=[],
            draft_answer="",
            final_answer=""
        )
        result = graph.invoke(initial_state)
        return result["final_answer"]
    except Exception as e:
        return f"Error occurred during research: {str(e)}"

# Interactive CLI
if __name__ == "__main__":
    print("Deep Research AI Agentic System")
    print("Type 'quit' to exit")
    while True:
        query = input("\nEnter your research query: ")
        if query.lower() == "quit":
            print("Exiting...")
            break
        response = run_deep_research(query)
        print("\nResearch Results:")
        print(response)
# Interactive CLI
if __name__ == "__main__":
    print("Deep Research AI Agentic System")
    print("Type 'quit' to exit")
    while True:
        query = input("\nEnter your research query: ")
        if query.lower() == "quit":
            print("Exiting...")
            break
        response = run_deep_research(query)
        print("\nResearch Results:")
        print(response)