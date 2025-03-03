# Deep Research AI Agentic System

## Overview
The Deep Research AI Agentic System is a sophisticated, dual-agent architecture designed to conduct in-depth research and generate structured, well-informed responses. It is built using LangChain, LangGraph, Tavily, and Google Gemini APIs, showcasing modularity, scalability, and attention to detail.

## System Architecture
- **Research Agent:** Uses Tavily for real-time web data retrieval, focusing on raw factual information.
- **Drafting Agent:** Leverages Google Gemini 1.5 Pro for natural language processing and answer generation.
- **Workflow Management:** Utilizes LangGraph to manage the seamless flow of data between agents.

## Key Features
- **Dynamic Model Adaptation:** Switched from OpenAI to Google Gemini due to quota constraints.
- **Error Handling:** Implements robust mechanisms for API and processing errors.
- **Future Expansion:** Designed with a vision for autonomous research and multimodal input capabilities.

## Installation
```bash
pip install langchain==0.3.19 langchain-community==0.3.18 langchain-core==0.3.37 langchain-openai==0.2.1 langchain-text-splitters==0.3.6 langgraph==0.2.39 langgraph-checkpoint==2.0.16 langgraph-sdk==0.1.53 pydantic==2.7.4 pydantic_core==2.18.4 pydantic-settings==2.8.1 tavily-python==0.5.0 python-dotenv
```

## Usage
```bash
python deep_research.py
```

## Example
```bash
Enter your research query: What are the latest trends in AI research?
Research Results:
[Detailed output from the AI system]
```

## Future Enhancements
- **Integration with X Search** to expand research capabilities.
- **Implementation of Caching** for faster responses.
- **Support for Multimodal Inputs**, such as PDFs and images.

## License
This project is licensed under the MIT License.