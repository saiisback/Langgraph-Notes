from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values
import os

load_dotenv()

# OpenRouter configuration - using OpenAI models through OpenRouter
openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_api_key = ""

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url=openrouter_base_url,
    api_key=openrouter_api_key,
    
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
