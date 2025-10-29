from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI

load_dotenv()

# OpenRouter configuration - using OpenAI models through OpenRouter
openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_api_key = ""

# Create two LLM instances for the interview
interviewer_llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    temperature=0,
    base_url=openrouter_base_url,
    api_key=openrouter_api_key,
)

interviewee_llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free", 
    temperature=0,
    base_url=openrouter_base_url,
    api_key=openrouter_api_key,
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    turn_count: int  # Track number of turns
    max_turns: int   # Maximum number of turns before ending

# System prompts for each role
interviewer_prompt = """
You are a professional HR interviewer conducting a job interview for a Software Engineer position.
You should ask thoughtful, relevant questions about:
- Technical skills and experience
- Problem-solving abilities
- Teamwork and communication
- Career goals and motivation
- Past projects and achievements

Keep your questions professional, clear, and engaging. Ask follow-up questions based on their responses.
After 5-6 exchanges, you can wrap up the interview.
"""

interviewee_prompt = """
You are a job candidate applying for a Software Engineer position. You have:
- 3 years of experience in Python and JavaScript
- Experience with web development and databases
- Strong problem-solving skills
- Good teamwork abilities
- Passion for learning new technologies

Answer questions professionally, provide specific examples from your experience, and show enthusiasm for the role.
Be honest about your skills and areas for growth.
"""

def interviewer_agent(state: AgentState) -> AgentState:
    """Interviewer asks questions."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=interviewer_prompt)] + messages
    
    response = interviewer_llm.invoke(messages)
    print(f"\n🎤 INTERVIEWER: {response.content}")
    
    return {
        'messages': [response],
        'turn_count': state['turn_count'] + 1,
        'max_turns': state['max_turns']
    }

def interviewee_agent(state: AgentState) -> AgentState:
    """Interviewee answers questions."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=interviewee_prompt)] + messages
    
    response = interviewee_llm.invoke(messages)
    print(f"\n👤 INTERVIEWEE: {response.content}")
    
    return {
        'messages': [response],
        'turn_count': state['turn_count'] + 1,
        'max_turns': state['max_turns']
    }

def should_continue(state: AgentState):
    """Check if interview should continue."""
    return state['turn_count'] < state['max_turns']


# Build the conversation graph
graph = StateGraph(AgentState)

# Add nodes for each agent
graph.add_node("interviewer", interviewer_agent)
graph.add_node("interviewee", interviewee_agent)

# Add conditional edges - alternate between interviewer and interviewee
graph.add_conditional_edges(
    "interviewer",
    should_continue,
    {True: "interviewee", False: END}
)
graph.add_conditional_edges(
    "interviewee", 
    should_continue,
    {True: "interviewer", False: END}
)

# Set entry point
graph.set_entry_point("interviewer")

# Compile the graph
interview_agent = graph.compile()

def run_interview():
    """Run the interview simulation."""
    print("\n" + "="*50)
    print("🎯 JOB INTERVIEW SIMULATOR")
    print("="*50)
    print("Interviewer: HR Manager")
    print("Interviewee: Software Engineer Candidate")
    print("Position: Software Engineer")
    print("="*50)
    
    # Initialize the interview
    initial_state = {
        "messages": [HumanMessage(content="Hello! Welcome to our interview. Let's begin.")],
        "turn_count": 0,
        "max_turns": 10  # 5 exchanges (10 total messages)
    }
    
    print("\n🎤 INTERVIEWER: Hello! Welcome to our interview. Let's begin.")
    
    # Run the interview
    result = interview_agent.invoke(initial_state)
   

if __name__ == "__main__":
    run_interview()