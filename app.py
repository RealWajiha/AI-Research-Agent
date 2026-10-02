import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Assistant")
st.write("Your intelligent AI-powered research agent.")


# --------------------------------
# API KEY
# --------------------------------

api_key = st.text_input(
    "🔑 Enter your OpenAI API Key",
    type="password"
).strip()

if not api_key:
    st.info("Enter your API key to start the Research Agent.")
    st.stop()


# --------------------------------
# AGENT STATE
# --------------------------------

class ResearchState(TypedDict):
    topic: str
    response: str


# --------------------------------
# LLM
# --------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=api_key
)


# --------------------------------
# RESEARCH AGENT
# --------------------------------

def research_agent(state: ResearchState):

    topic = state["topic"]

    prompt = f"""
You are an AI Research Assistant.

Research topic:
{topic}

Analyze the topic and create a structured research response.

Include:

1. Introduction
2. Key concepts
3. Important points
4. Applications
5. Advantages
6. Challenges
7. Future scope
8. Conclusion

Keep the information clear, structured and useful
for a student or beginner researcher.
"""

    messages = [
        SystemMessage(
            content="You are an intelligent research assistant."
        ),
        HumanMessage(content=prompt)
    ]

    result = llm.invoke(messages)

    return {
        "response": result.content
    }


# --------------------------------
# LANGGRAPH
# --------------------------------

graph = StateGraph(ResearchState)

graph.add_node(
    "research_agent",
    research_agent
)

graph.add_edge(
    START,
    "research_agent"
)

graph.add_edge(
    "research_agent",
    END
)

agent = graph.compile()


# --------------------------------
# USER INTERFACE
# --------------------------------

st.subheader("📚 Enter Research Topic")

topic = st.text_area(
    "Research Topic",
    placeholder="Example: Applications of Generative AI in Healthcare",
    height=150
)


if st.button("🚀 Start Research"):

    if not topic.strip():

        st.warning(
            "Please enter a research topic."
        )

    else:

        with st.spinner(
            "🤖 Research Agent is analyzing..."
        ):

            result = agent.invoke({
                "topic": topic,
                "response": ""
            })

        st.success(
            "Research completed!"
        )

        st.subheader("📝 Research Report")

        st.markdown(
            result["response"]
        )