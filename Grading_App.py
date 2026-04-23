# Initial Imports
import os
import streamlit as st
import shutil
import subprocess
import tempfile
import re
import pdfplumber
from typing import Annotated, TypedDict, Union
from langchain_classic.agents import tools, create_react_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import MessagesState
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Annotated, Literal, TypedDict
from langchain_core.runnables.config import RunnableConfig
from operator import add
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Optional


@tool
def java_compiler(code: str):
    """Compiles and Executes Java code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary Directory to run the code in and copy the code to a file inside it
        base_directory = os.path.dirname(os.path.abspath(__file__))
        java_bin = os.path.join(base_directory, "OpenJDK25U", "jdk-25.0.2+10", "bin")
        match = re.search(r'class\s+(\w+)', code)
        if match:
            class_name = match.group(1)
        else:
            class_name = "Main"
        student_file = os.path.join(tmpdir, f"{class_name}.java")
        with open(student_file, "w") as file:
            file.write(code)
        # Compile the code and run the code if it was successfully compiled. The output is then returned
        compiled_result = subprocess.run([os.path.join(java_bin, "javac.exe"), student_file], capture_output=True, text=True, cwd=tmpdir)
        if compiled_result.returncode == 0:
            run_result = subprocess.run([os.path.join(java_bin, "java.exe"), "-cp", tmpdir, class_name], capture_output=True, text=True, timeout=10)
            return run_result.stdout or run_result.stderr or "File doesn't Output Anything"
        # Error message is returned if the code wasn't successfully compiled
        else:
            return compiled_result.stdout or compiled_result.stderr or "Unknown Error."

# Java Compiler is added to tools that the LLM can use
tools = [java_compiler]
llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0.0).bind_tools(tools)

system_prompt = """You are a grading assistant. Follow this procedure to grade this student's code:
Compile and run the java code using the tool. DO NOT change the code in any way \n
If code compiles and run successfully, note that there are no syntax errors
If the code doesn't compile and run successfully, state how many syntax errors there are and fix them,
"then compile and run the fixed code using the tool. \n
Next use the rubric to create tests to determine if a given input returns the expected output.
You MUST create these test based on the rubric. For each test there MUST be an input and expected output. \n
Next run each test by doing the following: For each test you MUST change the code changing the main method to include the input in the created test, then compile and run the new java code using the tool,
then state whether the output of the code matches the expected output \n
Next grade the code based on the rubric. Take off points for syntax errors, missing requirements, and failed tests. \n
You MUST return the final grade, the grade breakdown, and the reasoning behind each grade \n
You MUST base your grade off of tool outputs. If the necessary result to provide a grade is missing provide a grade of N/A
If you provide any grades without using the tool for each one your answer is WRONG \n
STRICT RULES: NEVER simulate compiling or running code. NEVER guess outputs for any code. You MUST use the tool to compile or run all code and tests.
For each created test, there MUST be a tool call to compile and run the code. DO NOT hallucinate, you MUST use the tool to run all code or the step is not finished
If you provide any grades without using the tool for each one your answer is WRONG"""


prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt),
     ("placeholder", "{messages}")
     ]

)

# State Class is defined

agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

class Response(BaseModel):
    """A final response to the user."""

    answer: Optional[str] = Field(
        description="The final grade.",
        default=None,
    )


class AgentState(TypedDict):
    code: str
    rubric: str
    answer: str
    steps: Annotated[int, add]
    response: Response

def should_end(state: AgentState, config: RunnableConfig) -> Literal["grade", END]:
    max_reasoning_steps = config["configurable"].get("max_reasoning_steps", 10)
    if "100" in state.get("answer"):
        return END
    if state.get("steps", 1) > max_reasoning_steps:
        return END
    return "grade"

def grade(state):
    code = state["code"]
    grading_rubric = state["rubric"]
    files = HumanMessage(content=f"""Grade the code with this rubric. \n
                                 Code: \n {code} \n
                                 Rubric: \n {grading_rubric} \n""")


    result = agent.invoke({"messages": [files]})
    return {"answer": result["messages"][-1].content, "steps": 1}

# Construct the Graph
builder = StateGraph(AgentState)

# Define the two nodes in the cycle
builder.add_node("grade", grade)

# Define the logic flow
builder.set_entry_point("grade")
builder.add_conditional_edges("grade", should_end)

# Compile into a runnable application
graph = builder.compile()


# Streamlit Interface
st.set_page_config(page_title="AI Grading Bot", layout="wide")
st.title("AI Grading Assistant")

# Create chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Clears the chat history
def reset_button():
    st.session_state.chat_history.clear()
# Tells the model to run and generate a response
def run_model_button():
    with st.spinner("Executing Local Workflow..."):
        for event in graph.stream({"code": file_input.read().decode("utf-8"), "rubric": rubric_text, "answer": "" }, stream_mode=["updates"]):
            st.session_state.chat_history.append(AIMessage(str(event)))

# Sidebar for Reset button
with st.sidebar:
    st.title("Upload Grading Rubric")
    rubric = st.file_uploader(label="Please Upload a PDF File containing the grading rubric", type=["pdf"])
    if rubric:
        rubric_text = ""
        with pdfplumber.open(rubric) as pdf:
            for page in pdf.pages:
                rubric_text = rubric_text + page.extract_text()
        st.success("Grading Rubric Uploaded")
    st.title("Reset Chat History")
    st.button("Reset", on_click= reset_button)

# Chat Window
# Creates a place for users to upload files and a button to start the model's evaluation
file_input = st.file_uploader(label="Please Upload a Java File to be evaluated", type=["java"])
if file_input:
    st.session_state.chat_history.append(AIMessage("Files Uploaded: " + "\n\n" + file_input.name))
    st.button("Evaluate Code", on_click=run_model_button)

# Display the conversation
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)