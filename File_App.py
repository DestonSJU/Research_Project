# Initial Imports
import os
import streamlit as st
import shutil
import subprocess
import tempfile
import re
import pdfplumber
import liteparse
import json
import operator
from typing import Annotated, List, Tuple, TypedDict, Union, Literal
from langchain_classic.agents import tools
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from liteparse import LiteParse

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
code_llm = ChatOllama(model="qwen2.5-coder", temperature=0.0, top_p=0.1).bind_tools(tools, tool_choice="required")
brain_llm = ChatOllama(model="llama3.1:8b", temperature=0.0, top_p=0.1).bind_tools(tools)



# State Class is defined
class AgentState(TypedDict):
    # Add_messages is a specialized reducer that appends new messages to history
    messages: Annotated[list[BaseMessage], add_messages]
    rubric: str
    # Output is used to store and display the output returned by the java_compiler tool
    output: AIMessage
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    is_finished: bool


def code_node(state: AgentState):
    messages = state['messages']
    if len(messages) <= 1:
        system_prompt = SystemMessage(
            content="You are a grading assistant. You MUST use tools to compile and run any code. Run this code EXACTLY as it is and report whether the code compiles and runs successfully or not. DO NOT change anything about the code"
                    "You MUST pass raw Java code ONLY. Do NOT wrap it in JSON."
        )
        messages = [system_prompt] + messages
    response = code_llm.invoke(messages)
    return {"messages": [response]}

def syntax_node(state: AgentState):
    """
    The 'Thought' Node.
    The model examines the message history and decides whether
    to call a tool or provide a final answer.
    """
    messages = state['messages']
    last_message = state['messages'][-1]
    # Inject system instructions if this is the start of the thread
    if not last_message.tool_calls:
        system_prompt = SystemMessage(
            content= "You are a grading assistant. You MUST use tools to compile and run any code. Report the number of syntax errors in the original code. If there are any, fix them and run the new code until it compiles and runs successfully."
                     "You MUST pass raw Java code ONLY. Do NOT wrap it in JSON."
        )
        messages = [system_prompt]  + messages

    # Returns the LLM's response when the tool was not called
    response = code_llm.invoke(messages)
    return {"messages": [response]}

def print_status(node_name: str, subgoals: List[str], current_task: str = None):
    print(f"\n--- [NODE]: {node_name} ---")
    if subgoals:
        print(f"Subgoals remaining: {subgoals}")
    if current_task:
        print(f"Executing: {current_task}")

def planner_node(state: AgentState):
    print("\n[PLANNER]: Analyzing request and mapping subgoals...")
    initial_prompt = SystemMessage("You are a grading assistant. Evaluate and test each piece of grading criteria" + state['rubric'] +
                                   "You MUST use information in the previous messages or use tools to compile and run the code to test any inputs."
                                   "You MUST pass raw Java code ONLY. Do NOT wrap it in JSON.")
    messages = [initial_prompt] + state['messages']
    prompt = f"""Break down this request into multiple steps. You MUST have at least 1 step for every piece of grading criteria: {messages}
    Return ONLY JSON: {{"steps": ["step 1", "step 2", ...]}}"""

    output = brain_llm.invoke(prompt)

    # --- CHANGE STARTS HERE ---
    content = output.content.strip()
    if content.startswith("```"):
        # Strip triple backticks and 'json' identifier if present
        content = content.replace("```json", "").replace("```", "").strip()
    # --- CHANGE ENDS HERE ---

    try:
        plan_data = json.loads(content)
        steps = plan_data.get("steps", [])
        print_status("planner", steps)
        return {"plan": steps, "is_finished": False}
    except Exception as e:
        print(f"[PLANNER ERROR]: {e}. Falling back to dynamic search.")
        return {"plan": [state['messages']], "is_finished": False}

def execution_node(state: AgentState):
    if not state["plan"]:
        return {"is_finished": True}

    task = state["plan"][0]
    print_status("executor", state["plan"][1:], current_task=task)

    result = brain_llm.invoke(task)

    return {"past_steps": [(task, result)], "plan": state["plan"][1:]}

def replan_node(state: AgentState):
    if not state["plan"]:
        print_status("replan", [], "Synthesizing final answer")

        summary_prompt = (
            f"You are a grading assistant.\n"
            f"Grade this students code using ONLY this rubric: {state['rubric']}\n\n"
            f"You have access to these results {state['past_steps']}\n\n"
            f"Return ONLY JSON with the format: {{" +
            '"final grade": "number"' +
            '"reason": "explain why you gave that grade"' + "}"
        )

        summary = brain_llm.invoke(summary_prompt)
        try:
            res = json.loads(summary.content).get("answer", summary.content)
            return {"messages": [AIMessage(content=res)], "is_finished": True}
        except:
            # Clean up potential markdown formatting if JSON parsing fails
            clean_res = summary.content.replace("```json", "").replace("```", "").strip()
            return {"messages": [AIMessage(content=clean_res)], "is_finished": True}

    return {"is_finished": False}

def should_continue(state: AgentState) -> Literal["executor", "__end__"]:
    return END if state.get("is_finished") else "executor"

def call_model(state: AgentState):
    """
    The 'Thought' Node.
    The model examines the message history and decides whether
    to call a tool or provide a final answer.
    """
    messages = state['messages']
    grading_rubric = state['rubric']
    # If tool was called before, the LLM's response and the output from the tool are returned
    for m in messages:
        if isinstance(m, ToolMessage):
            response = brain_llm.invoke(messages)
            return {"messages": [response], "output": AIMessage(content=m.content)}
    # Inject system instructions if this is the start of the thread
    if len(messages) <= 1:
        system_prompt = SystemMessage(
            content= "You are a grading assistant. You MUST run the code to evaluate it and "
                     "give it a grade based on this rubric: " + grading_rubric,
        )
        messages = [system_prompt]  + messages

    # Returns the LLM's response when the tool was not called
    response = brain_llm.invoke(messages)
    return {"messages": [response], "output": AIMessage(content="")}

def code_node_should_continue(state: AgentState):
    """
    The Conditional Edge.
    Checks if the last message contains tool calls.
    """
    last_message = state['messages'][-1]
    # If LLM requested a tool, route to the 'tools' node
    if last_message.tool_calls:
        return "code_tools"
    # If the LLM didn't request a tool, loop is finished
    return "syntax"

def syntax_node_should_continue(state: AgentState):
    """
    The Conditional Edge.
    Checks if the last message contains tool calls.
    """
    last_message = state['messages'][-1]
    # If LLM requested a tool, route to the 'tools' node
    if last_message.tool_calls:
        return "syntax_tools"
    # If the LLM didn't request a tool, loop is finished
    return "planner"



# Construct the Graph
workflow = StateGraph(AgentState)

# Define the two nodes in the cycle
# Define the two nodes in the cycle
workflow.add_node("code", code_node)
workflow.add_node("code_tools", ToolNode(tools))

workflow.add_node("syntax", syntax_node)
workflow.add_node("syntax_tools", ToolNode(tools))

workflow.add_node("planner", planner_node)
workflow.add_node("executor", execution_node)
workflow.add_node("replanner", replan_node)

# Define the logic flow
workflow.set_entry_point("code")

workflow.add_conditional_edges("code", code_node_should_continue)
workflow.add_edge("code_tools", "code") # Loops back after tool execution

workflow.add_conditional_edges("syntax", syntax_node_should_continue)
workflow.add_edge("syntax_tools", "syntax")

workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "replanner")
workflow.add_conditional_edges("replanner", should_continue, {"executor": "executor", END: END})




# Compile into a runnable application
app = workflow.compile()


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
        result = app.invoke({"messages": [HumanMessage(content=file_input.read().decode("utf-8"))], "rubric": rubric_text})
        # Displays the output from the code and the LLM's response to it
        final_answer = result["messages"][-1].content
        st.session_state.chat_history.append(AIMessage(final_answer))

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