# Initial Imports
import os
import streamlit as st
import shutil
import subprocess
import tempfile
import re
import pdfplumber
import json
from typing import Annotated, TypedDict, Union
from langchain_classic.agents import tools
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

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
llm = ChatOllama(model="qwen2.5-coder:14b", temperature=0.0, top_p=0.0).bind_tools(tools)


# State Class is defined
class AgentState(TypedDict):
    # Add_messages is a specialized reducer that appends new messages to history
    messages: Annotated[list[BaseMessage], add_messages]
    rubric: str
    # Output is used to store and display the output returned by the java_compiler tool
    counter: int
    outputs: Annotated[list[BaseMessage], add_messages]

def create_constraints(state: AgentState):
    messages = state['messages']
    grading_rubric = state['rubric']
    system_prompt = SystemMessage(
        content="Analyze each row of this rubric and create a list of things that need to be tested. ONLY use the tools offered to you. DO NOT create functions" + grading_rubric)
    messages = [system_prompt] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}

def run_code(state: AgentState):
    messages = state['messages']
    last_message = state['messages'][-1]
    outputs = state['outputs']
    if isinstance(last_message, ToolMessage):
        outputs = outputs + [last_message]
        return {"messages": messages, "outputs": outputs}
    outputs = state['outputs']
    code = messages[0].content
    if len(messages) <= 1:
        system_prompt = SystemMessage(content="You MUST use the java_compiler tool to execute the code.")
        messages = [system_prompt] + messages
    else:
        system_prompt = SystemMessage(content="You are a grading assistant. Based on tool outputs, does this code run successfully? Respond with ONLY JSON:"
                                              "{{"
                                              "Does code compile and run successfully (Yes/No): "
                                              "Output returned by the Java compiler tool: "
                                              "}}")
        messages = [system_prompt] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}

def code_should_continue(state: AgentState):
    """
    The Conditional Edge.
    Checks if the last message contains tool calls.
    """
    messages = state['messages']
    last_message = state['messages'][-1]
    tool_output = False
    if isinstance(last_message, ToolMessage):
        return "code_output"
    last_message = state['messages'][-1]
    outputs = state['outputs']
    # If LLM requested a tool, route to the 'tools' node
    if last_message.tool_calls:
        return "tools"
    if tool_output:
        return "code_output"

        # If the LLM didn't request a tool, loop is finished
    return "run_code"

def code_output(state: AgentState):
    outputs = state['outputs']
    system_prompt = SystemMessage(
        content="You are a grading assistant. Based on tool outputs, does this code run successfully? Respond with ONLY JSON:"
                "{{"
                "Does code compile and run successfully (Yes/No): "
                "Output returned by the Java compiler tool: "
                "}}")
    outputs = [system_prompt] + outputs
    response = llm.invoke(outputs)
    return {"messages": [response]}

def call_model(state: AgentState):
    """
    The 'Thought' Node.
    The model examines the message history and decides whether
    to call a tool or provide a final answer.
    """
    messages = state['messages']
    grading_rubric = json.dumps(state['rubric'], indent=2)


    system_prompt = SystemMessage(content="What is 2+2")
    # If tool was called before, the LLM's response and the output from the tool are returned

    # Inject system instructions if this is the start of the thread
    if len(messages) <= 1:
        rubric_message = HumanMessage(content=grading_rubric)
        messages = [rubric_message] + messages
        system_prompt = SystemMessage(
            content= "What is 2+2 \n"
        )

    #if len(messages) > 5:
        #prompt = HumanMessage("You must call the java_compiler tool or your response is INVALID")
        #messages = [prompt] + messages

    # Returns the LLM's response when the tool was not called
    response = llm.invoke(messages + [system_prompt])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """
    The Conditional Edge.
    Checks if the last message contains tool calls.
    """
    messages = state['messages']
    last_message = state['messages'][-1]
    # If LLM requested a tool, route to the 'tools' node
    if last_message.tool_calls:
        return "tools"
    for m in messages:
        if isinstance(m, ToolMessage):
            return END
    #if "compilation" in last_message.content:
        #return "agent"
        # If the LLM didn't request a tool, loop is finished
    return END

def grade_code(state: AgentState):
    messages = state['messages']
    grading_rubric = state['rubric']
    system_prompt = SystemMessage(
        content= "You are a grading assistant. Provide a grade based on ONLY the rubric and test results. Take off points for syntax errors, missing requirements, and failed tests. \n"
                 "You MUST return the final grade, the grade breakdown, and the reasoning behind each grade \n"
                 "You MUST base your grade off of tool outputs. If the necessary result to provide a grade is missing provide a grade of N/A"
                 "If you provide any grades without using the tool for each one your answer is WRONG \n"
                 "This is the rubric:" + grading_rubric + "\n"
                 "These are the results: "

    )
    messages = [system_prompt] + messages
    response = llm.invoke([system_prompt])
    return {"messages": [response]}

# Construct the Graph
workflow = StateGraph(AgentState)

# Define the two nodes in the cycle
workflow.add_node("constraints", create_constraints)

workflow.add_node("code", run_code)
workflow.add_node("code_output", code_output)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("grade", grade_code)

# Define the logic flow
workflow.set_entry_point("agent")


workflow.add_conditional_edges("code", code_should_continue)
#workflow.add_edge("tools", "code")
workflow.add_edge("code_output", END)

workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", should_continue)
#workflow.add_edge("tools", "agent") # Loops back after tool execution
workflow.add_edge("grade", END)

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
        result = app.invoke({"messages": [HumanMessage(content=file_input.read().decode("utf-8"))], "rubric": rubric_text, "counter": 0, "output":[]})
        # Displays the output from the code and the LLM's response to it
        #final_answer = result["messages"][-1].content
        #st.session_state.chat_history.append(AIMessage(final_answer))
        # Used for debugging
        for m in result["messages"]:
            st.session_state.chat_history.append(m)

# Sidebar for Reset button
with st.sidebar:
    st.title("Upload Grading Rubric")
    rubric = st.file_uploader(label="Please Upload a PDF File containing the grading rubric", type=["json"])
    if rubric:
        rubric_text = json.load(rubric)
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