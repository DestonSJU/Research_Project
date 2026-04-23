# "You MUST perform these steps to grade this student's code:"
#                                              "Step 1: Compile and run the java code using the tool. DO NOT change the code in any way \n"
#                                              "Step 2: If code compiles and run successfully, note that there are no syntax errors"
#                                              "If the code doesn't compile and run successfully, state how many syntax errors there are and fix them,"
#                                              "then compile and run the fixed code using the tool. \n"
#                                              "Step 3: Next use the rubric to create tests to determine if a given input returns the expected output."
#                                              "You MUST create these test based on the rubric. For each test there MUST be an input and expected output. \n"
#                                              "Step 4: Next run each test by doing the following: For each test you MUST change the code changing the main method to include the input in the created test, then compile and run the new java code using the tool,"
#                                              "then state whether the output of the code matches the expected output \n"
#                                              "Step 5: Next grade the code based on the rubric. Take off points for syntax errors, missing requirements, and failed tests. \n"
#                                              "You MUST return the final grade, the grade breakdown, and the reasoning behind each grade \n"
#                                              "You MUST base your grade off of tool outputs. If the necessary result to provide a grade is missing provide a grade of N/A"
#                                              "If you provide any grades without using the tool for each one your answer is WRONG \n"
#                                              "STRICT RULES: NEVER simulate compiling or running code. NEVER guess outputs for any code. You MUST use the tool to compile or run all code and tests."
#                                              "For each step, you must provide the tool output as proof for your reasoning for each step"
#                                              "For each created test, there MUST be a tool call to compile and run the code. DO NOT hallucinate, you MUST use the tool to run all code or the step is not finished"
#                                              "If you provide any grades without using the tool for each one your answer is WRONG and INVALID"
#                                              "CRITICAL RULE: You MUST provide tool output with every grade and reasoning you give. If there is no tool output, the response is INVALID and you MUST try again."


# Initial Imports
import os
import streamlit as st
import shutil
import subprocess
import tempfile
import re
import pdfplumber
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
llm = ChatOllama(model="mistral-nemo", temperature=0.0).bind_tools(tools)


# State Class is defined
class AgentState(TypedDict):
    # Add_messages is a specialized reducer that appends new messages to history
    messages: Annotated[list[BaseMessage], add_messages]
    rubric: str
    # Output is used to store and display the output returned by the java_compiler tool
    counter: int
    output: Annotated[list[ToolMessage], add_messages]

def call_model(state: AgentState):
    """
    The 'Thought' Node.
    The model examines the message history and decides whether
    to call a tool or provide a final answer.
    """
    messages = state['messages']
    last_message = state['messages'][-1]
    grading_rubric = state['rubric']
    # If tool was called before, the LLM's response and the output from the tool are returned

    system_prompt = SystemMessage(content="You are a grading assistant. This is your rubric: " + grading_rubric)
    # Inject system instructions if this is the start of the thread
    if len(messages) <= 1:
        system_prompt = SystemMessage(
            content= system_prompt.content + "Follow this procedure to grade this student's code:"
                                             "Compile and run the java code using the tool. DO NOT change the code in any way \n"
                                             "If code compiles and run successfully, note that there are no syntax errors"
                                             "If the code doesn't compile and run successfully, state how many syntax errors there are and fix them,"
                                             "then compile and run the fixed code using the tool. \n"
                                             "Next use the rubric to create tests to determine if a given input returns the expected output."
                                             "You MUST create these test based on the rubric. For each test there MUST be an input and expected output. \n"
                                             "Next run each test by doing the following: For each test you MUST change the code changing the main method to include the input in the created test, then compile and run the new java code using the tool,"
                                             "then state whether the output of the code matches the expected output \n"
                                             "Next grade the code based on the rubric. Take off points for syntax errors, missing requirements, and failed tests. \n"
                                             "You MUST return the final grade, the grade breakdown, and the reasoning behind each grade \n"
                                             "You MUST base your grade off of tool outputs. If the necessary result to provide a grade is missing provide a grade of N/A"
                                             "If you provide any grades without using the tool for each one your answer is WRONG \n"
                                             "STRICT RULES: NEVER simulate compiling or running code. NEVER guess outputs for any code. You MUST use the tool to compile or run all code and tests."
                                             "For each created test, there MUST be a tool call to compile and run the code. DO NOT hallucinate, you MUST use the tool to run all code or the step is not finished"
                                             "If you provide any grades without using the tool for each one your answer is WRONG"
        )
    messages = [system_prompt] + messages


    # Returns the LLM's response when the tool was not called
    response = llm.invoke(messages)
    if not response.tool_calls:
        return {"messages": [AIMessage(content="Call the tool")]}
    return {"messages": [response]}

def should_continue(state: AgentState):
    """
    The Conditional Edge.
    Checks if the last message contains tool calls.
    """
    messages = state['messages']
    last_message = state['messages'][-1]
    output = state['output']
    # If LLM requested a tool, route to the 'tools' node
    if last_message.tool_calls:
        return "tools"
    if len(messages) > 7:
        return END
        # If the LLM didn't request a tool, loop is finished
    return "agent"

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
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("grade", grade_code)

# Define the logic flow
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # Loops back after tool execution
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