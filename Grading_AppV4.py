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
from langchain_classic.schema import output
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

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
            return json.dumps({
                "compiled": True,
                "ran": True,
                "output": run_result.stdout or "File Doesn't Output Anything",
                "error": run_result.stderr
            }, indent=2)
        # Error message is returned if the code wasn't successfully compiled
        else:
            return json.dumps({
                "compiled": False,
                "ran": False,
                "output": compiled_result.stdout,
                "error": compiled_result.stderr or "Unknown Compile Error"
            }, indent=2)

# LLM setup
tools = [java_compiler]
llm = ChatOllama(model="qwen2.5-coder:latest", temperature=0.0).bind_tools(tools)
final_llm = ChatOllama(model="qwen2.5-coder:latest", temperature=0.0, top_p=0.1)

# State Class is defined
class AgentState(TypedDict):
    # Add_messages is a specialized reducer that appends new messages to history
    messages: Annotated[list[BaseMessage], add_messages]
    rubric: str
    code: str
    file_name: str
    compiler_runs: int
    last_compile_result: str
    phase: str
    working_code: str
    test_harness_code: str
    compiler_results: list[dict]

def call_model(state: AgentState):
    """
    The 'Thought' Node.
    The model follows the designated workflow to go through the required phases and call tools when neccesary.
    """

    # Get state variables
    messages = state['messages']
    rubric = state['rubric']
    code = state['code']
    compiler_runs = state.get('compiler_runs', 0)
    last_compiler_result = state.get('last_compiler_result', "")
    phase = state.get('phase', "compile_original")
    working_code = state.get('working_code') or code

    # System prompt that describes required workflow
    system_prompt = SystemMessage(
        content=(
            "You are a grading assistant. Follow this workflow exactly:\n"
            "In the compile_original phase call the java_compiler tool with the original submitted code exactly as it is with no changes.\n"
            "If the original compile result is false, fix only the syntax errors that prevent the code from compiling. Do not fix any logical or formatting errors. "
            "Then call the java_compiler tool with the fixed code. \n "
            "Once there is working code that compile sucessfully, create test harness code that uses the inputs as parameters/test values and compares the expected outputs in the rubric with the actual output."
            "The test harness should print each test's input, expected output, actual output, and Pass/Fail. \n"
            "Call java_compiler with the test_harness code. The final grade must be based on the code's compilation and whether the test harness' actual outputs match the expected outputs in the rubric.\n"
            "Return: syntax error count and syntax errors present based only on the original submitted code's first compile result. "
            "Then include fixed syntax errors, test results, rubric score, and feedback. The test results must include the actul output of the code. Do no generate output values."
            "Mention logical issues only as grading feedback, not as fixes.\n"
            "Grade only using point values and categories explicitly present in the rubric. Report the earned points for each rubric section and the final total earned score. "
            "You must use this equation to calculate final grade. Final grade equation: (Comilation Score + All Test Scores) / Total Possible Points \n"
            "Note that the java_compiler tool returns JSON with compiled, ran, output, and error fields.\n"
            "If the message history already contains java_compiler output, continue from that output instead of restarting the workflow.\n"
            f"Current phase: {phase}\n"
            f"Compiler runs already completed: {compiler_runs}. The maximum allowed is 3.\n"
            f"Most recent compiler result:\n{last_compiler_result or 'No compiler result yet.'}\n"
            f"Rubric:\n{state['rubric']}\n"
            f"Original submitted code:\n{state['code']}\n"
            f"Current working code:\n{working_code}"
        )
    )

    response = llm.invoke([system_prompt] + messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """
    The Conditional Edge.
    Checks if the last message contains tool calls.
    """
    last_message = state['messages'][-1]
    # Returns final node if the graph has looped too many times
    if state['compiler_runs'] >= 10:
        return "final"
    # Returns tool if the message is a tool call
    if last_message.tool_calls:
        return "tool"
    # Retuns tool if the message is a JSON response that contains the tool call
    if isinstance(last_message, AIMessage):
        text = last_message.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            json_message = json.loads(text)
            if json_message.get("name") == "java_compiler":
                return "tool"
        except json.JSONDecodeError:
            pass
    # Return final if there is no tool call
    return "final"

def call_tool(state: AgentState):
    """
    The Tool Node.
    Runs the compiler tool, stores the output, and controls phases.
    """
    last_message = state['messages'][-1]
    compiler_runs = state.get('compiler_runs', 0)
    phase = state['phase']
    next_phase = ""
    tool_call_id = f"java_compiler_{compiler_runs + 1}"
    structured_message = False
    working_code = state['working_code'] or state['code']
    test_harness_code = state.get('test_harness_code', "")
    compiler_results = list(state.get('compiler_results', []))

    # Calls tool if last message is a tool call
    if last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        arguments = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", tool_call_id)
        structured_message = True
    # Calls tool if the last message is a JSON response with the tool call by extracting the tool call from it
    else:
        text = last_message.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        # Throws error if JSON isn't formatted correctly
        try:
            payload = json.loads(text)
            arguments = payload.get("arguments", {})
        except json.JSONDecodeError:
            arguments = {}

    code = arguments.get("code", state['working_code'] or state['code'])
    result = java_compiler.invoke({"code": code})
    # Throws error if JSON is invalid
    try:
        compile_result = json.loads(result)
    except json.JSONDecodeError:
        compile_result = {}

    # Adds tool output to compiler results
    compiler_results.append(
        {
            "run_number": compiler_runs + 1,
            "phase": phase,
            "code": code,
            "result": compile_result
        }
    )

    # Controls phases so that the LLM knows what to do
    if phase == "compile_original":
        if compile_result.get("compiled"):
            next_phase = "create_tests"
            working_code = code
        else:
            next_phase = "fix_syntax"
    elif phase == "fix_syntax":
        if compile_result.get("compiled"):
            next_phase = "create_tests"
            working_code = code
        else:
            next_phase = "final"
            working_code = code
    elif phase == "create_tests":
        next_phase = "final"
        test_harness_code = code

    # Used for regular tool call
    if structured_message:
        tool_result_message = ToolMessage(
            content=result,
            tool_call_id=tool_call_id,
            name="java_compiler",
        )
        result_message = HumanMessage(content=(
            f"Next phase: {next_phase}. If the next phase is create_tests, create a test harness by calling the java compiler tool that uses only the rubric's inputs as parameters/test values "
            "and compares actual outputs to the expected outputs in the rubric. Only use expected outputs listed in the rubric."
            "All inputs and expected outputs must be from the rubric. To do this, you must put the test harness code in the main method instead of its own class. "
            "If there is a declared variable, replace it with the test inputs."
            )
        )
    # Used for tool calls that were in JSON response
    else:
        tool_result_message = HumanMessage(content=(
            "java_compiler output:\n"
            f"{result}\n"
            f"Next phase: {next_phase}. If the next phase is create_tests, create a test harness by calling the java compiler tool that uses only the rubric's inputs as parameters/test values "
            "and compares actual outputs to the expected outputs in the rubric. Only use expected outputs listed in the rubric. "
            "All inputs and expected outputs must be from the rubric. To do this, you must put the test harness code in the main method instead of its own class. "
            "If there is a declared variable, replace it with the test inputs."
            )
        )

    # Used for regular tool calls
    if structured_message:
        return {
            "compiler_runs": compiler_runs + 1,
            "last_compile_result": result,
            "phase": next_phase,
            "working_code": working_code,
            "test_harness_code": test_harness_code,
            "compiler_results": compiler_results,
            "messages": [tool_result_message + result_message]
        }
    # Used for tool calls that were in JSON response
    else:
        return {
            "compiler_runs": compiler_runs + 1,
            "last_compile_result": result,
            "phase": next_phase,
            "working_code": working_code,
            "test_harness_code": test_harness_code,
            "compiler_results": compiler_results,
            "messages": [tool_result_message]
        }

def tool_should_continue(state: AgentState):
    """
    The Conditional Edge.
    Checks if the phase is final or not.
    """
    compiler_runs = state.get("compiler_runs", 0)
    phase = state['phase']
    # Returns final node if the compiler has run at least 10 times or the phase is final. Otherwise, this edge loops back to agent
    if compiler_runs >= 10:
        return "final"
    if phase == "final":
        return "final"
    else:
        return "agent"

def final_response(state: AgentState):
    """
    The Final Response Node.
    Produces the final grade and the grade breakdown.
    """
    # Prompt that produces the response that contains the final grade and point distribution
    system_prompt = SystemMessage(content=(
        "Give the final grade now. Do not call tools and do not output a tool call.\n"
        "Your response must be normal text with a grade. \n"
        "Use the compiler results and the test harness output. \n"
        "Grade based on whether the actual outputs from the tests_harness code match the expected outputs in the rubric. \n"
        "If syntax errors were fixed, list them. If the test harness didn't run because compilation failed, give a grade of 0 for the tests. \n"
        "The syntax errors count and list must be based only on the original submitted code's first compile result, which is the compiler_results item with phase compile_original. \n"
        "For test inputs, report the exact input values as written in the rubric. Use only point values and grading categories explicitly present in the rubric. \n"
        "You must include the actual earned score and list each rubric section with points earned and points possible, then list the final total earned points and total possible points. \n"
        "Strict grading rules:\n"
        "- For every section, first quote or copy the exact rubric criteria text being used.\n"
        "- For every section, copy the exact point value, deduction, and category from the rubric.\n"
        "- Do not give partial credit unless the rubric explicitly says to give partial credit.\n"
        "- Syntax errors count and list should only be based on compile_original code. \n"
        "- Don't count or list syntax errors in test harness code. \n"
        "- Compilation receives full points if there are no syntax errors in the compile_original code. \n"
        "- Take off 10 points in compilation score if there are any syntax errors in the compile_original code. Otherwise Compilation receives full points as described in the rubric. \n"
        "- Don't take off points in Compilation section if there are failed tests but no 0 syntax errors. \n"
        "- Compilation score must not be based on final code. It must be based on the original code.\n"
        "- Final grade must be calculated with this equation: (Compilation Score + All test scores) / Total possible points. \n"
        "- You must check that every equation is correct when calculating grades. Ensure that both sides of the equation equal each other. \n"
        "- All calculations must be mathematically correct.\n"
        "- Don't add points to final grade for tests that failed. \n"
        "- When listing final grade, list calculation, total points, and the percentage. \n"
        "- Passed test receive all points while failed tests receive 0 points.\n"
        "- Don't pass tests if they don't have actual output.\n"
        "- Do not make up scores. Don't add points for any outputs that aren't tests.\n"
        "- Don't assume that test pass. Ensure that a tests has passed before adding the points to the final grade. \n"
        "- You must explicitly sum all earned points step by step before stating the final grade."
        "Return: Name which is the file name except the .java, syntax feedback that contains syntax error count in the original code and a list of each syntax error that was fixed, test feedback that contains pass/fail, "
        "expected output and actual output for each test, grade breakdown that contains earned/possible points for each rubric section, "
        "final grade that is Compilation score + All Test scores, and feedback.\n"
        f"Rubric:\n{state['rubric']}\n"
        f"File Name: {state['file_name']}\n"
        f"Original submitted code:\n{state['code']}\n"
        f"Working code used before harness:\n{state['working_code']}\n"
        f"Test harness code:\n{state.get('test_harness_code', '')}\n"
        f"Compiler runs completed: {state.get('compiler_runs', 0)}\n"
        f"Final phase: {state['phase']}\n"
        f"Most recent compiler result:\n{state.get('last_compile_result', '')}\n"
        f"All compiler results:\n{json.dumps(state.get('compiler_results', []), indent=2)}"
        )
    )
    response = final_llm.invoke([system_prompt])
    return {"messages": [response]}

# Construct the Graph
workflow = StateGraph(AgentState)

# Define the nodes in the workflow
workflow.add_node("agent", call_model)
workflow.add_node("tool", call_tool)
workflow.add_node("final", final_response)

# Define the logic flow
workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", should_continue)
workflow.add_conditional_edges("tool", tool_should_continue)
workflow.add_edge("final", END)

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
    files_finished = 0
    files_total = len(file_input)
    for file in file_input:
        with st.spinner("Grading Code... Finished " + str(files_finished) + " out of " + str(files_total)):
            result = app.invoke(
                {
                    "messages": [HumanMessage(content="Evaluate the submitted Java file using the required phases.")],
                    "rubric": rubric_text,
                    "code": file.getvalue().decode("utf-8"),
                    "file_name": file.name,
                    "compiler_runs": 0,
                    "last_compile_result": "",
                    "phase": "compile_original",
                    "working_code": file.getvalue().decode("utf-8"),
                    "test_harness_code": "",
                    "compiler_results": [],
                },
                {"recursion_limit": 10},
            )
            # Displays the LLM's response
            final_answer = result["messages"][-1].content
            st.session_state.chat_history.append(AIMessage(final_answer))
            files_finished = files_finished + 1

# Sidebar for Reset button and rubric upload
with st.sidebar:
    st.title("Upload Grading Rubric")
    # Uploads and stores a JSON rubric
    rubric_file = st.file_uploader(label="Please upload a JSON file containing the grading rubric", type=["json"])
    if rubric_file:
        rubric_data = json.load(rubric_file)
        rubric_text = json.dumps(rubric_data, indent=2)
        st.success("Grading rubric uploaded")
    # Reset button
    st.title("Reset Chat History")
    st.button("Reset", on_click=reset_button)

# Chat Window
# Creates a place for users to upload files and a button to start the model's evaluation
file_input = st.file_uploader(label="Please Upload a Java File to be evaluated", type=["java"], accept_multiple_files=True)
if file_input:
    st.success("Files Uploaded")
    if not rubric_file:
        st.warning("There is no rubric.")
    else:
        st.button("Evaluate Code", on_click=run_model_button)

# Display the conversation
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)