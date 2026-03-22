# Initial Imports
import os
import streamlit as st
from typing import TypedDict, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# --- 1. MODEL CONFIGURATION ---
# Gemma writes the final answer
writer_llm = ChatOllama(model="gemma2:2b", temperature=0.3)
# DeepSeek handles the logical/compliance audit if necessary
reasoner_llm = ChatOllama(model="deepseek-r1:1.5b")
# mxbai handles the document search
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = InMemoryVectorStore(embedding=embeddings)

# --- 2. STATE & GRAPH LOGIC ---
class State(TypedDict):
    messages: List
    context: str
    answer: str

def retrieve(state: State):
    """Librarian: Pulls the right facts from your documents."""
    query = state["messages"][-1].content
    docs = vector_store.similarity_search(query, k=2)
    context_text = "\n\n".join([d.page_content for d in docs])
    return {"context": context_text}

def generate(state: State):
    """Writer: Gemma-2b drafts a professional response."""
    prompt = f"Using ONLY this context: {state['context']}, answer: {state['messages'][-1].content}"
    response = writer_llm.invoke(prompt)
    return {"answer": response.content}

# Building the workflow: Search -> Write -> Done
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# --- 3. STREAMLIT INTERFACE ---
st.set_page_config(page_title="AI Grading Bot", layout="wide")
st.title("AI Grading Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for managing documents
with st.sidebar:
    st.header("Upload Directory")
    uploaded_directory_path = st.text_input("Insert a Directory Path")
    # Check if directory is a valid path
    if os.path.isdir(uploaded_directory_path):
        uploaded_directory = DirectoryLoader(path=uploaded_directory_path, recursive=True)
        if uploaded_directory:
            # Will load the files and display the directory contents if
            # All the files can be read successfully
            try:
                documents = uploaded_directory.load()
                vector_store.add_documents(documents)
                st.success("Valid Directory Uploaded")
                st.text("Directory Contents:")
                st.text(os.listdir(uploaded_directory_path))
            # Exception is thrown if there is a file that can't be read
            except Exception as e:
                st.error("There was an error reading some files in this directory")
    else:
        st.error("No Valid Directory Path")

# Chat Window
user_input = st.chat_input("Ask a question about the contents of the directory")
if user_input:
    st.session_state.chat_history.append(HumanMessage(user_input))

    with st.spinner("Executing Local Workflow..."):
        result = graph.invoke({"messages": st.session_state.chat_history})
        final_answer = result["answer"]
        st.session_state.chat_history.append(AIMessage(final_answer))

# Display the conversation
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)