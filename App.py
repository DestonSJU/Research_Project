import streamlit as st
from typing import TypedDict, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. MODEL CONFIGURATION ---
# Gemma writes the final answer
writer_llm = ChatOllama(model="gemma2:2b", temperature=0.3)
# DeepSeek handles the logical/compliance audit
reasoner_llm = ChatOllama(model="deepseek-r1:1.5b")
# mxbai handles the document search
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = InMemoryVectorStore(embedding=embeddings)


# --- 2. STATE & GRAPH LOGIC ---
class State(TypedDict):
    messages: List
    context: str
    answer: str
    compliance_report: str


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


def compliance_check(state: State):
    """Auditor: DeepSeek-R1 checks the answer for corporate errors."""
    check_prompt = f"Analyze this answer for corporate compliance and accuracy based on the provided data: {state['answer']}"
    # DeepSeek uses 'thinking' to find flaws
    audit = reasoner_llm.invoke(check_prompt)
    return {"compliance_report": audit.content}


# Building the workflow: Search -> Write -> Audit -> Done
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("compliance_check", compliance_check)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "compliance_check")
builder.add_edge("compliance_check", END)
graph = builder.compile()

# --- 3. STREAMLIT INTERFACE ---
st.set_page_config(page_title="Local Doc Bot", layout="wide")
st.title("📂 Local Corporate Documentation Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for managing documents
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a text document (.txt)")
    if uploaded_file:
        raw_text = uploaded_file.read().decode()
        # Split text so it's easier to search
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.create_documents([raw_text])
        vector_store.add_documents(chunks)
        st.success("File uploaded and indexed!")

# Chat Window
user_input = st.chat_input("Ask a question about your documents...")
if user_input:
    st.session_state.chat_history.append(HumanMessage(user_input))

    with st.spinner("Executing Local Workflow..."):
        result = graph.invoke({"messages": st.session_state.chat_history})
        final_answer = result["answer"]
        # Add the compliance warning if needed
        report = result.get("compliance_report", "")
        formatted_response = f"{final_answer}\n\n---\n**🤖 Compliance Audit:**\n{report}"
        st.session_state.chat_history.append(AIMessage(formatted_response))

# Display the conversation
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)