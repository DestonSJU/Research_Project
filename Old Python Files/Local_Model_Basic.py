from langchain_ollama import ChatOllama

chat = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0,
)

messages = [
    (
        "system",
        "You are a helpful assistant.",
    ),
    ("human", "What makes LangChain great for working with LLMs?"),
]
ai_msg = chat.invoke(messages)
print(ai_msg.content)

