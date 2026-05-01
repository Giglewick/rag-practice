from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

response = llm.invoke("Erkläre RAG (retrieval augmented generation) in einem einfachen Satz.")
print(response.content)