from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def create_chunks(documents, chunk_size: int = 800, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks, persist_directory: str = "chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )


def create_rag_chain(model_name: str = "llama3.2"):
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question only with the following context.

        Context:
        {context}

        Question:
        {question}
        """
    )

    llm = ChatOllama(model=model_name)

    return prompt | llm


def answer_question(vectorstore, chain, question: str, k: int = 4):
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in results])

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer.content, results