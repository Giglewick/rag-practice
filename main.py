from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_documents():
    documents = []

    for file in Path("data").glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        documents.append({
            "content": text,
            "source": file.name
        })

    return documents


def main():
    documents = load_documents()
    print("Dokumente geladen:", len(documents))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # list comprehension
    texts = [doc["content"] for doc in documents]
    metadatas = [{"source": doc["source"]} for doc in documents]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="chroma_db"
    )

    question = "Was ist ein Betriebssystem?"
    results = vectorstore.similarity_search(question, k=2)

    print("\nFrage:", question)
    print("\nGefundene Dokumente:")

    for number, doc in enumerate(results, start=1):
        print(f"\nTreffer {number}")
        print("Quelle:", doc.metadata["source"])
        print("Text:", doc.page_content)


if __name__ == "__main__":
    main()