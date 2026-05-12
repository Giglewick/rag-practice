from src.document_loader import load_pdf_documents
from src.rag_pipeline import (
    create_chunks,
    create_vectorstore,
    create_rag_chain,
    answer_question,
)


def print_sources(results):
    seen_sources = set()

    for doc in results:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        if page is not None:
            source_info = f"{source}, page {page + 1}"
        else:
            source_info = source

        if source_info not in seen_sources:
            print("-", source_info)
            seen_sources.add(source_info)


def main():
    documents = load_pdf_documents()
    print("documents loaded:", len(documents))

    chunks = create_chunks(documents)
    print("chunks created:", len(chunks))

    vectorstore = create_vectorstore(chunks)
    chain = create_rag_chain()

    question = input("Your question: ")

    answer, results = answer_question(vectorstore, chain, question)

    print("\nquestion:")
    print(question)

    print("\nanswer:")
    print(answer)

    print("\norigin:")
    print_sources(results)


if __name__ == "__main__":
    main()