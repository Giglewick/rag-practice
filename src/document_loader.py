from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(data_dir: str = "data"):
    documents = []

    for file in Path(data_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        documents.extend(loader.load())

    return documents