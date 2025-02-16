from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import List
from langchain.schema.document import Document

### DOCUMENT PROCESSING FUNCTIONS ###


def load_pdf(file_path: str) -> List[Document]:
    """
    Loads PDF documents from the specified path.

    Args:
        file_path (str): Path to a single PDF file or a directory containing multiple PDFs.

    Returns:
        List[Document]: A list of Document objects extracted from PDFs.
    """
    loader = PyPDFDirectoryLoader(file_path)
    return loader.load()


def load_csv(file_path: str) -> List[Document]:
    """
    Loads CSV documents from the specified path.

    Args:
        file_path (str): Path to a single CSV file or a directory containing multiple CSVs.

    Returns:
        List[Document]: A list of Document objects extracted from CSVs.
    """
    loader = CSVLoader(file_path=file_path)
    return loader.load()


def load_documents(file_path: str) -> List[Document]:

    documents = []
    for document in load_pdf(file_path):
        documents.append(document)

    for document in load_csv(file_path):
        documents.append(document)

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    length_function=len,
) -> List[Document]:
    """
    Splits a list of Document objects into smaller, overlapping chunks for easier processing.

    Args:
        documents (List[Document]): List of documents to be split.
        chunk_size (int, optional): The maximum number of characters in each chunk. Defaults to 800.
        chunk_overlap (int, optional): Overlapping characters between chunks for context preservation. Defaults to 80.
        length_function (function, optional): Function to calculate chunk length. Defaults to len.

    Returns:
        List[Document]: A list of smaller chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    return text_splitter.split_documents(documents)
