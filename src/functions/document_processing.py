from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import List
from langchain.schema.document import Document
import os

### DOCUMENT PROCESSING FUNCTIONS ###


def get_document_paths(directory_path: str, extension: str) -> List[str]:
    """
    Retrieves all file paths with the given extension from the specified directory.

    Args:
        directory_path (str): Path to the directory containing the files.
        extension (str): File extension to filter (e.g., "pdf", "csv").

    Returns:
        List[str]: A list of file paths that match the given extension.
    """
    paths = []
    for item in os.listdir(directory_path):
        if item.endswith(f".{extension}"):
            paths.append(f"{directory_path}/{item}")

    return paths


def load_pdf(directory_path: str) -> List[Document]:
    """
    Loads all PDF documents from a specified directory.

    Args:
        directory_path (str): Path to the directory containing PDF files.

    Returns:
        List[Document]: A list of Document objects extracted from PDFs.
    """
    file_paths = get_document_paths(directory_path, "pdf")
    documents: List[Document] = []

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    return documents


def load_csv(directory_path: str) -> List[Document]:
    """
    Loads all CSV documents from a specified directory.

    Args:
        directory_path (str): Path to the directory containing CSV files.

    Returns:
        List[Document]: A list of Document objects extracted from CSVs.
    """
    file_paths = get_document_paths(directory_path, "csv")
    documents: List[Document] = []

    for file_path in file_paths:
        loader = CSVLoader(file_path)
        documents.extend(loader.load())

    return documents


def load_documents(directory_path: str) -> List[Document]:
    """
    Loads all document files (PDF and CSV) from a specified directory.

    Args:
        directory_path (str): Path to the directory containing documents.

    Returns:
        List[Document]: A list of Document objects extracted from PDFs and CSVs.
    """
    documents: List[Document] = []

    # Load PDFs
    documents.extend(load_pdf(directory_path))

    # Load CSVs
    documents.extend(load_csv(directory_path))

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
