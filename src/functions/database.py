from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from typing import List, Set, Tuple
import os
import shutil

from functions.embedding import get_embedding_function
from functions.id_calculation import calculate_chunk_ids
from constants import MODEL, CHROMA_PATH

### DATABASE FUNCTIONS ###

def load_chroma_database() -> Chroma:
    """
    Loads or initializes the Chroma vector database.

    Returns:
        Chroma: An instance of the Chroma database.
    """
    return Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(MODEL)
    )

def get_existing_document_ids(db: Chroma) -> Set[str]:
    """
    Retrieves the set of existing document IDs from the Chroma database.

    Args:
        db (Chroma): The Chroma database instance.

    Returns:
        Set[str]: A set containing IDs of existing documents.
    """
    existing_items = db.get(include=[])  # Fetch only the IDs
    return set(existing_items["ids"])

def filter_new_chunks(chunks_with_ids: List[Document], existing_ids: Set[str]) -> List[Document]:
    """
    Filters out chunks that are already present in the database.

    Args:
        chunks_with_ids (List[Document]): List of document chunks with assigned IDs.
        existing_ids (Set[str]): Set of existing document IDs in the database.

    Returns:
        List[Document]: List of new chunks to be added to the database.
    """
    return [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

def add_to_chroma(chunks: List[Document]) -> None:
    """
    Adds new document chunks to the Chroma vector database after checking for duplicates.

    Args:
        chunks (List[Document]): List of document chunks to be added.
    """
    db = load_chroma_database()
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_ids = get_existing_document_ids(db)

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = filter_new_chunks(chunks_with_ids, existing_ids)

    if new_chunks:
        print(f"[i] Adding {len(new_chunks)} new documents...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("[i] No new documents to add.")

### DATABASE MAINTENANCE FUNCTIONS ###

def clear_database() -> None:
    """
    Deletes all stored data in the Chroma vector database.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)