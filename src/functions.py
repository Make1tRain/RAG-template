# Import Pdfs
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
import os
import shutil
from typing import List, Set, Tuple

from constants import MODEL, CHROMA_PATH, DATA_PATH, PROMPT_TEMPLATE

### PDF DOCUMENT PROCESSING FUNCTIONS ###

def load_documents(file_path: str) -> List[Document]:
    """
    Loads PDF documents from the specified path.

    Args:
        file_path (str): Path to a single PDF file or a directory containing multiple PDFs.

    Returns:
        List[Document]: A list of Document objects extracted from PDFs.
    """
    loader = PyPDFDirectoryLoader(file_path)
    return loader.load()

def split_documents(
    documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 80, length_function=len
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
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function
    )
    return text_splitter.split_documents(documents)

### EMBEDDING FUNCTIONS ###

def get_embedding_function(model_name: str) -> OllamaEmbeddings:
    """
    Returns an embedding function for the specified model.

    Args:
        model_name (str): The name of the language model to use for embeddings.

    Returns:
        OllamaEmbeddings: An embedding function instance used for vector representation.
    """
    return OllamaEmbeddings(model=model_name)

### CHUNK ID CALCULATION ###

def generate_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """
    Generates a unique chunk ID based on document source, page number, and chunk index.

    Args:
        source (str): The source file path of the document.
        page (int): Page number from which the chunk originates.
        chunk_index (int): The sequential index of the chunk on the page.

    Returns:
        str: A uniquely generated chunk ID in the format "source:page:chunk_index".
    """
    return f"{source}:{page}:{chunk_index}"

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Assigns unique IDs to document chunks based on their source and position.

    Args:
        chunks (List[Document]): List of Document objects to be processed.

    Returns:
        List[Document]: Documents with assigned unique chunk IDs.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        # Increment chunk index if still on the same page, otherwise reset index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Assign the generated chunk ID
        chunk.metadata["id"] = generate_chunk_id(source, page, current_chunk_index)
        last_page_id = current_page_id

    return chunks

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

### QUERY FUNCTIONS ###

def fetch_similar_documents(query_text: str, db: Chroma, k: int = 5) -> List[Tuple[Document, float]]:
    """
    Retrieves similar documents from the Chroma database based on the input query.

    Args:
        query_text (str): The input query string.
        db (Chroma): The Chroma database instance.
        k (int, optional): The number of similar documents to retrieve. Defaults to 5.

    Returns:
        List[Tuple[Document, float]]: A list of tuples containing Document objects and similarity scores.
    """
    return db.similarity_search_with_score(query_text, k=k)

def format_query_prompt(context_text: str, query_text: str) -> str:
    """
    Formats a query prompt using the retrieved document context.

    Args:
        context_text (str): The textual context from retrieved documents.
        query_text (str): The original user query.

    Returns:
        str: A formatted prompt string ready for the language model.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(context=context_text, question=query_text)

# 1. Normalize to Cosine-Like Similarity (Lower Scores Closer to 1)
# def normalize_similarity(scores: List[float]) -> List[float]:
#     """
#     Normalizes Euclidean distances to a similarity score between 0 and 1.

#     Args:
#         scores (List[float]): List of Euclidean distances.

#     Returns:
#         List[float]: Normalized similarity scores where 1 is most relevant.
#     """
#     min_score = min(scores)
#     max_score = max(scores)

#     # Convert to a similarity score where 1 = most relevant, 0 = least relevant
#     return [(max_score - score) / (max_score - min_score) for score in scores]


# 2. Logarithmic Scaling (If Scores Are Too High)
# import numpy as np

# def log_transform_scores(scores: List[float]) -> List[float]:
#     """
#     Applies a logarithmic transformation to make large Euclidean distances easier to interpret.

#     Args:
#         scores (List[float]): List of Euclidean distances.

#     Returns:
#         List[float]: Log-transformed similarity scores.
#     """
#     return [1 / (1 + np.log(score)) for score in scores]  # Lower scores remain closer to 1


def generate_sources(results): 
    sources = [(doc.metadata.get("id", None), score) for doc, score in results]
    # for i in range(len(sources)): 
    #     sources[i][1] = normalize_score(sources[i][1])

    return sources

def query_rag(model:Ollama, query_text: str) -> str:
    """
    Queries the Chroma database using a Retrieval-Augmented Generation (RAG) approach.

    Args:
        query_text (str): The user's query input.

    Returns:
        str: The AI-generated response based on the retrieved documents.
    """
    db = load_chroma_database()
    results = fetch_similar_documents(query_text, db)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = format_query_prompt(context_text, query_text)

    print(prompt)
    
    response_text = model.invoke(prompt)

    sources = generate_sources(results)
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return response_text

### DATABASE MAINTENANCE FUNCTIONS ###

def clear_database() -> None:
    """
    Deletes all stored data in the Chroma vector database.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


# def main(): 
#     # Create the model 
#     model = Ollama(model=MODEL)

#     # Read the documents 
#     documents = split_documents(load_documents(DATA_PATH))
    
#     # Calculate the unique id's
#     calculate_chunk_ids(documents)

#     # Add the documents to the database
#     add_to_chroma(documents)

#     # Query the database
#     query_rag(model, "What is the capital of France?")