from langchain_community.llms.ollama import Ollama

from functions.document_processing import load_documents, split_documents
from functions.id_calculation import calculate_chunk_ids
from functions.database import add_to_chroma   
from functions.query import query_rag
from constants import MODEL, DATA_PATH

def initialize(): 
    # Initialize the language model
    model = Ollama(model=MODEL)

    # Load and preprocess documents
    documents = split_documents(load_documents(DATA_PATH))

    # Compute unique chunk IDs for the documents
    calculate_chunk_ids(documents)

    # Store the processed documents in the vector database
    add_to_chroma(documents)

    return model 


def run_single_cli(prompt: str) -> str:
    """
    Runs a single query against a RAG (Retrieval-Augmented Generation) pipeline.

    Steps:
    1. Initializes an Ollama model.
    2. Loads and splits documents from the specified data path.
    3. Computes unique chunk IDs for document processing.
    4. Adds processed documents to a Chroma vector database.
    5. Queries the database with the provided prompt and returns the generated answer.

    Args:
        prompt (str): The query string input.

    Returns:
        str: The generated response from the model.
    """

    model = initialize()

    # Retrieve an answer using the query
    answer = query_rag(model, prompt)
    
    print(answer)
    return answer

def run_multi_cli() -> None:
    """
    Runs an interactive CLI loop for querying a RAG pipeline.

    Steps:
    1. Initializes an Ollama model.
    2. Loads and processes documents.
    3. Computes unique chunk IDs.
    4. Stores documents in a vector database.
    5. Repeatedly prompts the user for queries and retrieves responses.
    6. Exits when the user inputs "exit" or "bye".

    Returns:
        None
    """
    model = initialize()

    while True:
        # Prompt user for input
        prompt = input("> ")

        # Exit the loop if the user types "exit" or "bye"
        if prompt.lower() in {"exit", "bye"}:
            break

        # Retrieve and print an answer
        answer = query_rag(model, prompt)
        print(answer)

