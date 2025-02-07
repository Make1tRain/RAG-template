from langchain_community.llms.ollama import Ollama

from functions.pdf_processing import load_documents, split_documents
from functions.id_calculation import calculate_chunk_ids
from functions.database import add_to_chroma   
from functions.query import query_rag
from constants import MODEL, DATA_PATH

def run_single_cli(prompt:str): 
    # Create the model 
    model = Ollama(model=MODEL)

    # Read the documents 
    documents = split_documents(load_documents(DATA_PATH))
    
    # Calculate the unique id's
    calculate_chunk_ids(documents)

    # Add the documents to the database
    add_to_chroma(documents)

    # Query the database
    answer = query_rag(model, prompt)
    print(answer)

    return answer

def run_multi_cli(): 
    # Create the model 
    model = Ollama(model=MODEL)

    # Read the documents 
    documents = split_documents(load_documents(DATA_PATH))
    
    # Calculate the unique id's
    calculate_chunk_ids(documents)

    # Add the documents to the database
    add_to_chroma(documents)

    prompt = None

    while True: 
        # Query the database
        prompt = input(">")
        if ((prompt == "exit") or (prompt == "bye")): 
            break
        answer = query_rag(model, prompt)
        print(answer)
    return answer