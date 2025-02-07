import argparse
from functions import *


def parse_cli_args():
    """
    Parses command-line arguments for interacting with the script.

    Supported options:
    - `-d` or `--document`: Path to the document (PDF) or directory of PDFs.
    - `-i` or `--input`: A query string for searching within stored documents.
    - `-f` or `--function`: The function to execute (e.g., 'load', 'query', 'clear').

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="CLI interface for document processing and retrieval.")

    parser.add_argument(
        "-db", "--database",
        type=str,
        help="Path to Chroma Database."
    )

    parser.add_argument(
        "-d", "--data",
        type=str,
        help="Path to Data."
    )
    
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Prompt to be asked to AI Agent."
    )    

    parser.add_argument(
        "-rm", "--remove_db",
        type=str,
        help="Delete the databse"
    )    

    return parser.parse_args()


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

def main():
    args = parse_cli_args()

    if args.remove_db: 
        clear_database()

    if args.prompt: 
        run_single_cli(args.prompt) 
    else: 
        run_multi_cli()


if __name__ == "__main__":
    main()
