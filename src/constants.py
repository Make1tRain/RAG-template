MODEL = "deepseek-r1:14b"
CHROMA_PATH = "chroma"
DATA_PATH = "./data"  

PROMPT_TEMPLATE = """ 
Answer the question based on the following context: 
{context}

--- 
Answer the question based on the above context: {question}"""


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