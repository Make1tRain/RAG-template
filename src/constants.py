MODEL = "deepseek-r1:14b"
CHROMA_PATH = "chroma"
DATA_PATH = "./data"  

PROMPT_TEMPLATE = """ 
Answer the question based on the following context: 
{context}

--- 
Answer the question based on the above context: {question}"""