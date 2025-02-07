from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from typing import List, Tuple

from constants import PROMPT_TEMPLATE
from functions.database import load_chroma_database

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