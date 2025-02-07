from langchain_community.embeddings.ollama import OllamaEmbeddings

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