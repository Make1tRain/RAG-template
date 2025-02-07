from langchain.schema.document import Document
from typing import List, Set, Tuple
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