<!-- # AI-agent-learning

For this project, `ollama` needs to be installed. For this project, we will use the DeepSeek R1 model with 14b parameters. 
To install the model: `ollama run deepseek-r1:14b`.  -->

# RAG-Supported AI-Agent Template

This repository serves as a **template** for building a **Retrieval-Augmented Generation (RAG)** AI-agent. It provides a modular structure to process PDF documents, split them into manageable text chunks, assign unique IDs, store them in a vector database using embeddings, and then query a language model (via Ollama) to generate context-aware responses.

---

## 🚀 Features

- **PDF Document Processing:** Extract text from PDF documents.
- **Text Chunking:** Split documents into smaller, overlapping chunks.
- **Unique Chunk IDs:** Generate unique identifiers for each chunk.
- **Vector Database Storage:** Use Chroma for efficient storage and retrieval of embeddings.
- **RAG Pipeline:** Retrieve relevant document chunks and generate responses using a language model.
- **CLI Tools:** Single-query and interactive multi-query command-line interfaces for querying.

---

## 📦 Required Packages

Install the necessary packages by running:

```bash
pip install ollama pypdf langchain-community chromadb
```
or 
```bash 
python -m pip install -r requirements.txt
```

## 🗂️ Project Structure

```
.
├── constants.py              # Configuration constants (MODEL, CHROMA_PATH, DATA_PATH, PROMPT_TEMPLATE)
├── functions
│   ├── database.py           # Functions to manage the Chroma database operations
│   ├── embedding.py          # Embedding function setup using OllamaEmbeddings
│   ├── id_calculation.py     # Functions for generating unique chunk IDs
│   ├── pdf_processing.py     # Functions for loading and splitting PDF documents
│   └── query.py              # Functions for querying the RAG pipeline
├── main.py                   # Main script with CLI implementations (run_single_cli & run_multi_cli)
└── README.md                 # This file
```

## ⚙️ Setup and Usage
1. Clone the Repository
```bash 
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install the Required Packages
```bash 
python -m pip install -r requirements.txt
```
3. Prepare Your Data
Place your PDF documents in the directory specified by the DATA_PATH constant in constants.py (default is ./data).

4. Running the AI-Agent
Single Query CLI
To run a single query against the RAG pipeline, execute:
```bash 
python main.py "Your query here"
``` 
**Interactive Multi-Query CLI** \
To launch an interactive session:
```bash 
python main.py
``` 
Then type your queries. Type exit or bye to quit the session.

## 🔍 How It Works
Document Processing:

Loading: The agent loads PDF documents from the specified directory. 

Splitting: Documents are split into smaller, overlapping chunks for easier processing.

Unique Chunk IDs: Each text chunk is assigned a unique ID based on its source, page number, and position, preventing duplicate entries in the database.

Database Management: Processed chunks are stored in a Chroma vector database. Only new, unique chunks are added, ensuring efficient updates.

Retrieval and Generation: The system retrieves contextually similar chunks based on the user's query using vector similarity, and then combines this context with the query to generate an answer using a language model.

## 📝 Template Purpose
Note: This repository is intended as a template to help you apply the RAG (Retrieval-Augmented Generation) approach. It is designed to be customized and extended based on your specific requirements, whether you are working with different data sources or integrating with other language models.

## Note
- Also, this is a very basic template to be used and one I created to just learn how to implement RAG, so use at your own discretion. 
- Finally, I used `deepseek-r1:14b` model for this template along with the given file structure for the chorma database and the data to be fed to the AI-Agent which can all be changed in `constants.py`