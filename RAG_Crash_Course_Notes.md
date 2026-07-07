# RAG (Retrieval-Augmented Generation) Crash Course Notes

*Based on Krishnaik's RAG Tutorial Transcript and GitHub Repository:*  
https://github.com/krishnaik06/RAG-Tutorials

---

# 1. What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances Large Language Models (LLMs) by retrieving relevant information from external knowledge sources before generating a response.

## Core Flow

User Query → Retrieval → Context → LLM → Response

---

# 2. Problems Solved by RAG

## Hallucination

LLMs only know information available during training.

Example:

- Model trained until Aug 1
- User asks about an event on Aug 20
- Model may generate an incorrect answer

## Private Company Knowledge

Examples:

- HR Policies
- Finance Documents
- Internal SOPs
- Product Documentation

Fine-tuning is expensive and must be repeated when data changes.

RAG allows dynamic retrieval without retraining.

---

# 3. RAG Architecture

```text
User Query
    |
    v
Query Embedding
    |
    v
Vector Search
    |
    v
Vector Database
    |
    v
Retrieved Context
    |
    v
Prompt + Context + Query
    |
    v
LLM
    |
    v
Answer
```

---

# 4. Two Main Pipelines

## Data Injection Pipeline

```text
Documents
   |
   v
Data Parsing
   |
   v
Chunking
   |
   v
Embeddings
   |
   v
Vector Database
```

## Retrieval Pipeline

```text
User Query
    |
    v
Embedding
    |
    v
Similarity Search
    |
    v
Retrieved Context
    |
    v
Prompt + Context
    |
    v
LLM
    |
    v
Response
```

---

# 5. Document Structure in LangChain

```python
from langchain_core.documents import Document

Document(
    page_content="This is RAG content",
    metadata={
        "source": "example.txt",
        "author": "Krishna"
    }
)
```

## Important Components

### page_content

Actual document text.

### metadata

Additional information:

```python
{
    "source": "policy.pdf",
    "author": "Krishna",
    "page": 5
}
```

Metadata helps in filtering retrieved results.

---

# 6. Document Loaders

## Text Loader

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
documents = loader.load()
```

## PDF Loader

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("policy.pdf")
documents = loader.load()
```

## Directory Loader

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader
)

docs = loader.load()
```

---

# 7. Chunking

Chunking divides large documents into smaller pieces.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
```

## Why Chunk?

- Fits model context limits
- Improves retrieval accuracy
- Reduces embedding cost

---

# 8. Embeddings

Embeddings convert text into numerical vectors.

Example:

```text
"What is RAG?"
```

becomes

```text
[0.25, 0.71, 0.12, ...]
```

## Open Source Embedding Example

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embedding = model.encode(
    "What is RAG?"
)

print(len(embedding))
```

Output:

```text
384
```

---

# 9. Vector Database

Popular Vector Databases:

- ChromaDB
- FAISS
- Pinecone
- Weaviate
- Qdrant

Stores embeddings and performs similarity search.

---

# 10. ChromaDB Example

```python
import chromadb

client = chromadb.PersistentClient(
    path="./vector_db"
)

collection = client.get_or_create_collection(
    "documents"
)
```

Insert data:

```python
collection.add(
    ids=["1"],
    documents=["What is RAG"],
    embeddings=[[0.12, 0.45]]
)
```

---

# 11. Retrieval Process

Convert user query to embedding:

```python
query_embedding = model.encode(
    "What is leave policy?"
)
```

Query vector database:

```python
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)
```

Retrieved documents become context.

---

# 12. Augmentation

Combine:

```text
Prompt
+
Retrieved Context
+
User Question
```

Example:

```text
Context:
Employees receive 20 annual leaves.

Question:
What is leave policy?
```

---

# 13. Generation

```python
response = llm.invoke(prompt)
```

Output:

```text
Employees are eligible for 20 annual leaves.
```

---

# 14. End-to-End RAG Example

```python
loader = PyMuPDFLoader("policy.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [d.page_content for d in chunks]
embeddings = model.encode(texts)
```

Store:

```python
collection.add(
    ids=[str(i) for i in range(len(texts))],
    documents=texts,
    embeddings=[e.tolist() for e in embeddings]
)
```

Retrieve:

```python
query_embedding = model.encode(
    "What is leave policy?"
)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)
```

---

# 15. Simple Real-World Example

Company Policy:

```text
Employees can take 20 annual leaves.
Employees receive medical insurance.
```

User asks:

```text
How many annual leaves do I get?
```

RAG Process:

```text
Question
  -> Embedding
  -> Vector Search
  -> Context Retrieval
  -> LLM
  -> Answer
```

Answer:

```text
Employees are eligible for 20 annual leaves.
```

---

# Interview Questions

## What is RAG?

Retrieval-Augmented Generation combines external knowledge retrieval with LLM generation.

## Why RAG instead of Fine-Tuning?

- Less expensive
- Faster implementation
- Easy knowledge updates
- No retraining needed

## Main Components

1. Data Loading
2. Parsing
3. Chunking
4. Embeddings
5. Vector Database
6. Retrieval
7. Augmentation
8. Generation

## What is Chunking?

Splitting large documents into smaller manageable pieces.

## What is Embedding?

Numeric vector representation of text.

## What is Vector Database?

Database optimized for storing and searching embeddings.

---

# Recommended Learning Path

1. Document Loaders
2. Document Structure
3. Chunking Strategies
4. Embeddings
5. Vector Databases
6. Retrieval Pipeline
7. LLM Integration
8. Modular RAG
9. Advanced RAG
10. Agentic RAG
