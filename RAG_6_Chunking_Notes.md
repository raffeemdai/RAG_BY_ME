# RAG – Chunking / Splitting Notes

## 1. Two Models Used in RAG
In a RAG (Retrieval-Augmented Generation) system, we work with two models:

- **LLM (Large Language Model)** – generates the final answer.
- **Embedding Model** – converts text into vectors (numbers) so it can be stored and searched in a **vector database**.

## 2. Why We Chunk Documents
- Documents can be large, so before creating embeddings, we split them into smaller pieces called **chunks**.
- Chunking makes retrieval more accurate and efficient, because we search over small, focused pieces of text instead of one huge document.
- In **LangChain**, each chunk is represented as a `Document` object.

## 3. Chunk Overlap
- Chunks are usually not split as hard, separate blocks.
- We keep some **overlap** (repeated text) between consecutive chunks so that context/ideas at the boundary aren't lost.

## 4. 5 Chunking Strategies for RAG
*(Source: [dailydoseofds.com – 5 Chunking Strategies for RAG](https://www.dailydoseofds.com/p/5-chunking-strategies-for-rag/))*

| # | Strategy | How it works | Pros | Cons |
|---|----------|--------------|------|------|
| 1 | **Fixed-size chunking** | Split text into equal-sized pieces (by characters/words/tokens), usually with overlap | Simple, easy to implement, easy batch processing | Can cut sentences/ideas in half; important info may get split across chunks |
| 2 | **Semantic chunking** | Split text by meaning: embed sentences/paragraphs, group them together while similarity (cosine similarity) stays high; start a new chunk when similarity drops | Keeps natural flow of language, preserves complete ideas, better retrieval accuracy | Needs a similarity threshold, which varies by document |
| 3 | **Recursive chunking** | First split by natural separators (paragraphs/sections); if a chunk is still too big, split it further recursively | Keeps natural flow, respects document structure | Slightly more complex to implement, more compute |
| 4 | **Document structure-based chunking** | Use the document's own structure (headings, sections, paragraphs) as chunk boundaries | Keeps logical/structural integrity of the document | Assumes document has a clear structure; chunk sizes may vary and exceed model limits (can combine with recursive splitting) |
| 5 | **LLM-based chunking** | Ask an LLM to create semantically meaningful chunks | Most accurate/context-aware chunking | Most expensive (compute cost); limited by the LLM's context window |

**Takeaway:** No single strategy is "best" for everyone. Semantic chunking works well in many cases, but the right choice depends on your content type, embedding model, and available compute — always test.

## 5. Setup
Install the required package:
```bash
uv add langchain-text-splitters
```

## 6. Useful References
- Chunking strategies article: https://www.dailydoseofds.com/p/5-chunking-strategies-for-rag/
- LangChain document loaders notebook: https://github.com/GenAIDevelopment/agenticai/blob/main/june26/rag/rag_learning_gcp/documents_document_loaders.ipynb
- LangChain text splitters (docs): https://docs.langchain.com/oss/python/integrations/splitters
- Chunking visualizer tool (mentioned in class — refer to shared link)

## 7. Practice / Experiment
**Prompt idea to try:** Ask an LLM to generate a table of popular embedding models by provider, along with their **token limit** and **vector dimensions**.

## 8. Best Practice Tip
In real-world pipelines, **loading and splitting a document are usually combined into a single step** rather than done separately — load the file and immediately split it into chunks.

## 9. Exercise
While loading and splitting a document, attach **extra metadata** to each chunk:
- `ingested_date` → the current date
- `project` → `"learning"`

This metadata gets stored along with each chunk/vector, so later you can filter or trace results by when they were ingested or which project they belong to.

👉 Refer to the shared notebook for the hands-on implementation.

---

## 10. Code Walkthrough
*(Source notebook: [`documents_document_loaders.ipynb`](https://github.com/GenAIDevelopment/agenticai/blob/main/june26/rag/rag_learning_gcp/documents_document_loaders.ipynb))*

This section explains, step by step, the actual code used to load documents, split them into chunks, create embeddings, and store/search them in a vector store.

### 10.1 The `Document` Object
Every piece of text in LangChain is wrapped in a `Document` object. It has two parts: the text itself (`page_content`) and extra info about it (`metadata`).

```python
from langchain_core.documents import Document

# creating a document object
doc = Document(
    page_content="This is sample doc",
    metadata={
        "source": "https://directai.blog/articles/docs/1.html",
        "author": "khaja",
        "date": "04-07-2026"
    }
)
```
**Explanation:** `page_content` holds the actual text. `metadata` is a dictionary you can use to store where the text came from, who wrote it, when it was added, etc. This metadata travels with the chunk all the way into the vector store, so you can filter/trace results later.

---

### 10.2 Loading Documents
LangChain provides **loaders** for different file types. A loader reads a file and converts it into one or more `Document` objects.

#### a) Load a plain text file
```python
from langchain_community.document_loaders import TextLoader

text_loader = TextLoader(
    file_path="./data/intro.txt",
    autodetect_encoding=True
)
documents = text_loader.load()

len(documents)                # how many Document objects were created
print(documents[0].page_content)   # the actual text
documents[0].metadata          # auto-added metadata (e.g. file source)
```
**Explanation:** `TextLoader` reads a `.txt` file. `autodetect_encoding=True` avoids errors when the file isn't plain UTF‑8. `.load()` returns a list of `Document` objects — for a single text file, this is usually a list with **one** `Document`.

#### b) Load a PDF file
```python
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(file_path="./data/sample.pdf")
docs = pdf_loader.load()

len(docs)   # PyPDFLoader creates one Document PER PAGE
```
**Explanation:** `PyPDFLoader` reads a PDF and returns **one `Document` per page** (not one per file), which is why `len(docs)` usually matches the page count.

#### c) Load a CSV file
```python
from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(file_path="./data/sample.csv")
docs = csv_loader.load()

len(docs)   # CSVLoader creates one Document PER ROW
```
**Explanation:** `CSVLoader` turns **each row** of the CSV into its own `Document`.

#### d) Load an entire folder (Directory Loader)
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

dir_loader = DirectoryLoader(
    path="./data/text",
    glob="*.txt",          # pick only .txt files
    loader_cls=TextLoader   # use TextLoader for each matched file
)
docs = dir_loader.load()

len(docs)
docs[0].page_content
docs[0].metadata
```
**Explanation:** `DirectoryLoader` scans a folder, matches files using the `glob` pattern (e.g. `*.txt`), and loads each matching file using the loader you specify in `loader_cls`. Great when you have many files to ingest at once.

You can do the same for PDFs in nested folders:
```python
directory_loader = DirectoryLoader(
    path="./data/pdf",
    glob="**/*.pdf",       # ** means "search in all subfolders too"
    loader_cls=PyPDFLoader
)
documents = directory_loader.load()

len(documents)
documents[0].metadata
documents[-1].metadata     # metadata of the last document
```
**Explanation:** `**/*.pdf` recursively looks for PDFs inside the folder and all its subfolders.

---

### 10.3 Splitting Documents into Chunks

#### a) Fixed-size chunking – `CharacterTextSplitter`
```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=87,      # max characters per chunk
    chunk_overlap=0,    # no overlap between chunks
    separator=""        # split anywhere, even mid-word
)

chunks = splitter.split_documents(documents)
len(chunks)
chunks[0].metadata
print(chunks[0].page_content)
print(chunks[1].page_content)
```
**Explanation:** This is the **fixed-size chunking** strategy from our notes above. `chunk_size=87` means every chunk is at most 87 characters. With `chunk_overlap=0`, consecutive chunks don't repeat any text — so a sentence can get cut right in the middle.

Now the same splitter **with overlap**:
```python
splitter = CharacterTextSplitter(
    chunk_size=87,
    chunk_overlap=16,   # last 16 characters of a chunk repeat in the next chunk
    separator=""
)
chunks = splitter.split_documents(documents)

print(chunks[0].page_content)
print(chunks[1].page_content)
```
**Explanation:** Adding `chunk_overlap=16` makes the end of one chunk repeat at the start of the next — this helps preserve context that would otherwise be lost at the boundary.

#### b) Recursive chunking – `RecursiveCharacterTextSplitter`
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=15,
    separators=["\n\n", "\n", " ", ""]   # tries these, in order
)
chunks = splitter.split_documents(documents)

len(chunks)
print(chunks[0].page_content)
print(chunks[1].page_content)
```
**Explanation:** This is the **recursive chunking** strategy. It first tries to split on paragraph breaks (`\n\n`). If a piece is still too big, it tries line breaks (`\n`), then spaces (` `), and finally individual characters (`""`) as a last resort. This keeps sentences/paragraphs intact much better than plain fixed-size splitting.

You can also split raw strings (not just `Document` objects) with `.split_text()`:
```python
story = """Once upon a time, a quiet clockmaker named Thomas lived in a small valley...
... (full story text) ...
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["."]     # split on sentence boundaries (periods)
)
chunks = splitter.split_text(story)

len(chunks)
for chunk in chunks:
    print(chunk)
```
**Explanation:** `split_text()` works directly on a plain Python string. Here, `separators=["."]` tells it to prefer splitting at sentence endings, so chunks stay close to full sentences.

#### c) Document structure-based chunking – `MarkdownHeaderTextSplitter`
```python
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader

md_doc = TextLoader(file_path="./data/story.md", encoding="utf-8")

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
])

chunks = splitter.split_text(md_doc.load()[0].page_content)

len(chunks)
print(chunks[0].page_content)
```
**Explanation:** This is the **document structure-based chunking** strategy. Instead of counting characters, it splits the markdown file at its headings (`#`, `##`, `###`). Each chunk also gets metadata telling you which header/section it came from — useful for keeping related content together.

---

### 10.4 Load + Split in One Step
As mentioned in the notes, in real projects it's common to load and split in a **single operation** using `load_and_split()`.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(
    file_path="./data/story.txt",
    autodetect_encoding=True
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

documents = loader.load_and_split(splitter)   # load + split together

len(documents)
for document in documents:
    print(document.page_content)
    print(document.metadata)
```
**Explanation:** `load_and_split(splitter)` combines the loading step and the chunking step into one call — you pass it the splitter you want to use, and it returns the final list of chunked `Document` objects directly.

> 💡 **Exercise tie-in:** To complete the metadata exercise (`ingested_date`, `project`), you would loop over `documents` after `load_and_split()` and update each `document.metadata` dictionary, e.g.:
> ```python
> from datetime import date
> for document in documents:
>     document.metadata["ingested_date"] = str(date.today())
>     document.metadata["project"] = "learning"
> ```

---

### 10.5 Creating Embeddings
Once we have chunks, we convert each chunk's text into a vector using an **embedding model**.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-005")

result = embedding.embed_query("What is Refund ?")
len(result)        # vector dimensions
result[:10]        # first 10 numbers of the vector
```
**Explanation:** `embed_query()` converts a single piece of text (like a user's question) into a vector (a list of numbers). `len(result)` tells you the **dimensionality** of that embedding model (e.g. 768).

Trying a different embedding model:
```python
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
result = embedding.embed_query("What is Refund ?")
len(result)
```
**Explanation:** Different embedding models produce vectors of different sizes — this is why the "popular embeddings comparison table" exercise (Section 7) is useful, so you know what to expect from each provider/model.

Embedding multiple documents at once:
```python
from langchain_core.documents import Document

documents = [
    Document(page_content="Refund is a process where money is returned to a customer.",
             metadata={"source": "https://example.com/refund"}),
    Document(page_content="A refund request must be submitted within 30 days of purchase.",
             metadata={"source": "https://example.com/refund-policy"}),
    Document(page_content="Refunds are processed within 5-7 business days after approval.",
             metadata={"source": "https://example.com/processing-time"})
]

texts = [doc.page_content for doc in documents]
embeddings = embedding.embed_documents(texts)

print(len(embeddings))      # number of vectors (= number of documents)
print(len(embeddings[0]))   # dimensions of each vector
```
**Explanation:** `embed_documents()` is the batch version of `embed_query()` — it converts a **list** of texts into a **list** of vectors in one call, which is what you'd use when embedding all your chunks before storing them.

---

### 10.6 Storing & Searching Vectors (Vector Store)
The vectors need to be stored somewhere searchable — this is the **vector database**. Here, `Chroma` is used.

```python
from langchain_chroma import Chroma

hello_vs = Chroma(
    collection_name="firstvs",
    embedding_function=embedding,       # which embedding model to use
    persist_directory="./vectorstores/hello"  # where to save it on disk
)

hello_vs.add_documents(documents=documents)   # embed + store the documents
```
**Explanation:** `Chroma` is a local vector database. When you call `add_documents()`, it automatically uses the `embedding_function` you gave it to convert each `Document`'s text into a vector, and saves both the vector and the original text/metadata.

Searching (retrieval):
```python
results = hello_vs.as_retriever().invoke("money")

for result in results:
    print(result)
```
**Explanation:** `.as_retriever()` turns the vector store into a "retriever" object. Calling `.invoke("money")` embeds the query `"money"` and returns the most semantically similar chunks — this is the **retrieval** step of RAG (the "R" in RAG).

---

### 10.7 Summary of the Flow
```
Load file(s)  →  Split into chunks  →  Embed chunks  →  Store in vector DB  →  Retrieve on query
 (Loader)         (Text Splitter)      (Embedding)        (Chroma)              (Retriever)
```
