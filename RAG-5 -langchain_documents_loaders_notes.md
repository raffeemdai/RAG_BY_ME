# LangChain RAG Basics: Documents, Loaders, Splitters, Embeddings & Vector Store

Study notes based on the notebook `documents_document_loaders.ipynb` from the
[GenAIDevelopment/agenticai](https://github.com/GenAIDevelopment/agenticai) repo

https://directai.blog/2026/07/04/gen-ai-developer-classroom-notes-04-jul-2026/


(`june26/rag/rag_learning_gcp/`). Simplified explanations + the code, in the order
a RAG pipeline is actually built: **Document → Loader → Splitter → Embedding → Vector Store**.

---

## 1. What is a `Document`?

Think of a `Document` as a simple **container with two things**:

1. `page_content` — the actual text.
2. `metadata` — a dictionary of extra info *about* that text (source URL, author, page number, date, etc.)

That's it. It's just a labeled box of text. Everything else in LangChain's RAG
workflow (loaders, splitters, retrievers) works by producing or consuming `Document` objects.

> Reference: [`Document` class docs](https://reference.langchain.com/python/langchain-core/documents/base/Document)

```python
from langchain_core.documents import Document

# creating a document object manually
doc = Document(
    page_content="This is sample doc",
    metadata={
        "source": "https://directai.blog/articles/docs/1.html",
        "author": "khaja",
        "date": "04-07-2026"
    }
)
```

**Key point:** In real projects you almost never build `Document` objects by hand
like this. Instead, you **load** them from files, databases, or websites using a
**Document Loader**. Manual creation is mostly for testing or when your data
already exists as plain Python strings (see the embeddings section below).

---

## 2. What is a Document Loader?

A **Document Loader** is a helper class that:
- Reads from some source (a `.txt` file, a `.pdf`, a `.csv`, a folder, a website, a database…)
- Converts whatever it reads into one or more `Document` objects

Loaders live in the `langchain_community` package (not `langchain_core`), so
you need to install it separately:

```bash
uv add langchain_community
```

(If you're not using `uv`, the equivalent is `pip install langchain_community`.)

> Reference: [Full list of document loaders](https://docs.langchain.com/oss/python/integrations/document_loaders)

Some loaders need extra packages installed too (e.g. PDF loaders need a PDF
parsing library). LangChain will usually tell you exactly what's missing if you
try to run a loader without its dependency.

---

## 3. TextLoader — reading a `.txt` file

Simplest loader — reads one plain text file into a single `Document`.

```python
from langchain_community.document_loaders import TextLoader

text_loader = TextLoader(
    file_path="./data/intro.txt",
    autodetect_encoding=True   # avoids encoding errors on weird files
)

documents = text_loader.load()

len(documents)               # usually 1 — the whole file is one Document
print(documents[0].page_content)   # the file's text
documents[0].metadata        # e.g. {'source': './data/intro.txt'}
```

**Notice:** `.load()` always returns a **list** of `Document` objects, even if
there's only one. This consistent return type is what lets every loader plug
into the rest of the pipeline the same way.

---

## 4. PyPDFLoader — reading a `.pdf` file

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(file_path="./data/sample.pdf")
docs = pdf_loader.load()

len(docs)
```

**Key difference from TextLoader:** PDFs are usually split **one `Document` per page**,
not one `Document` for the whole file. So `len(docs)` = number of pages, and each
`Document`'s metadata typically includes the page number.

---

## 5. CSVLoader — reading a `.csv` file

```python
from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(file_path="./data/sample.csv")
docs = csv_loader.load()

len(docs)
```

**Key point:** CSVLoader creates **one `Document` per row**. Each row's columns
get turned into `key: value` text inside `page_content`, and the row number
usually ends up in `metadata`.

---

## 6. DirectoryLoader — loading many files at once

Instead of loading files one at a time, `DirectoryLoader` scans a folder and
applies a chosen loader to every matching file.

**Loading a folder of `.txt` files:**

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

dir_loader = DirectoryLoader(
    path="./data/text",
    glob="*.txt",          # only pick up .txt files
    loader_cls=TextLoader   # use TextLoader for each one
)

docs = dir_loader.load()

len(docs)
docs[0].page_content
docs[0].metadata
```

**Loading a folder of `.pdf` files (including subfolders):**

```python
directory_loader = DirectoryLoader(
    path="./data/pdf",
    glob="**/*.pdf",        # ** means "look in subfolders too"
    loader_cls=PyPDFLoader
)

documents = directory_loader.load()

len(documents)
documents[0].metadata
documents[-1].metadata     # metadata of the last Document loaded
```

**Simple way to think about it:** `DirectoryLoader` = "for each file matching
this pattern, run this loader, and combine all the results into one big list
of `Document`s."

---

## 7. Why we split documents into chunks

A single `Document` (e.g. one PDF page, or a whole `.txt` file) is often **too
big** to hand directly to an embedding model or an LLM prompt. So before
storing documents for retrieval, we break them into smaller **chunks** using a
**Text Splitter**. This is the "chunking" step of RAG.

First, load a text file normally:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="./data/text/01_code_of_conduct.txt",
    encoding="utf-8"
)
documents = loader.load()

len(documents)          # 1 (whole file)
documents[0].metadata
```

### 7a. CharacterTextSplitter — split by character count

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=87,       # max characters per chunk
    chunk_overlap=0,     # no overlap between chunks
    separator=""         # cut anywhere, ignore natural breaks
)

chunks = splitter.split_documents(documents)

len(chunks)
chunks[0].metadata          # inherits + extends original metadata
print(chunks[0].page_content)
print(chunks[1].page_content)
```

**Why `chunk_overlap`?** With `chunk_overlap=0`, a sentence that spans the
boundary between two chunks gets cut in half and loses context. Adding overlap
repeats a bit of text at the start of the next chunk so context isn't lost:

```python
splitter = CharacterTextSplitter(
    chunk_size=87,
    chunk_overlap=16,    # repeat the last 16 characters in the next chunk
    separator=""
)
chunks = splitter.split_documents(documents)

print(chunks[0].page_content)
print(chunks[1].page_content)   # notice it repeats the tail of chunk[0]
```

### 7b. RecursiveCharacterTextSplitter — split "smartly"

This is the **recommended default** splitter for most cases. Instead of
cutting at a fixed character count no matter what, it tries a **list of
separators in priority order** — first try splitting on paragraph breaks
(`\n\n`), then line breaks (`\n`), then spaces (`" "`), and only as a last
resort cut mid-word (`""`). This keeps chunks more semantically coherent.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=15,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)

len(chunks)
print(chunks[0].page_content)
print(chunks[1].page_content)
```

**You can also split raw strings directly** (not just `Document` objects) with
`.split_text()`:

```python
story = """Once upon a time, a quiet clockmaker named Thomas lived in a small
valley surrounded by foggy hills. ... """

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["."]      # try to split at sentence boundaries first
)

chunks = splitter.split_text(story)   # returns a list of strings, not Documents

len(chunks)
for chunk in chunks:
    print(chunk)
```

### 7c. MarkdownHeaderTextSplitter — split by document structure

For Markdown files, it often makes more sense to split by **headings**
(`#`, `##`, `###`) rather than by raw character count — this keeps each
section's content together and stores the heading trail in metadata.

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

### 7d. Shortcut: `load_and_split()`

Loaders have a convenience method that loads **and** splits in one step:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(file_path="./data/story.txt", autodetect_encoding=True)

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

documents = loader.load_and_split(splitter)

len(documents)
for document in documents:
    print(document.page_content)
    print(document.metadata)
```

---

## 8. Embeddings — turning text into vectors

An **embedding** is a list of numbers (a vector) that represents the
*meaning* of a piece of text. Texts with similar meaning end up with similar
vectors. This is what makes "search by meaning" (semantic search) possible.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-005")

result = embedding.embed_query("What is Refund ?")

len(result)        # the vector's dimension count
result[:10]        # peek at the first 10 numbers
```

Different embedding models produce vectors of different sizes:

```python
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
result = embedding.embed_query("What is Refund ?")
len(result)         # a different dimension count than text-embedding-005
```

**`embed_query` vs `embed_documents`:**
- `embed_query(text)` → embeds **one** string (used for the user's search question).
- `embed_documents(list_of_texts)` → embeds **many** strings at once (used for
  all your chunks before storing them).

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Refund is a process where money is returned to a customer.",
        metadata={"source": "https://example.com/refund"}
    ),
    Document(
        page_content="A refund request must be submitted within 30 days of purchase.",
        metadata={"source": "https://example.com/refund-policy"}
    ),
    Document(
        page_content="Refunds are processed within 5-7 business days after approval.",
        metadata={"source": "https://example.com/processing-time"}
    )
]

texts = [doc.page_content for doc in documents]
embeddings = embedding.embed_documents(texts)

print(len(embeddings))      # number of documents embedded
print(len(embeddings[0]))   # dimension of each embedding vector
```

---

## 9. Vector Store — storing & searching embeddings

Once text is embedded, you need somewhere to **store** those vectors so you can
later search: "which stored chunks are most similar to this new question?"
That's what a **vector store** does. Here the example uses **Chroma**, a
lightweight local vector database.

```python
from langchain_chroma import Chroma

hello_vs = Chroma(
    collection_name="firstvs",
    embedding_function=embedding,        # Chroma will call this to embed text automatically
    persist_directory="./vectorstores/hello"  # saves to disk so it survives restarts
)

# add documents — Chroma embeds them internally using embedding_function
hello_vs.add_documents(documents=documents)
```

Now search it using a **retriever** — this embeds your query, compares it
against every stored vector, and returns the most similar `Document`s:

```python
results = hello_vs.as_retriever().invoke("money")

for result in results:
    print(result)
```

Notice you searched with the word `"money"`, but the stored text talks about
"refund" — this works because the search is based on **meaning** (vector
similarity), not exact keyword matching.

---

## 10. The full picture — how it all fits together

```
 Raw files (.txt/.pdf/.csv/folder)
        │
        ▼
 Document Loader  ──► Document objects (page_content + metadata)
        │
        ▼
 Text Splitter    ──► smaller Document "chunks"
        │
        ▼
 Embedding model  ──► numeric vectors representing each chunk's meaning
        │
        ▼
 Vector Store     ──► stores chunks + vectors, supports similarity search
        │
        ▼
 Retriever.invoke(query) ──► returns the most relevant chunks for a question
```

This pipeline — **Load → Split → Embed → Store → Retrieve** — is the backbone
of every Retrieval-Augmented Generation (RAG) system. Everything after this
(passing retrieved chunks to an LLM to generate an answer) builds directly on
top of these five steps.

---

### Quick reference — packages used

| Purpose | Package |
|---|---|
| Core `Document` class | `langchain_core` |
| File/PDF/CSV/Directory loaders | `langchain_community` |
| Text splitters | `langchain_text_splitters` |
| Google embeddings | `langchain_google_genai` |
| Chroma vector store | `langchain_chroma` |

```bash
uv add langchain_community
```
