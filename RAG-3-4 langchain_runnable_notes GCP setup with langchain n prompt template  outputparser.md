# LangChain Notes: `Runnable`
kHAJA sir NOtes

30th june 2026 class : https://directai.blog/2026/06/30/gen-ai-developer-classroom-notes-30-jun-2026/

2nd July/2026 – https://directai.blog/2026/07/02/gen-ai-developer-classroom-notes-02-jul-2026/
## What is `Runnable`?

- `Runnable` is a **base class** in LangChain.
- Reference docs: [Runnable — LangChain Python Reference](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable)
- Any class that inherits from `Runnable` gains a standard, predictable execution interface.
- **Core idea:** anything derived from `Runnable` can be executed using the same set of methods, regardless of what it actually does internally (LLM call, parsing, retrieval, etc).

---

## The 4 Core Methods of the Runnable Interface

| Method | Purpose |
|---|---|
| `invoke()` | Execute the runnable synchronously (single input → single output) |
| `ainvoke()` | Async version of `invoke()` — execute asynchronously |
| `batch()` | Execute multiple inputs **in parallel** |
| `stream()` | Return the response incrementally as it's generated (uses `yield`, i.e., a generator) |

**Key point:** These 4 methods form a consistent contract — once you know them, you can run *any* Runnable the same way.

---

## Runnable in Practice

### Chat Models
- `ChatModels` are derived from `Runnable`.
- All chat models are derived from `BaseChatModel` (which itself is a `Runnable`).
- This is why you can call `.invoke()`, `.stream()`, etc. directly on a chat model like `ChatGoogleGenerativeAI`.

### Messages
- All message types (`HumanMessage`, `AIMessage`, `SystemMessage`, etc.) are derived from `BaseMessage`.

### LCEL (LangChain Expression Language)
- LCEL lets you **chain** Runnables together using the pipe operator `|`.
- **Rule:** every item used in an LCEL expression must itself be derived from `Runnable`.

```python
chain = llm | parser
```

- Here, both `llm` and `parser` must be `Runnable` instances.
- The chain itself becomes a new `Runnable` — so it *also* supports `.invoke()`, `.ainvoke()`, `.batch()`, and `.stream()`.

---

## Setup Note: Working with `.ipynb` in VS Code

To use Jupyter notebooks (`.ipynb`) for experimenting with LangChain code:

1. Create a file with the extension `.ipynb`
2. In the terminal, install the Jupyter kernel dependency:
   ```bash
   uv add ipykernel
   ```
3. Select the appropriate kernel (your `.venv`) inside VS Code when running notebook cells.

---

## Quick Summary

- `Runnable` = the universal interface for executable LangChain components.
- 4 methods to remember: **invoke, ainvoke, batch, stream**.
- `ChatModels` → `BaseChatModel` → `Runnable`.
- Messages → `BaseMessage`.
- LCEL chains (`a | b | c`) require every component to be a `Runnable`.

---

## How `ChatModels` relate to `Runnable` (Inheritance Chain)

```
Runnable
   ↑
BaseChatModel   (adds chat-specific logic: handling messages, roles, etc.)
   ↑
ChatGoogleGenerativeAI / ChatOpenAI / ChatAnthropic / etc.
```

- `Runnable` gives **every** LangChain component (chains, retrievers, parsers, chat models — everything) the standard `invoke/ainvoke/batch/stream` interface.
- `BaseChatModel` adds **chat-specific behavior** on top — handling a list of `BaseMessage` objects (`HumanMessage`, `AIMessage`, `SystemMessage`), managing conversation-style input/output, and generation config.
- A specific model class (e.g. `ChatGoogleGenerativeAI`) plugs in the actual API call logic — everything else it inherits for free.

**Why this matters:** because of this inheritance chain, this line works the same way no matter which chat model is used:

```python
response = llm.invoke([HumanMessage(content="What is capital of France?")])
```

Swap `ChatGoogleGenerativeAI` for `ChatOpenAI` or `ChatAnthropic` — same `.invoke()` call, same `BaseMessage` input, same interface. That consistency is the whole point of the `Runnable` abstraction: **learn one interface, it works across the entire LangChain ecosystem**.

---

## Which RAG/LCEL Components Are Runnable?

| Component | Is it a `Runnable`? |
|---|---|
| **Prompt** (`ChatPromptTemplate`, `PromptTemplate`) | ✅ Yes |
| **LLM / ChatModel** (`ChatGoogleGenerativeAI`, etc.) | ✅ Yes |
| **Output Parser** (`StrOutputParser`, `JsonOutputParser`) | ✅ Yes |
| **Response** (e.g., `AIMessage` returned after `.invoke()`) | ❌ No |

### Details

**Prompt → Runnable ✅**
`PromptTemplate` / `ChatPromptTemplate` inherit from `Runnable`, so `.invoke()` works on them and they can sit inside an LCEL chain:

```python
prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
prompt.invoke({"country": "France"})
```

**LLM / ChatModel → Runnable ✅**
`ChatGoogleGenerativeAI` → `BaseChatModel` → `Runnable`. So `.invoke()`, `.stream()`, etc. work directly on it.

**Output Parser → Runnable ✅**
Classes like `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser` derive from `Runnable` (via `BaseOutputParser` / `BaseTransformOutputParser`), which is why they can be chained with `|`.

**Response → NOT a Runnable ❌**
The response is the *output* data object you get back after `.invoke()` — not an executable component.
- For a chat model, it's typically an `AIMessage` (derived from `BaseMessage`, not `Runnable`)
- It just holds data: `.content`, `.response_metadata`, `.usage_metadata`, etc.
- You don't "invoke" a response — you just read/use its content.

```python
response = llm.invoke([HumanMessage(content="Hi")])
response.pretty_print()   # data access, not a Runnable execution method
```

### Why this matters for LCEL

```python
chain = prompt | llm | output_parser
result = chain.invoke({"country": "France"})
```

Every link in the pipe (`prompt`, `llm`, `output_parser`) **must** be a `Runnable` — that's the rule for LCEL. The final `result` is just plain data (string, dict, or message object) — not something you chain further with `|`.

**Mental model:**
> `Runnable` = things that **do work** (transform input → output)
> Non-Runnable = the **data** flowing through the pipeline (messages, strings, responses)

---
# Prompt vs Prompt Template — Runnable Point of View (LCEL / RAG)

## What is a Runnable?

In LangChain Expression Language (LCEL), a **Runnable** is any component that
implements a common interface (`.invoke()`, `.batch()`, `.stream()`,
`.ainvoke()`, etc.). This uniform interface is what lets you **chain
components together with the `|` (pipe) operator** — the output of one
becomes the input of the next, just like Unix pipes.

```
component1 | component2 | component3
```

This only works if **all three are Runnables**.

---

## Prompt vs Prompt Template

| | Plain string / f-string | `PromptTemplate` / `ChatPromptTemplate` |
|---|---|---|
| What it is | Just Python text you build yourself, e.g. `f"Answer: {question}"` | A LangChain object that holds a template with placeholders |
| Is it a Runnable? | **No** — it's a plain `str`, has no `.invoke()`, can't be piped | **Yes** — inherits from `Runnable`, has `.invoke()`, `.batch()`, `.stream()` |
| Can you use it in an LCEL chain (`\|`)? | No — you'd have to call `.format()` manually *before* the chain starts | Yes — it slots directly into a chain |
| What `.invoke()` does | N/A | Takes a dict of variables and returns a `PromptValue` (or list of messages) |

### Code example

```python
from langchain_core.prompts import PromptTemplate

# PromptTemplate IS a Runnable
prompt = PromptTemplate.from_template(
    "Answer the question using only the context.\n\nContext: {context}\n\nQuestion: {question}"
)

# because it's Runnable, .invoke() works directly:
result = prompt.invoke({
    "context": "Paris is the capital of France.",
    "question": "What is the capital of France?"
})
print(result)   # a PromptValue / formatted prompt

# a plain string has none of this — you'd have to do:
raw_prompt = f"Context: {'Paris is the capital of France.'}\nQuestion: {'What is the capital of France?'}"
# raw_prompt.invoke(...)  -> AttributeError, strings aren't Runnable
```

**Why this matters for RAG:** because `PromptTemplate` is Runnable, you can
chain it directly with a retriever, an LLM, and an output parser using `|`,
without writing manual glue code (`.format()`, then passing to the model,
then parsing). A plain string breaks that chain — you'd have to step outside
LCEL to build it.

---

## Which RAG / LCEL Components Are Runnable?

| Component | Runnable? | Notes |
|---|---|---|
| `PromptTemplate` / `ChatPromptTemplate` | Yes | `.invoke(dict)` returns a formatted prompt |
| `ChatModel` / `LLM` (e.g. `ChatOpenAI`, `ChatGoogleGenerativeAI`) | Yes | `.invoke(prompt)` returns an `AIMessage` |
| `OutputParser` (e.g. `StrOutputParser`) | Yes | `.invoke(AIMessage)` returns a parsed value (e.g. plain string) |
| Retriever (`vectorstore.as_retriever()`) | Yes | `.invoke(query_string)` returns `list[Document]` |
| `VectorStore` itself (e.g. `Chroma`) | No | Has `.similarity_search()` etc., but is not Runnable directly. Call `.as_retriever()` to get a Runnable wrapper |
| `Document` object | No | It's just a data container (`page_content` + `metadata`), not a processing step |
| `TextSplitter` (e.g. `RecursiveCharacterTextSplitter`) | No | Has `.split_documents()`, but isn't part of the invoke/pipe interface. It's a pre-processing utility used before building the chain |
| `DocumentLoader` (e.g. `TextLoader`) | No | Has `.load()`, used once up front to build your data. Not part of the runtime chain |
| Plain Python function | Yes, if wrapped | Wrap with `RunnableLambda(fn)` to make any function pipeable |
| A dict of Runnables `{"context": retriever, "question": RunnablePassthrough()}` | Yes | LCEL auto-converts a dict into a `RunnableParallel`. Each value runs and its output fills that key |
| `RunnablePassthrough()` | Yes | Passes input through unchanged. Used to forward the original question alongside retrieved context |

**Rule of thumb:** anything involved in the **request/response flow at query
time** (prompt → retriever → LLM → parser) is Runnable. Anything involved in
**one-time data preparation** (loading files, splitting text, building the
vector store) is **not** Runnable — you run those once, upfront, outside the
chain.

---

## Putting It Together — A Typical RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

retriever = vectorstore.as_retriever()          # Runnable

prompt = ChatPromptTemplate.from_template(       # Runnable
    "Answer using only this context:\n{context}\n\nQuestion: {question}"
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # Runnable

output_parser = StrOutputParser()                # Runnable

# Everything here is Runnable, so | works end to end:
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # RunnableParallel
    | prompt
    | llm
    | output_parser
)

answer = rag_chain.invoke("What is a refund policy?")
print(answer)
```

Because **every step** here is Runnable, `rag_chain` itself becomes a single
Runnable — you can `.invoke()`, `.stream()`, or `.batch()` the **whole
pipeline** as one unit. That's the entire point of LCEL: composability
through a shared interface.


---
## Embedding vs Indexing (RAG)

Two separate steps in the RAG pipeline, often confused because they happen back-to-back.

### Embedding

**What it is:** Converting text into a numerical vector (a list of numbers) that captures its semantic meaning.

- Input: a chunk of text (e.g., "Paris is the capital of France")
- Output: a vector like `[0.021, -0.384, 0.117, ...]` (e.g., 768 or 1536 dimensions depending on the model)
- Done by an **embedding model** (e.g., `GoogleGenerativeAIEmbeddings`, `OpenAIEmbeddings`, `HuggingFaceEmbeddings`)
- Purpose: text with similar meaning ends up with vectors that are numerically "close" to each other (measurable via cosine similarity, etc.)

Think of embedding as: **text → math**

```python
embedding_vector = embeddings_model.embed_query("What is the capital of France?")
```

### Indexing

**What it is:** Storing and organizing embedding vectors (usually in a vector store/database) so they can be **searched efficiently** later.

- Input: many embedding vectors (+ their original text/metadata)
- Output: a searchable data structure inside a vector database
- Done by a **vector store** (e.g., `Chroma`, `FAISS`, `Pinecone`, `Weaviate`)
- Purpose: enables fast **similarity search** — given a query vector, quickly find the "nearest" stored vectors, instead of comparing against every single one linearly

Think of indexing as: **organizing the math so it's searchable fast**

```python
vectorstore = Chroma.from_documents(documents, embedding=embeddings_model)
```

*(Note: this single line actually does both — embeds each document AND indexes it into Chroma.)*

### Side-by-Side

| | Embedding | Indexing |
|---|---|---|
| **What it does** | Converts text → vector | Stores/organizes vectors for fast search |
| **Input** | Raw text chunk | Embedding vectors (+ metadata) |
| **Output** | A single vector | A searchable structure (index) |
| **Tool used** | Embedding model | Vector database/store |
| **Analogy** | Translating a book into a "meaning code" | Building a library catalog so you can find that book fast |
| **Happens** | Once per document/chunk | Once, after embedding, to organize everything |

### In the Full RAG Flow

```
Documents 
   ↓ (split into chunks)
Text Chunks 
   ↓ (embed each chunk)      ← EMBEDDING happens here
Embedding Vectors 
   ↓ (store in vector DB)    ← INDEXING happens here
Searchable Vector Index
   ↓ (query comes in → embed query → similarity search)
Retrieved Relevant Chunks
   ↓
Passed to LLM as context → Final Answer
```

**One-line summary:** Embedding creates the meaning-vector for a piece of text; indexing is what makes a *collection* of those vectors efficiently searchable.

---

## RAG (Retrieval-Augmented Generation) — Core Concepts

RAG works in **two main phases**:

1. **Indexing Phase** — Preparing a store for our private/external data
2. **Retrieval + Generation Phase** — Searching the private data and passing the results to the LLM

```
PHASE 1: Indexing                    PHASE 2: Retrieval + Generation
─────────────────────                ────────────────────────────────
Load documents                       User asks a question
   ↓                                    ↓
Split into chunks                    Embed the question
   ↓                                    ↓
Embed each chunk                     Search vector DB for similar chunks
   ↓                                    ↓
Store in vector database             Retrieve top-matching chunks
                                         ↓
                                      Pass chunks + question to LLM
                                         ↓
                                      LLM generates final answer
```

**Why "private data"?** LLMs are trained on public/general data and don't know about your specific documents (company docs, PDFs, internal wikis, etc.). RAG is how you "teach" the LLM about that data **at query time**, without retraining it — by retrieving relevant snippets and feeding them into the prompt as context.

---

## Phase 1: Indexing Phase (Preparing the Store)

This is the **setup/preparation** phase — done once (or whenever your source data changes), *before* any user asks a question.

**Goal:** Take your raw private data and convert it into a searchable format inside a vector database.

**Steps involved:** Load → Chunk → Embed → Store *(each explained in the "Terms" section below)*

---

## Vector Databases

**What they are:** Databases specifically designed to store data in **vector form** (i.e., as embeddings — lists of numbers representing meaning).

**Why we use them:** Normal databases search by exact match or keyword (`WHERE name = 'Paris'`). Vector databases search by **similarity** — "find me the chunks whose *meaning* is closest to this query," even if no exact words match.

This is what makes semantic search possible: a query like *"What's the money-back policy?"* can retrieve a chunk that says *"Refunds are processed within 30 days"* — even though no words overlap.

### Popular Vector Databases

| Database | Notes |
|---|---|
| **FAISS** | Facebook AI Similarity Search — fast, open-source, runs locally, no server needed |
| **ChromaDB** | Open-source, lightweight, popular for prototyping and small-to-medium RAG apps |
| **Weaviate** | Open-source, supports hybrid search (keyword + vector), can self-host or use cloud |
| **Pinecone** | Fully managed cloud vector DB, popular for production-scale apps |
| **pgvector** | A PostgreSQL extension — adds vector search capability to a regular Postgres DB |
| **Cloud provider vector DBs** | e.g., AWS (OpenSearch/Bedrock), Google (Vertex AI Vector Search), Azure (AI Search) — managed options tied to a cloud ecosystem |

---

## Key Terms in RAG

### 1. Chunking
**What it is:** Splitting large documents into smaller, manageable pieces ("chunks") before embedding.

**Why:** 
- LLMs and embedding models have limited context windows — you can't embed/process an entire 200-page PDF as one unit.
- Smaller, focused chunks lead to more precise retrieval (you get the specific relevant paragraph, not an entire document).

**How:** Usually done by size (e.g., 500 characters/tokens per chunk) with some **overlap** between chunks so context isn't lost at chunk boundaries.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

### 2. Embedding
**What it is:** Converting a chunk of text into a numerical vector that captures its semantic meaning.

*(Covered in detail earlier — text → vector, done by an embedding model, similar meaning → numerically close vectors.)*

### 3. Retrievers
**What it is:** A component whose job is to **fetch the most relevant chunks** from the vector database, given a query.

- A `Retriever` in LangChain is itself a `Runnable` — you can call `.invoke(query)` on it.
- Internally, it typically: embeds the incoming query → performs a similarity search against the vector store → returns the top-k matching chunks.
- Retrievers can be simple (plain similarity search) or advanced (e.g., adding re-ranking, filtering by metadata, hybrid keyword+vector search, MMR for diversity).

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke("What is the refund policy?")
```

### 4. Document Loaders
**What it is:** Components responsible for **loading raw data** from various sources into a standard `Document` object that LangChain can work with.

- Different sources need different loaders: PDF, Word, CSV, web pages, databases, Notion, Slack, etc.
- Each loader handles the source-specific parsing so you get consistent `Document` objects out, regardless of where the data came from.

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("company_policy.pdf")
documents = loader.load()
```

### 5. Document Types
**What it is:** Refers to the different **formats/sources** of data that can feed into a RAG pipeline, each typically needing its own loader.

Common document types:
- PDFs
- Word documents (.docx)
- Plain text / Markdown files
- CSV / Excel spreadsheets
- Web pages (HTML)
- Databases (SQL)
- APIs / JSON data
- Notion pages, Slack messages, Confluence, etc.

Regardless of the original type, once loaded, everything becomes a standard **`Document` object** in LangChain — with `.page_content` (the text) and `.metadata` (source info, page number, etc.) — so the rest of the pipeline (chunking, embedding, indexing) works the same way no matter the original format.

---

## Putting It All Together (Indexing Phase Pipeline)

```
1. Document Loaders  → Load raw files (PDF, DOCX, CSV, etc.) into Document objects
        ↓
2. Chunking          → Split large Documents into smaller overlapping chunks
        ↓
3. Embedding         → Convert each chunk into a vector (embedding model)
        ↓
4. Vector Database    → Store vectors + original text + metadata (FAISS/Chroma/Pinecone/etc.)
```

Then in the **query phase**, a **Retriever** wraps the vector database to fetch relevant chunks for a given user question, which are then passed to the LLM as context to generate the final answer.

---

## Setting Up GCP with LangChain (Gemini via Vertex AI)

Instead of using a direct Google AI Studio API key, this setup connects LangChain to Gemini models **through Google Cloud Platform (GCP) / Vertex AI**, which is the enterprise-grade route (useful for billing, IAM, and org-level control).

### Steps

1. **Install the Google Cloud SDK (`gcloud`)** — needed to authenticate your machine with GCP.
2. **Open the GCP Console**, note your **Project ID**, and navigate to the *Agent Platform* overview.
3. **Enable required APIs** if GCP prompts you to (e.g., Vertex AI / Generative AI APIs).
4. **Authenticate from the terminal:**
   ```bash
   gcloud init
   gcloud auth application-default login
   ```
   - `gcloud init` — sets up/selects your GCP project and default config.
   - `gcloud auth application-default login` — creates local "Application Default Credentials" so SDKs (like LangChain) can authenticate as you, without hardcoding keys.

5. **Set up the project folder with `uv`:**
   ```bash
   mkdir rag_learning_gcp
   cd rag_learning_gcp
   uv init .
   uv add langchain python-dotenv langchain-google-genai ipykernel
   ```

6. **Select the `rag_learning_gcp` interpreter** in VS Code (`Ctrl+Shift+P` → Python: Select Interpreter).

7. **Create a `.env` file** with your GCP project details:
   ```
   GOOGLE_CLOUD_PROJECT=''
   GOOGLE_CLOUD_LOCATION='us-central1'
   GOOGLE_GENAI_USE_VERTEXAI=true
   ```
   - `GOOGLE_CLOUD_PROJECT` — your GCP project ID (fill this in).
   - `GOOGLE_CLOUD_LOCATION` — the region for Vertex AI calls (e.g., `us-central1`).
   - `GOOGLE_GENAI_USE_VERTEXAI=true` — tells the `langchain-google-genai` library to route calls through **Vertex AI** instead of the public Gemini API — this is what lets `ChatGoogleGenerativeAI` authenticate using your `gcloud` login instead of an API key.

### Example project structure (from the reference commit)

```
rag_learning_gcp/
├── .python-version                              # pins Python 3.13
├── pyproject.toml                               # deps: langchain, langchain-google-genai, python-dotenv, ipykernel
├── main.py                                      # entry point script
├── utils.py                                     # helper to build the LLM client
└── prompts_models_structured_output.ipynb       # notebook with prompt/output-parser examples
```

**`utils.py`** — a reusable helper function to get the chat model:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_model_from_gcp(model_name: str = "gemini-2.5-flash-lite") -> ChatGoogleGenerativeAI:
    """Returns a chat model configured to use GCP/Vertex AI."""
    return ChatGoogleGenerativeAI(model=model_name)
```

**`main.py`** — using that helper:
```python
from utils import get_model_from_gcp

def main():
    llm = get_model_from_gcp()
    result = llm.invoke("What is 2+2?")
    result.pretty_print()

if __name__ == "__main__":
    main()
```

Keeping model-creation logic in `utils.py` (instead of repeating it everywhere) is a good practice — one place to change the model name, add config, or swap providers.

Reference: [commit with these changes](https://github.com/GenAIDevelopment/agenticai/commit/79c501681a2b4b5378ba0f5eaf5a3e606df96c52)

---

## Prompt Templating

**Why templates instead of plain f-strings?** You *could* build a prompt manually with Python f-strings (`f"What is capital of {country}"`), and that works — but templates give you a **reusable, LCEL-composable, provider-agnostic** object that plugs directly into a chain, supports validation, and (in tools like LangSmith) can be versioned/shared.

```python
# Manual f-string approach (works, but not a Runnable / not reusable in a chain)
country = input("Enter any country name: ")
prompt = f"What is capital of {country}"
response = llm.invoke(prompt)
```

### `PromptTemplate` — plain text prompts

Used for simple, single-block prompts (no distinct roles):

```python
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    template="What is capital of {country}? and Also show their achievements in the following sport {sport}"
)

chain = prompt_template | llm   # prompt_template is a Runnable, so this LCEL chain works

response = chain.invoke({"country": "India", "sport": "Hockey"})
```

- `{country}` and `{sport}` are **placeholders** filled in at `.invoke()` time.
- Because `PromptTemplate` is a `Runnable`, it can be piped (`|`) directly into an LLM (also a `Runnable`) to form a chain.

### `ChatPromptTemplate` — role-based prompts

Chat models expect a **list of messages with roles** (`system`, `user`, `assistant`), not just one block of text. `ChatPromptTemplate` is built for that:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}."),
    ("user", "What is significance of {topic} for {audience}")
])

prompt.invoke({"topic": "Sports", "audience": "students"})
```

- Each tuple is `(role, message_template)`.
- `system` — sets the model's behavior/persona for the whole conversation.
- `user` — the human's message.
- `assistant` — can also be included to show prior AI turns (useful for few-shot or continuing a conversation).
- Placeholders (`{topic}`, `{audience}`) work the same way as `PromptTemplate`, just embedded inside each role's message.

Chaining it with the LLM:
```python
chain = prompt | llm
response = chain.invoke({"topic": "Sports", "audience": "students"})
print(response.content)
```

### Template Syntax Reference (LangSmith / LangChain)

LangChain/LangSmith support **two template formats**:

| Format | Syntax | Best for |
|---|---|---|
| **f-string** | `{variable}` | Simple, flat variable substitution |
| **mustache** | `{{variable}}` | Complex data — loops, conditionals, nested objects |

**f-string** (what's used above):
- Simple lookup: `{name}` → replaced with the value of `name`.
- **Limitations:** no dot notation (`{user.name}` won't work — the whole string is treated as one variable name), no expressions, no loops/conditionals, no array indexing.

**mustache** (used for more advanced cases, e.g. LangSmith evaluators):
- Nested access: `{{user.name}}` traverses nested objects.
- **Sections** `{{#items}}...{{/items}}` — loop over arrays, or render conditionally if a value is truthy.
- **Inverted sections** `{{^items}}...{{/items}}` — render only when a value is empty/missing (handy for "no results found" fallback text).
- Supports comments: `{{! this won't render }}`.
- Useful for few-shot prompting (looping over example lists) and for evaluators that need structured access to conversation threads (`all_messages`, `human_ai_pairs`, `first_human_last_ai`).

**Rule of thumb:** stick with f-string style (`{variable}`) for straightforward LangChain prompts like the ones above; reach for mustache only when you need loops/conditionals/nested data (mostly relevant in LangSmith's Playground/evaluators, less so in plain LangChain Python code).

Reference: [Prompt template format guide](https://docs.langchain.com/langsmith/prompt-template-format)

---

## Output Parsers & Structured Output

**The problem:** By default, an LLM's response is a `BaseMessage` (e.g., `AIMessage`) — free-form text, not guaranteed to follow any particular structure. Often you need the output in a **specific, predictable shape** (plain string, JSON, a Python object, a list, etc.) so your downstream code can use it reliably.

**Output Parsers** are `Runnable` components that sit at the *end* of a chain and transform the raw LLM output into that desired structure.

### Common Output Parser classes (from `langchain_core.output_parsers`)

| Class | Purpose |
|---|---|
| `StrOutputParser` | Extracts just the plain string content from the LLM response |
| `JsonOutputParser` | Parses the LLM output into a JSON object |
| `XMLOutputParser` | Parses output using XML format |
| `CommaSeparatedListOutputParser` | Parses a comma-separated string into a Python list |
| `PydanticToolsParser` / `JsonOutputKeyToolsParser` | Parse structured "tool call" style output (e.g., from OpenAI function/tool calling) |
| `BaseOutputParser` | Base class — all custom output parsers inherit from this |
| `BaseLLMOutputParser` | Abstract base class for parsing model outputs generally |

Reference: [Output parsers API reference](https://reference.langchain.com/python/langchain-core/output-parsers)

### Example: `StrOutputParser`

Without a parser, `chain.invoke(...)` returns a full `AIMessage` object (you'd need `.content` to get the text). `StrOutputParser` does that extraction for you, as part of the chain:

```python
from langchain_core.output_parsers import StrOutputParser

output = StrOutputParser()
chain = prompt | llm | output   # every link must be a Runnable — prompt, llm, and output all are

response = chain.invoke({"topic": "Sports", "audience": "students"})
print(response)   # response is now a plain string, not an AIMessage object
```

### Structured Output with Pydantic — `.with_structured_output()`

For more complex, guaranteed structure (e.g., specific fields with specific types), LangChain lets you define a **Pydantic model** describing the exact shape you want, and pass it to `.with_structured_output()` on the LLM. The model is instructed (under the hood) to return data matching that schema — no manual parsing needed.

```python
from pydantic import BaseModel, Field

class SupportTicket(BaseModel):
    issue_category: str = Field(description="One of: billing, technical, shipping, other")
    resolved: bool = Field(description="Whether the issue appears resolved based on the conversation")
    next_action: str = Field(description="What should happen next, one short sentence")

prompt = ChatPromptTemplate.from_messages([
    ("system", "you summarize customer support conversations into structured tickets."),
    ("user", "My last order never arrived and it's been 2 weeks"),
    ("assistant", "I'm sorry to hear that, I have issued a reshipment, tracking number sent to your registered email and phone number"),
    ("user", "Got it thank you")
])

chain = prompt | llm.with_structured_output(SupportTicket, method='json_schema')
response = chain.invoke({})
response   # returns a SupportTicket object with issue_category, resolved, next_action populated
```

**Breaking this down:**
- Each `Field(description=...)` tells the LLM (via the generated JSON schema) exactly what that field means — this description is what guides the model to fill it in correctly.
- `method='json_schema'` tells LangChain to use JSON-schema-based structured output generation (as opposed to, e.g., tool-calling-based extraction).
- The `prompt` here embeds a **mock conversation** (system + user + assistant + user turns) — this gives the LLM the full context of a support conversation to summarize into the `SupportTicket` structure.
- The result (`response`) is a validated Python object (a `SupportTicket` instance) — not a raw string — so you can safely access `response.issue_category`, `response.resolved`, etc. in your code without additional parsing.

### Why this matters

| Approach | What you get | When to use |
|---|---|---|
| No parser | `AIMessage` (raw response, `.content` = string) | Quick experiments, don't need structure |
| `StrOutputParser` | Plain string | You just want the text, not the full message object |
| `JsonOutputParser` | Python `dict` | You need JSON but don't need strict schema validation |
| `.with_structured_output(PydanticModel)` | Validated Pydantic object | You need guaranteed, typed, validated fields — ideal for feeding results into downstream code/APIs/databases |

Reference: [notebook with full examples](https://github.com/GenAIDevelopment/agenticai/blob/main/june26/rag/rag_learning_gcp/prompts_models_structured_output.ipynb)

---

## `uv` Command Reference (for this project)

### Project Setup

| Command | Description |
|---|---|
| `uv init` | Create a new project in current directory |
| `uv init myproject` | Create a new project in a new folder |
| `uv init --lib` | Initialize as a library (with `src/` layout) |
| `uv init --app` | Initialize as an application (default) |
| `uv init --python 3.12` | Initialize with a specific Python version |

### Dependency Management

| Command | Description |
|---|---|
| `uv add <package>` | Add a package as a dependency |
| `uv add <package>==1.2.3` | Add a specific version |
| `uv add "<package>>=1.0,<2.0"` | Add with version constraints |
| `uv add --dev <package>` | Add a dev-only dependency |
| `uv add --optional <group> <package>` | Add to an optional dependency group |
| `uv add -r requirements.txt` | Add all packages from a requirements file |
| `uv remove <package>` | Remove a dependency |
| `uv sync` | Install/sync environment to match lockfile |
| `uv sync --upgrade` | Sync and upgrade all packages |
| `uv lock` | Generate/update the lockfile without installing |
| `uv lock --upgrade` | Upgrade all pinned versions in lockfile |
| `uv lock --upgrade-package <pkg>` | Upgrade a single package in lockfile |
| `uv tree` | Show dependency tree |

### Running Code

| Command | Description |
|---|---|
| `uv run main.py` | Run a script in the project's environment |
| `uv run python` | Open a Python REPL in the environment |
| `uv run -- pytest` | Run a command/tool inside the environment |
| `uv run --with <package> script.py` | Run with a temporary extra package (not saved to project) |
| `uv run --env-file .env main.py` | Run with a specific env file explicitly loaded |

### Python Version Management

| Command | Description |
|---|---|
| `uv python list` | List installed & available Python versions |
| `uv python install 3.12` | Install a specific Python version |
| `uv python pin 3.12` | Pin project to a specific Python version |
| `uv python find` | Show which Python uv would use |
| `uv python uninstall 3.11` | Remove a Python version |

### Virtual Environments

| Command | Description |
|---|---|
| `uv venv` | Create a `.venv` virtual environment |
| `uv venv --python 3.12` | Create venv with specific Python version |
| `uv venv myenv` | Create venv with a custom name/path |

### Tools (like `pipx`)

| Command | Description |
|---|---|
| `uv tool install <tool>` | Install a CLI tool globally (isolated env) |
| `uv tool run <tool>` | Run a tool without installing (like `npx`) |
| `uvx <tool>` | Shorthand for `uv tool run` |
| `uv tool list` | List installed tools |
| `uv tool uninstall <tool>` | Remove a tool |
| `uv tool upgrade <tool>` | Upgrade a tool |

### Pip-Compatible Interface

| Command | Description |
|---|---|
| `uv pip install <package>` | Install a package (pip-style, no project tracking) |
| `uv pip install -r requirements.txt` | Install from requirements file |
| `uv pip freeze` | List installed packages |
| `uv pip list` | List installed packages (formatted) |
| `uv pip uninstall <package>` | Uninstall a package |
| `uv pip compile requirements.in -o requirements.txt` | Compile a locked requirements file |

### Build & Publish

| Command | Description |
|---|---|
| `uv build` | Build a package (wheel + sdist) |
| `uv publish` | Publish package to PyPI |

### Cache & Maintenance

| Command | Description |
|---|---|
| `uv cache clean` | Clear uv's cache |
| `uv cache dir` | Show cache directory location |
| `uv self update` | Update uv itself |
| `uv version` | Show uv version |
