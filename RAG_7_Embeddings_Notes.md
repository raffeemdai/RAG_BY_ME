# RAG Embeddings — Simple Notes
*(Embeddings · Vector Databases · Indexing Pipeline)*

https://directai.blog/2026/07/07/gen-ai-developer-classroom-notes-07-jul-2026/

https://www.pinecone.io/learn/vector-database/



---

## 1. What is an Embedding? (in one line)

> An embedding is **meaning turned into a vector** — a point in high-dimensional space, so that "similar meaning" = "nearby points."

- A sentence, word, or document goes in → a list of numbers comes out (e.g. `[0.02, -0.13, 0.44, ...]`, often 384–3072 numbers long).
- Words/text used in similar contexts end up close together (measured using **cosine similarity**, **dot product**, or **Euclidean distance**).
- Classic example: `king - man + woman ≈ queen` — meaning can be manipulated with simple vector arithmetic.
- An embedding *model* knows the whole vocabulary it was trained on and maps every word/token into this space.

---

## 2. Evolution of Embeddings (simple timeline)

| Year | Milestone | Why it mattered |
|------|-----------|------------------|
| **2013** | **Word2Vec** (Mikolov et al., Google) — CBOW & Skip-gram | First practical, efficient way to learn dense word vectors from raw text; captured semantic + syntactic analogies |
| **2014** | **GloVe** (Stanford) | Used global word co-occurrence statistics instead of just local context; positioned as a Word2Vec competitor |
| **2016** | **FastText** (Facebook/Meta) | Represented words as bags of character n-grams → handled misspellings & out-of-vocabulary words |
| **2018** | **ELMo** | Introduced *contextual* embeddings — same word gets a different vector depending on sentence context |
| **2018** | **BERT** (Transformer-based) | Deep bidirectional attention became the new state of the art for contextual embeddings |
| **2019–2022** | **Sentence-BERT / Universal Sentence Encoder** | Made whole-sentence and document embeddings practical (not just words) |
| **2022–2023** | **OpenAI `text-embedding-ada-002`, Cohere Embed, Google `text-embedding-gecko/005`** | Embeddings offered as a cloud API/service — no need to host your own model |
| **2023–2026** | **Long-context, multilingual & multimodal embeddings** (e.g. `text-embedding-3-large`, Voyage-3, Gemini embeddings) | Same vector space now spans text, and increasingly images/audio; longer inputs, better retrieval accuracy |

*(See the interactive timeline below for a visual view.)*

---

## 3. Embedding Models — Two Ways to Get One

| Type | Examples | Notes |
|------|----------|-------|
| **Open-source (self-hosted)** | Sentence-Transformers (`all-mpnet-base-v2`), Nomic Embed, BGE | Free, run locally/on your own servers, full data privacy, but you manage the compute |
| **Cloud / Embedding-as-a-Service** | OpenAI (`text-embedding-3-large`), Google Vertex (`text-embedding-005`), Cohere (`embed-english-v3.0`), AWS Titan, Mistral, Voyage AI | Pay-per-call API, no infra to manage, usually strong quality, but data leaves your environment |

### LangChain's Embedding Interface
LangChain defines a base class `Embeddings` (in `langchain_core.embeddings`) with just **two methods**:

```python
embeddings.embed_query("a single search query")      # -> one vector
embeddings.embed_documents(["doc1", "doc2", "doc3"])  # -> list of vectors
```

Every provider (OpenAI, Google, Cohere, HuggingFace, Ollama, etc.) implements this same interface — so you can swap providers with one line of code.

---

## 4. Vector Database vs Relational Database

| Aspect | Relational Database (SQL) | Vector Database |
|---|---|---|
| **What it stores** | Rows & columns of scalar data (text, numbers, dates) | High-dimensional vectors (embeddings) + metadata |
| **Query type** | Exact match / range queries (`WHERE age > 30`) | **Similarity search** — "find the *k* nearest vectors" (Approximate Nearest Neighbor / ANN) |
| **Matching logic** | Boolean logic, joins, exact equality | Distance/similarity metrics: cosine similarity, dot product, Euclidean distance |
| **Index type** | B-Tree, Hash index | HNSW, IVF, Product Quantization (PQ), LSH |
| **Schema** | Rigid, predefined schema | Flexible — vector + free-form metadata (JSON-like filters) |
| **Typical operations** | `SELECT, INSERT, UPDATE, DELETE, JOIN` | `upsert (add/update vectors), delete, similarity_search, filter by metadata` |
| **Consistency** | Strong (ACID) | Often eventual consistency (trades perfect accuracy for speed) at large scale |
| **Result correctness** | Exact | Approximate (accuracy vs. speed tradeoff by design) |
| **Best for** | Structured business data, transactions | Semantic search, RAG, recommendations, image/audio similarity |

**Key idea:** a vector DB does everything a relational DB does operationally (CRUD, filtering, backups, access control, scaling) — but its *core query* is "what is closest to this point?" instead of "what exactly matches this condition?"

---

## 5. Organization & Operations Inside a Vector Database
*(Deep dive — based on Pinecone's "What is a Vector Database" article)*

**How data is organized:**
1. **Vector index** — the actual embeddings, organized via an ANN algorithm so nearest-neighbor search is fast instead of scanning every vector.
2. **Metadata index** — a normal structured index (like a mini relational table) attached to each vector, e.g. `{city: "Paris", chunk_id: 3}`.
3. **Sharding** — data split across nodes/partitions (often by similarity clusters) for scale; queries fan out to all shards and results are merged ("**scatter-gather**" pattern).
4. **Replication** — copies of data across nodes for fault tolerance:
   - **Eventual consistency** — faster, more available, but temporary mismatches possible between replicas.
   - **Strong consistency** — all replicas updated before a write completes; safer but higher latency.

### 5.1 Vector index vs. full vector database
A standalone index like **FAISS** speeds up similarity search but isn't a database. A vector database adds what FAISS lacks:

| Capability | Why it matters |
|---|---|
| Data management (insert/update/delete) | FAISS needs custom glue code to persist and mutate data |
| Metadata storage & filtering | Query using both semantic similarity *and* structured filters |
| Scalability | Built-in distributed/parallel scaling vs. custom Kubernetes-style setups |
| Real-time updates | New data becomes queryable quickly; FAISS often needs a full re-index |
| Backups & collections | Routine backups; snapshot an index as a reusable "collection" |
| Ecosystem integration | Plugs into ETL (Spark), analytics (Tableau, Grafana), and AI tooling (LangChain, LlamaIndex) |
| Security & access control | Multitenancy via namespaces, built-in access rules |

### 5.2 The 3-stage query pipeline
1. **Indexing** — vectors are mapped into a data structure (via PQ, LSH, HNSW, etc.) built for fast search.
2. **Querying** — the query vector is compared to indexed vectors using a similarity metric to find nearest neighbors.
3. **Post-processing** *(optional)* — results may be re-ranked with a different similarity measure before being returned.

Because exhaustive/exact search doesn't scale, vector databases use **Approximate Nearest Neighbor (ANN)** search — trading a little accuracy for a lot of speed.

### 5.3 Indexing algorithms

| Algorithm | How it works | Trade-off |
|---|---|---|
| **Random Projection** | Multiply vectors by a random projection matrix to reduce dimensions while preserving similarity | Faster search, but quality depends on how random the matrix is (expensive to generate well) |
| **Product Quantization (PQ)** | *Lossy* compression: split each vector into chunks → build a "codebook" per chunk (k-means) → encode each chunk as a codebook index | More codebook entries = better accuracy but higher search cost, and vice versa |
| **Locality-Sensitive Hashing (LSH)** | Hash functions map similar vectors into the same "buckets"; only compare within the query's bucket | Very fast; more hash functions = better accuracy but higher compute |
| **HNSW** (Hierarchical Navigable Small World) | Graph-based structure — nodes are vector clusters, edges connect similar clusters; query navigates the graph toward nearest vectors | Popular default; strong speed/accuracy balance |

### 5.4 Similarity measures

| Measure | Range | Meaning |
|---|---|---|
| **Cosine similarity** | -1 to 1 | 1 = identical direction, 0 = orthogonal, -1 = opposite |
| **Euclidean distance** | 0 to ∞ | 0 = identical vectors; straight-line distance |
| **Dot product** | -∞ to ∞ | Positive = same direction, 0 = orthogonal, negative = opposite |

### 5.5 Metadata filtering
Vector DBs usually keep **two indexes**: a vector index and a metadata index. Filtering can happen:

| Approach | Pros | Cons |
|---|---|---|
| **Pre-filtering** (before similarity search) | Shrinks the search space upfront | May miss relevant vectors just outside the filter; heavy filters slow the query |
| **Post-filtering** (after similarity search) | Full vector search runs uninterrupted | Extra overhead discarding irrelevant results afterward |

### 5.6 Production operations
| Area | What it covers |
|---|---|
| **Performance & fault tolerance** | Sharding (scatter-gather) + replication (eventual vs strong consistency) |
| **Monitoring** | Resource usage (CPU/memory/disk/network), query latency/throughput/error rate, node & replication health |
| **Access control** | Data protection, regulatory compliance, auditing/accountability, scaling permissions as teams grow |
| **Backups & collections** | Regular backups; Pinecone lets you snapshot an index as a "collection" to repopulate later |
| **API & SDKs** | A simple API layer + language SDKs so developers focus on the use case, not infrastructure |

### 5.7 Serverless vector databases — the next generation
First-generation vector DBs are accurate, fast, scalable — but **expensive**, because storage and compute are tightly coupled. Serverless architecture solves three pain points:

| Pain point | Serverless solution |
|---|---|
| **Separation of storage & compute** | Geometric partitioning splits the index into sub-indices (Voronoi-style regions) so a query searches only relevant partitions |
| **Freshness** | A temporary **"freshness layer"** caches new vectors while they wait to be placed into the (slower-to-build) partitioned index; a query router checks both the freshness layer and the main index |
| **Multitenancy** | Users with very different query loads (e.g. 20/sec vs 20/month) get intelligently co-located on shared hardware, balancing cost and latency |

### 5.8 Operations you can perform (quick reference)
| Operation | Description |
|---|---|
| `upsert` / `add_documents` | Insert or update vectors + metadata |
| `delete` | Remove vectors by ID |
| `similarity_search(query, k, filter)` | Return top-k nearest vectors, optionally filtered by metadata |
| `search_by_vector` | Search directly using a vector (skip re-embedding) |
| **Metadata filtering** | Narrow results using structured filters (pre- or post-filter) |
| Backups / collections | Snapshot an index for reuse or recovery |
| Access control | Restrict who can read/write which vectors/namespaces |

---

## 6. Vector Database Options (cheat sheet)

| Category | Examples |
|---|---|
| **Developer / local** | Chroma, FAISS (a library, not a full DB) |
| **Familiar tech, vector-enabled** | pgvector (Postgres extension), MongoDB Atlas Vector Search |
| **Cloud provider native** | AWS (OpenSearch/Bedrock), Azure (Cosmos DB), GCP (Vertex AI Vector Search) |
| **SaaS / managed** | Pinecone |
| **Open-source, standalone** | Weaviate, Qdrant, Milvus |

**LangChain** provides one unified interface (`add_documents`, `delete`, `similarity_search`) across **all** of the above — so switching vector stores mostly means swapping one import line.

---

## 7. Exercise Recap — Simple Indexing Pipeline

**Goal:** Build a small RAG index of city descriptions.

```
Step 1 — Prepare data
   /cities/
     paris.txt      (≥200 lines, in paragraphs)
     tokyo.txt
     ...

Step 2 — Split
   RecursiveCharacterTextSplitter → chunks ("documents")

Step 3 — Embed
   embedding model: text-embedding-005 (Google Vertex)

Step 4 — Store
   Chroma vector store, persisted at ./data/example1

Step 5 — Metadata per chunk
   { "city": "<city name>", "chunk_id": <int> }
```

### Minimal code sketch
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embeddings = VertexAIEmbeddings(model="text-embedding-005")

docs = []
for filename in os.listdir("cities"):
    city = filename.replace(".txt", "")
    text = open(f"cities/{filename}").read()
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        docs.append(Document(page_content=chunk, metadata={"city": city, "chunk_id": i}))

vector_store = Chroma(
    collection_name="cities",
    embedding_function=embeddings,
    persist_directory="./data/example1",
)
vector_store.add_documents(docs)

# Query example
results = vector_store.similarity_search("famous river in Paris", k=3, filter={"city": "paris"})
```

---

## 8. One-Line Summary
**Embeddings turn text into meaning-carrying vectors → Vector databases store & search those vectors by similarity, not exact match → LangChain gives one consistent interface to plug in any embedding model + any vector store, which is the foundation of a RAG pipeline.**


# Vector Databases — Detailed Notes
*(Source: Pinecone — "What is a Vector Database & How Does it Work?")*

---

## 1. What is a Vector Database?

> A vector database indexes and stores vector embeddings for fast retrieval and similarity search, with capabilities like CRUD operations, metadata filtering, horizontal scaling, and serverless.

- Modern AI applications (LLMs, generative AI, semantic search) rely on **vector embeddings** — numeric representations that carry semantic meaning and have many attributes/dimensions.
- Traditional **scalar-based databases** cannot efficiently handle this kind of high-dimensional, complex data at scale.
- Vector databases combine:
  - The **capabilities of a traditional database** (CRUD, scaling, security, backups) — which a standalone vector index lacks, **plus**
  - **Specialization in embeddings** — which traditional scalar databases lack.

### How it fits into an AI application (3-step flow)
1. An **embedding model** creates vector embeddings for the content to be indexed.
2. The embedding is **inserted into the vector database**, along with a reference to the original content.
3. When a query comes in, the **same embedding model** embeds the query text, and the database is searched for the most **similar** stored vectors — those vectors' original content is returned.

---

## 2. Vector Index vs. Vector Database

Standalone indexes like **FAISS** (Facebook AI Similarity Search) speed up similarity search, but lack full database capabilities. A vector database adds:

| Capability | What it means |
|---|---|
| **Data management** | Easy insert / delete / update — FAISS needs custom work to integrate with storage |
| **Metadata storage & filtering** | Store metadata per vector; filter search results using it |
| **Scalability** | Built for distributed/parallel scaling; standalone indexes need custom infra (e.g. Kubernetes) |
| **Real-time updates** | New data is queryable quickly; standalone indexes often need full re-indexing |
| **Backups & collections** | Routine backups; Pinecone lets you save specific indexes as "collections" for reuse |
| **Ecosystem integration** | Plugs into ETL (Spark), analytics (Tableau, Segment), visualization (Grafana), and AI tooling (LangChain, LlamaIndex, Cohere) |
| **Security & access control** | Built-in access control, multitenancy via namespaces to isolate data between users |

---

## 3. How a Vector Database Works — the Pipeline

Traditional databases match on **exact values** (`WHERE x = y`). Vector databases match using a **similarity metric** — finding the vector *most similar* to the query vector.

This is done via **Approximate Nearest Neighbor (ANN) search** — algorithms trade a small amount of accuracy for a large gain in speed (near-perfect accuracy is possible, but true exhaustive/exact search doesn't scale).

**Three-stage pipeline:**
1. **Indexing** — vectors are mapped into a special data structure (via PQ, LSH, HNSW, etc.) that enables fast search.
2. **Querying** — the indexed query vector is compared against indexed dataset vectors to find nearest neighbors, using a similarity metric.
3. **Post-processing** — (optional) final results may be re-ranked using a different similarity measure before being returned.

---

## 4. Indexing Algorithms

| Algorithm | How it works | Trade-off |
|---|---|---|
| **Random Projection** | Multiplies vectors by a random projection matrix to reduce dimensionality while preserving similarity | Faster search, but quality depends on how "random" the matrix truly is (expensive to generate well) |
| **Product Quantization (PQ)** | *Lossy* compression: split vector into chunks → build a "codebook" per chunk (via k-means) → encode each chunk as a codebook index → reassemble | More codebook entries = more accuracy but higher search cost, and vice versa |
| **Locality-Sensitive Hashing (LSH)** | Hash functions map similar vectors into the same "buckets"; query is hashed and compared only within its bucket | Very fast, but approximate; more hash functions = better accuracy but higher compute cost |
| **HNSW (Hierarchical Navigable Small World)** | Builds a hierarchical, graph-like structure where nodes = vector clusters and edges = similarity; queries navigate the graph toward the closest vectors | Popular for a strong speed/accuracy balance |

---

## 5. Similarity Measures

Used to compare the query vector against stored vectors:

| Measure | Range | Notes |
|---|---|---|
| **Cosine similarity** | -1 to 1 | 1 = identical direction, 0 = orthogonal, -1 = opposite |
| **Euclidean distance** | 0 to ∞ | 0 = identical vectors; measures straight-line distance |
| **Dot product** | -∞ to ∞ | Positive = same direction, 0 = orthogonal, negative = opposite direction |

The choice of measure affects results — pick based on use case (e.g. cosine is common for text embeddings).

---

## 6. Metadata Filtering

- Every stored vector can carry **metadata** (e.g. `category`, `date`, `source`).
- Vector DBs typically maintain **two indexes**: a vector index and a metadata index.
- Filtering can happen **before** or **after** the similarity search:

| Approach | Pros | Cons |
|---|---|---|
| **Pre-filtering** | Shrinks the search space upfront | May cause the system to miss relevant vectors that don't strictly satisfy the filter; heavy filters add overhead |
| **Post-filtering** | Ensures the full vector search runs uninterrupted | Extra overhead discarding irrelevant results after the fact |

Vector DBs optimize this using advanced metadata indexing and parallel processing — balancing filter accuracy against query speed.

---

## 7. Database Operations (Production Readiness)

### a) Performance & Fault Tolerance
- **Sharding** — partitioning data across nodes (e.g. by similarity clusters); queries use a "**scatter-gather**" pattern (sent to all shards, results combined).
- **Replication** — multiple copies of data across nodes for resilience:
  - **Eventual consistency** — faster, more available, but temporary mismatches possible.
  - **Strong consistency** — all replicas updated before a write completes; safer but higher latency.

### b) Monitoring
- **Resource usage** — CPU, memory, disk, network.
- **Query performance** — latency, throughput, error rates.
- **System health** — node status, replication health, component status.

### c) Access Control
Important for:
1. **Data protection** — guard sensitive data from unauthorized access.
2. **Compliance** — meet regulations (e.g. healthcare, finance).
3. **Accountability/auditing** — track user activity, trace breaches.
4. **Scalability/flexibility** — adapt permissions as the organization grows.

### d) Backups & Collections
- Regular backups to external/cloud storage for recovery.
- Pinecone specifically allows saving an index snapshot as a **"collection"** to later populate a new index.

### e) API & SDKs
- A simple, developer-friendly API layer wraps all this complexity.
- Language-specific SDKs let developers focus on use cases (semantic search, Q&A, hybrid search, image similarity, recommendations) without worrying about infrastructure.

---

## 8. Serverless Vector Databases (the "next generation")

First-generation vector DBs are accurate, fast, and scalable — but **expensive**, because compute and storage are tightly coupled. Serverless architecture solves three pain points:

| Pain point | Serverless solution |
|---|---|
| **Separation of storage & compute** | Geometric partitioning breaks the index into sub-indices (like Voronoi-style regions) so a query only searches relevant partitions, decoupling cost from constant compute |
| **Freshness** | A temporary **"freshness layer"** acts as a cache for new vectors while they wait to be placed into the (slower-to-build) partitioned index; a query router checks both the freshness layer and the main index |
| **Multitenancy** | Users with different query loads (e.g. 20 queries/sec vs. 20/month) must be intelligently co-located on shared hardware — grouping similar usage patterns keeps latency low without wasting cost on idle infrastructure |

This gives a new architecture generation: **separated storage/compute + freshness + multitenancy**, optimized for elastic, cost-efficient AI workloads.

---

## 9. Summary Takeaways

- Vector databases exist because **scalar-based traditional databases can't efficiently manage high-dimensional embedding data** at production scale.
- They combine **traditional database operational features** (CRUD, backups, access control, scaling) with **specialized ANN search** for embeddings.
- Core pipeline: **Index → Query → Post-process**, using algorithms like PQ, LSH, HNSW and similarity measures like cosine, Euclidean, and dot product.
- Metadata filtering (pre- or post-) allows finer-grained, structured queries alongside semantic similarity.
- Production concerns — sharding, replication, monitoring, access control, backups — make vector databases genuinely different from (and more capable than) a bare vector index like FAISS.
- **Serverless vector databases** are the emerging generation, solving cost/elasticity problems via separated storage-compute, freshness layers, and smart multitenancy.
