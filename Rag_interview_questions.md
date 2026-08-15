# RAG & GraphRAG Interview Prep Guide
### Theory + Simple Answers for All 70 Questions

---

## Chapter 1: RAG Foundations and System Design Fundamentals

### Q1. Design a document QA system for 50,000 internal documents
**Concept:** A basic RAG (Retrieval-Augmented Generation) pipeline has 4 stages: ingest → embed & index → retrieve → generate.

**Simple Answer:**
- **Ingestion:** Parse docs (PDF/DOCX/HTML) → clean text → split into chunks (~300–500 tokens, with overlap).
- **Embedding:** Convert chunks to vectors using an embedding model (e.g., OpenAI/Cohere/BGE).
- **Storage:** Store vectors in a vector DB (pgvector is enough at this scale — 50K docs is small).
- **Retrieval:** On a query, embed it, do top-K similarity search (K=5–10).
- **Generation:** Feed retrieved chunks + question into an LLM with a prompt that forces citation.
- **Leave out:** Don't add a knowledge graph, reranker, or agentic loop yet — 50K docs doesn't need that complexity. Add them only if evaluation shows a specific gap (e.g., multi-hop questions).
- **Justification principle:** Every component should be justified by a measured failure, not "best practice."

---

### Q2. Answers are confident and wrong — how do you localize the fault?
**Concept:** RAG failures happen at one of 3 layers: **retrieval** (wrong/no evidence retrieved), **context assembly** (evidence retrieved but lost/buried), or **generation** (LLM ignores or misreads evidence — hallucination despite good context).

**Simple Answer:** Debug in this order:
1. **Replay the query** and look at what chunks were actually retrieved. If the correct chunk isn't there → retrieval problem (embedding model, chunking, or index issue).
2. If the correct chunk **is** retrieved but the answer is still wrong → check the **prompt/context assembly** (Is it buried in the middle? Is it cut off due to context limits?).
3. If the chunk is present, prominent, and the answer still contradicts it → it's a **generation/faithfulness problem** (the model is ignoring context) — try a stricter prompt, lower temperature, or a faithfulness classifier.
This is basically a funnel: verify retrieval first (cheapest to check), then context handling, then generation — because most "hallucinations" are actually retrieval failures in disguise.

---

### Q3. Why not just use a 1M token context window instead of RAG?
**Concept:** Long context ≠ good retrieval. This relates to the "lost in the middle" phenomenon and cost/latency scaling.

**Simple Answer:**
- **Cost & latency:** Stuffing 1M tokens into every query is extremely expensive and slow (every query re-processes the whole corpus).
- **Accuracy:** Research shows LLMs attend poorly to information buried in the middle of long contexts ("lost in the middle") — recall drops even though the tokens are technically "there."
- **Freshness:** A giant context still needs re-loading whenever data changes; RAG updates incrementally.
- **Precision & governance:** RAG lets you filter by permissions, versioning, and relevance — a giant context dumps everything, including things the user shouldn't see.
- RAG is essentially a **relevance filter** that keeps the model focused on the few pieces of evidence that actually matter, which is both cheaper and more accurate than "give the model everything."

---

### Q4. Customer support bot regressed after a model upgrade — diagnose
**Concept:** Embedding models are not interchangeable — each has its own vector space. Swapping the embedding model without re-indexing breaks retrieval.

**Simple Answer:** The most likely cause: the team upgraded the **embedding model** but **did not re-embed and re-index the existing corpus**. Now queries are embedded in the new model's vector space, while stored chunks are still embedded in the old space — similarity scores become meaningless (comparing apples to oranges). Fix: re-embed the entire corpus with the new model and rebuild the index before switching traffic over. Always treat an embedding model change as a **full migration**, not a hot-swap.

---

### Q5. How do you choose K, and what breaks at the extremes?
**Concept:** K = number of chunks retrieved and passed to the LLM. It's a precision/recall and cost/latency trade-off.

**Simple Answer:**
- **Method:** Sweep K (e.g., 3, 5, 10, 20) on a labeled eval set, track answer accuracy/faithfulness and latency/cost — pick the K where accuracy plateaus.
- **K too small:** You risk missing relevant evidence (low recall) → incomplete or wrong answers.
- **K too large:** You add noise/irrelevant chunks, increase cost & latency, and dilute the model's attention (the useful chunk gets lost among distractors — hurts precision even if recall is fine).
- In practice, a reranker + smaller K (e.g., retrieve 50, rerank, keep top 5) usually beats a large flat K.

---

### Q6. Design the abstention policy and defend it against product pressure
**Concept:** Abstention = the system says "I don't know" instead of guessing when evidence is weak. This is a precision/recall trade-off between hallucination risk and helpfulness.

**Simple Answer:** Design a **confidence gate**: combine retrieval similarity score, number of supporting chunks, and an LLM-based faithfulness/self-check score. If confidence is below a threshold, respond with a partial answer + "I don't have enough information" rather than fabricating.
**Defense to product:** Frame it with numbers — show the hallucination rate at the current threshold vs. a lower one, and its cost (user trust, compliance risk, support tickets from wrong answers). Propose a middle ground: instead of a hard refuse/answer binary, show **confidence-tiered responses** (fully answer / answer with caveat / ask clarifying question / refuse) so the bot refuses less often but never confidently lies.

---

### Q7. Take this design from 50,000 documents to 10 million documents
**Concept:** Scaling RAG requires changes in indexing (ANN algorithms), infra (sharding), and pipeline (incremental ingestion), while the core retrieve→generate loop stays the same.

**Simple Answer:**
- **What stays the same:** The overall architecture — chunk → embed → retrieve → generate.
- **What changes:**
  - **Vector index:** Move from exact/brute-force or small pgvector setup to an ANN index (HNSW/IVF) or a dedicated vector DB (Milvus, Qdrant, Pinecone) with sharding.
  - **Ingestion:** Must become an incremental, event-driven pipeline (not a batch script) with dedup and versioning.
  - **Retrieval:** Add hybrid search (dense + BM25) and reranking, since precision matters more with more distractor documents.
  - **Metadata filtering:** Needed to reduce search space (by department, date, access level) before doing vector search.
  - **Infra:** Horizontal scaling, caching layer for hot queries, monitoring for index staleness.

---

## Chapter 2: Document Ingestion, Parsing, and Chunking

### Q8. Design the ingestion pipeline for 5M documents, mixed formats
**Concept:** Ingestion pipelines are format-aware ETL systems: extract → normalize → chunk → embed → index, with different extraction paths per format.

**Simple Answer:**
- **Format detection & routing:** Route each doc to the right parser — OCR (e.g., Tesseract/Textract) for scanned PDFs, native text extraction for PDFs/DOCX, HTML-to-text with tag stripping for Confluence, structured parsing for CSV.
- **Normalization:** Convert everything to a common clean-text + metadata schema (title, source, date, permissions).
- **Chunking:** Format-aware chunking (e.g., respect headings in Confluence, table boundaries in CSV).
- **Embedding & indexing:** Batch embed, write to vector DB with metadata for filtering.
- **Pipeline design:** Use a queue-based, distributed pipeline (e.g., Kafka + workers) so failures in one document don't block others, with retry/dead-letter handling for corrupt files.
- **Idempotency:** Use content hashes so re-ingesting the same doc doesn't duplicate it.

---

### Q9. Explain how you would select chunk size empirically (method, not a number)
**Concept:** Chunk size is a search-space parameter, tuned like a hyperparameter using a labeled eval set.

**Simple Answer:** Method:
1. Build a small labeled set of (question, correct source passage) pairs.
2. Try several chunk sizes/overlaps (e.g., 200/50, 500/100, 1000/200 tokens).
3. For each configuration, run retrieval and measure **recall@K** (did the correct passage get retrieved?) and downstream **answer accuracy**.
4. Also track cost/latency (bigger chunks = fewer chunks = cheaper index, but noisier context).
5. Pick the smallest chunk size that doesn't hurt recall — smaller chunks are more precise but risk losing context; the empirical sweep finds the sweet spot for *your* content type (a legal contract needs different chunking than a chat log).

---

### Q10. Recall works overall (0.86) but fails completely (0.09) for one doc class
**Concept:** Aggregate metrics hide per-segment failures — always slice metrics by document type/source.

**Simple Answer:** This is almost always a **parsing or chunking failure specific to that format** — e.g., that document class is scanned images with no OCR applied (so chunks are empty or garbage), or it's a table-heavy format where chunking destroys row/column structure, or it uses a different language/encoding the embedding model handles poorly. Steps: pull 10 examples from that class, look at the actual stored chunks — if they're empty/garbled, it's a parsing bug; if they look fine but never get retrieved, it's likely an embedding mismatch (e.g., that content type embeds poorly) or a metadata filter silently excluding it.

---

### Q11. Wiki assistant returns the same navigation text for every question
**Concept:** Boilerplate contamination — repeated structural text (menus, headers, footers) gets chunked and embedded like content, and because it appears in many pages it can dominate similarity matches or simply be an easy, generic match for vague queries.

**Simple Answer:** The scraper is capturing site chrome (nav bar, sidebar) as part of every page's content. Because this text is short, generic, and repeated across the corpus, it embeds to a "central" vector that's mediocre-but-not-terrible-similarity to almost any query, so it keeps surfacing. Fix: strip boilerplate during HTML parsing (target the main content div, use readability-style extraction), and add a dedup/near-duplicate filter so repeated boilerplate chunks are only indexed once (or excluded).

---

### Q12. When is semantic chunking worth 40x the ingestion cost?
**Concept:** Semantic chunking (using embeddings/LLMs to find natural topic boundaries) vs. fixed-size chunking — a cost/quality trade-off.

**Simple Answer:** Worth it when: (1) documents have highly variable structure with no reliable delimiters (e.g., long-form legal or medical narratives where a fixed window frequently splits mid-argument), and (2) retrieval precision directly drives high-stakes outcomes (compliance, medical, legal) where a wrong or incomplete chunk is costly. Not worth it when: documents are short, well-structured (already have headings/paragraphs you can split on), or the corpus is large and cost-sensitive with low per-query stakes (e.g., internal FAQ bot) — there, fixed-size chunking with heading-aware splitting gets 90% of the benefit for a fraction of the cost.

---

### Q13. Handle document updates/deletions without reindexing everything
**Concept:** Incremental index maintenance — treat the vector index like a database, not a static artifact.

**Simple Answer:**
- **Track state:** Store a content hash + version per document/chunk.
- **Updates:** On re-ingest, compare hash; if changed, delete old chunk vectors for that doc and insert new ones (most vector DBs support delete-by-metadata-filter, e.g., `doc_id`).
- **Deletions:** Soft-delete via metadata flag (fast) or hard delete via ID (reclaims space), triggered by a change-detection job (webhook or periodic diff).
- **Chunk-level granularity:** Only re-chunk/re-embed the changed sections of a document, not the whole document, when possible.
- This turns a 40-hour full rebuild into a small delta job that runs continuously.

---

### Q14. Why does adding the section heading to each chunk improve retrieval so much?
**Concept:** Chunks lose surrounding context when split; prepending the heading re-injects topical/semantic context into the embedding.

**Simple Answer:** A chunk like "...must be submitted within 30 days" is ambiguous alone — the embedding model doesn't know what it's about. Prepending "Section: Refund Policy" gives the embedding model a strong topical anchor, so the resulting vector sits closer to queries about refunds. Mechanically, it increases the semantic signal-to-noise ratio in short chunks and disambiguates chunks that would otherwise be generic-sounding, directly improving cosine similarity to relevant queries.

---

## Chapter 3: Embeddings and Vector Database Architecture

### Q15. Design storage for a multi-tenant platform, 2,000 tenants, contractual isolation
**Concept:** Multi-tenancy in vector DBs balances isolation guarantees against operational overhead — options range from shared index+metadata filter to fully separate indexes/collections.

**Simple Answer:**
- **Small tenants (few hundred docs):** Use a shared index with a mandatory `tenant_id` metadata filter applied on every query (cheap, efficient, but isolation depends on filter correctness — risky for "contractual" isolation).
- **Large tenants (100K+ docs) or strict compliance needs:** Give them dedicated collections/indexes (or even separate DB instances) — stronger isolation, easier to audit, easier to delete-on-offboarding.
- **Hybrid approach:** Tier by tenant size/sensitivity — small tenants share infra with strict filters + row-level security; large/regulated tenants get dedicated indexes.
- Always test the "did filter X leak into tenant Y's results" case explicitly in CI, since contractual isolation means a filter bug is a legal incident, not just a bug.

---

### Q16. Explain HNSW well enough to implement search over it
**Concept:** HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor graph-based index.

**Simple Answer:**
- **Structure:** Vectors are nodes in a multi-layer graph. The top layer has very few nodes with long-range links (highway); each lower layer has progressively more nodes with shorter-range links, down to the bottom layer, which contains every vector.
- **Insertion:** A new vector is assigned a random top layer (probability decreases with layer height). It's inserted by greedily walking from the top layer down, at each layer finding its nearest neighbors and connecting to them (with a max-connections parameter, `M`).
- **Search:** Start at the entry point in the top layer. Greedily move to the neighbor closest to the query vector until no closer neighbor exists at that layer, then drop to the next layer and repeat, using the current best point as the new starting point. At the bottom layer, do a local beam search (`ef` parameter controls breadth) to find the top-K nearest neighbors.
- **Why it's fast:** The top layers let you skip across large sections of the vector space quickly (like a skip list), while the bottom layer gives fine-grained accuracy — giving logarithmic-ish search time instead of scanning every vector.

---

### Q17. Search returns nothing for one customer, but their docs are indexed
**Concept:** Silent metadata/filter mismatches are one of the most common "invisible" vector DB bugs.

**Simple Answer:** Check, in order: (1) **Metadata filter mismatch** — is `tenant_id`/`customer_id` stored as a string in the index but queried as an int (or vice versa), or is there a typo/case mismatch? (2) **Empty embeddings** — did ingestion for this customer fail silently, leaving zero vectors (rows exist in a DB table but never got embedded)? (3) **Index partition issue** — if sharded, is this tenant's shard offline or not yet merged into the searchable index? (4) **Permission/ACL filter** — is an access-control layer wrongly zeroing out results for this customer's role? This is a "check the boring stuff first" bug — always verify data actually landed in the *searchable* index, not just the source DB.

---

### Q18. Recall dropped after migrating from exhaustive to HNSW search
**Concept:** HNSW is approximate — it trades some recall for massive speed gains, controlled by parameters like `ef_search`, `M`, and `ef_construction`.

**Simple Answer:** This is expected — HNSW doesn't guarantee finding the true top-K, only an approximation. The fix isn't to revert to brute force (which won't scale), but to **tune HNSW's recall/speed trade-off**: increase `ef_search` (search-time exploration breadth) and `ef_construction` (build-time graph quality) — this recovers most of the lost recall at a modest latency cost. Present leadership with the recall-vs-latency curve so they can pick an operating point, rather than reverting to exhaustive search, which won't scale to production traffic/corpus size.

---

### Q19. pgvector or a dedicated vector database, for 8 engineers / 3M chunks?
**Concept:** Trade-off between operational simplicity (using tools you already run) vs. specialized performance/features.

**Simple Answer:** For a small team and 3M chunks (a moderate scale), **pgvector is a strong choice**: no new infra to learn/operate, transactional consistency with your existing relational data (metadata joins are trivial), and pgvector's HNSW/IVFFlat indexes handle millions of vectors fine. A dedicated vector DB (Pinecone/Milvus/Qdrant) becomes worth it when you need: sub-50ms latency at very high QPS, advanced features (multi-vector search, built-in reranking), or scale beyond tens of millions of vectors where Postgres write/index performance becomes a bottleneck. With 8 engineers, minimizing new operational surface area usually wins — start with pgvector, migrate later if you hit a measured limit.

---

### Q20. Upgrade the embedding model with zero downtime, 25M chunks, 768→1024 dims
**Concept:** Blue-green migration pattern for vector indexes — new dimensionality means a fully separate index, not an in-place update.

**Simple Answer:**
1. Stand up a **new, separate index** with the new (1024-dim) embeddings.
2. **Backfill:** Re-embed all 25M chunks in the background using the new model, writing to the new index — this can take time but doesn't affect production, which still reads from the old index.
3. **Shadow test:** Run queries against both indexes in parallel (old serves users, new is compared offline) to validate quality before cutover.
4. **Cutover:** Once validated, switch read traffic to the new index (feature-flag/config switch — instant, zero downtime).
5. **Cleanup:** Decommission the old index after a safety window.
Because dimensions differ, you *cannot* do an in-place index update — the two embedding spaces are fundamentally incompatible, so this must be a full parallel-build-then-swap.

---

### Q21. Why do similarity scores from different models mean different things?
**Concept:** Embedding models are trained with different objectives, loss functions, and normalization — cosine similarity is a relative, not absolute, measure calibrated to each model's training distribution.

**Simple Answer:** Similarity scores aren't a universal, calibrated "percent match" — they're specific to the geometry each model learned. One model's embeddings might cluster tightly (most scores between 0.7–0.95), while another spreads scores widely (0.2–0.6 for good matches). So a hardcoded threshold like "alert below 0.7" from one model is meaningless for another — you'd either alert constantly or never. **Evaluate that proposal:** reject the fixed threshold; instead, calibrate a threshold empirically per model using a labeled eval set (e.g., pick the threshold that best separates relevant/irrelevant chunks on your data), and re-calibrate whenever the embedding model changes.


## Chapter 4: Building a Complete Basic RAG System

### Q22. Design a conversational RAG service with a strict 2-second p95
**Concept:** Latency budgeting — decompose the pipeline into stages, assign a time budget to each, and design fallbacks for when budgets are exceeded.

**Simple Answer:**
- **Budget breakdown (example for 2s p95):** Query embedding (~50ms) → retrieval (~150ms) → reranking (~100ms, optional) → LLM generation (~1.5s, dominant cost) → overhead/network (~200ms).
- **Caching:** Cache embeddings for repeated/similar queries, cache full responses for common questions (semantic cache), cache retrieval results for a conversation session.
- **What you drop under load:** First drop reranking (adds latency for a quality gain), then reduce K, then fall back to a smaller/faster LLM, and as a last resort return a "still working" partial response or queue the request.
- **Streaming:** Stream LLM tokens to the user immediately so perceived latency is much lower than actual full-generation time.
- Generation is almost always the bottleneck — most latency engineering effort should go there (model choice, streaming, prompt length).

---

### Q23. Citations point at the wrong sources
**Concept:** Citation mapping bugs — the citation mechanism (index-to-source mapping) is decoupled from the actual generation, so it can silently misalign.

**Simple Answer:** Likely causes: (1) the chunk **order sent to the LLM** differs from the order used to build the citation map (e.g., you reranked chunks after building the citation index, so citation `[2]` now points to the wrong chunk); (2) the LLM is asked to cite by number but **hallucinates a plausible-looking number** instead of actually using the tool/structure you gave it; (3) **off-by-one indexing** between your internal array and what's shown to the model. Fix: build the citation map from the *exact* final chunk order sent to the LLM, use structured output (force the model to emit citations tied to explicit chunk IDs, not free-form numbers), and validate post-hoc — programmatically check that the cited chunk actually contains text overlapping with the claim, and flag/strip citations that don't verify.

---

### Q24. Why does placing the best chunk in the middle of the prompt hurt accuracy?
**Concept:** "Lost in the middle" — LLMs exhibit a U-shaped attention/recall curve across long contexts, most reliably recalling information at the beginning and end.

**Simple Answer:** Transformer attention doesn't treat all context positions equally in practice — empirically, models are best at using information near the **start** and **end** of the prompt, and worse at using information buried in the **middle**, even though architecturally every token is "visible." So if your most relevant chunk lands mid-prompt, the model is statistically less likely to use it correctly. Fix: your context builder should place the highest-relevance chunk(s) **first and/or last** in the prompt (not by raw retrieval rank, but reordered for the "best chunks near the edges" pattern), and keep the prompt as short as needed rather than padding with lower-value chunks.

---

### Q25. Answers degrade only for long conversations, cost triples
**Concept:** Context window bloat — conversational RAG accumulates full history + retrieved chunks each turn, causing linear/quadratic growth in prompt size, cost, and noise.

**Simple Answer:** By turn 8, the system is likely sending the **entire conversation history plus newly retrieved chunks each turn**, without pruning. This means: (1) cost triples because prompt tokens grow with every turn; (2) accuracy drops because the prompt now contains a lot of old, possibly irrelevant chunks/messages competing for the model's attention with the actually-relevant new context (the "lost in the middle" effect compounds). Fix: summarize or truncate older conversation turns, re-retrieve fresh context each turn rather than accumulating all past retrievals, and cap total context size with a sliding window + running summary of the conversation so far.

---

### Q26. Retrieval failure: refusal or best-effort answer?
**Concept:** This is fundamentally a risk-tolerance/policy decision, not a technical one — the "right" answer depends on the cost of a wrong answer vs. the cost of no answer.

**Simple Answer:** Don't pick one extreme — implement **tiered response behavior** based on confidence:
- High confidence (strong retrieval match) → direct answer with citations.
- Medium confidence → answer, but explicitly flagged as uncertain ("Based on limited information...") with an offer to escalate/search further.
- Low confidence → refuse and redirect (e.g., "I couldn't find this in our docs — here's who to ask").
This satisfies both sides: product gets an answer most of the time, compliance gets a hard refusal exactly when evidence is genuinely weak. The key deliverable to both stakeholders is the **calibration data** — showing false-answer rate at each confidence tier so the threshold is a data-backed business decision, not a guess.

---

### Q27. Version prompts and roll them out safely
**Concept:** Prompt engineering needs the same rigor as code deployment: versioning, testing, gradual rollout, and rollback.

**Simple Answer:**
- **Version control:** Treat prompts as code — store in git, tag versions, log which prompt version produced each response.
- **Offline eval gate:** Every prompt change must pass a regression test suite (a labeled eval set covering known edge cases) before being considered for release — this is what catches "improved on my test set, regressed in prod," by making the test set broader/more representative over time.
- **Canary/A-B rollout:** Ship the new prompt to a small % of traffic first, monitor live metrics (thumbs-down rate, faithfulness scores, latency) before full rollout.
- **Fast rollback:** Prompt changes should be a config flip, not a code deploy, so a bad prompt can be reverted in seconds.
- **Root cause for the regression:** Usually the offline eval set was too narrow (didn't represent real production query diversity) — expand it with real production failure cases over time.

---

### Q28. Walk through everything between "Enter" and the first token
**Concept:** End-to-end latency trace of a RAG request — useful for identifying parallelization opportunities.

**Simple Answer:** Sequence: (1) Request hits API → auth/rate-limit check (a few ms). (2) Query embedding generated (~30–80ms, network + model inference). (3) Vector search against the index (~20–150ms depending on index size/method). (4) [Optional, can be parallel with #3] Lexical/BM25 search if hybrid. (5) Fusion + reranking of combined results (~50–150ms). (6) Context assembly — format chunks + history into the prompt (near-instant, CPU only). (7) LLM call begins — **this is where most time goes**: time-to-first-token depends on prompt length (prefill) and queueing/model load; can be 300ms–2s+. (8) First token streams back to the user.
**Parallelizable:** Dense + lexical retrieval can run concurrently; guardrail/safety checks can run concurrently with generation start. **Not parallelizable:** embedding → retrieval → generation is inherently sequential since each step depends on the last. The LLM prefill/generation step is almost always the dominant cost.


## Chapter 5: Advanced Retrieval and Hybrid RAG

### Q29. Design hybrid retrieval within a 400ms budget
**Concept:** Hybrid retrieval = dense (semantic) + lexical (keyword/BM25) search, combined via fusion, optionally reranked — each stage adds latency and quality.

**Simple Answer:** Budget: dense search (~100ms) + BM25 search (~50ms, can run **in parallel** with dense) → fusion via Reciprocal Rank Fusion (~5ms, cheap) → rerank top ~50 candidates with a cross-encoder (~150–200ms, the most expensive optional stage) → return top-K. Total ≈ 300–350ms, leaving headroom.
**What you cut first under pressure:** Reranking first (biggest latency cost for a quality gain), then reduce the reranker's candidate pool size, then fall back to dense-only if BM25 isn't adding value for that query type. Run dense and lexical searches **concurrently**, not sequentially, since they're independent — that alone saves ~50ms.

---

### Q30. Why fuse ranks instead of normalizing and adding scores?
**Concept:** Reciprocal Rank Fusion (RRF) vs. score-based fusion — score fusion requires scores to be on comparable scales, which dense/lexical scores are not.

**Simple Answer:** BM25 scores and cosine similarity scores live on **completely different, uncalibrated scales** (BM25 can range 0–30+ depending on term frequency; cosine similarity is bounded -1 to 1) — normalizing them (e.g., min-max scaling) is fragile because the "shape" of each distribution differs per query, so a normalized 0.8 from BM25 doesn't mean the same thing as 0.8 from dense search. **RRF sidesteps this entirely** by using each result's **rank position**, not its raw score — it's a simple formula (`1/(k + rank)`) that just needs "did this appear near the top of this list," which is comparable across totally different scoring systems. This makes fusion robust without any calibration/tuning per retriever.

---

### Q31. Hybrid retrieval performs worse than dense alone
**Concept:** BM25 can inject noise for certain query types (especially semantic/paraphrased queries), and fusion weighting can be miscalibrated.

**Simple Answer:** This is possible when: (1) the query set is dominated by semantic/conceptual questions (not exact keyword matches) where BM25 surfaces lexically-similar-but-semantically-irrelevant chunks that then get fused in and displace better dense results; (2) the fusion weighting (if using weighted RRF) over-weights BM25; (3) BM25 index has quality issues (bad tokenization/stopword handling for your domain). Fix: evaluate BM25 and dense separately on your golden set to see which is actually driving the regression, tune the fusion weight (or use pure RRF instead of weighted), and consider query-type routing — send keyword-heavy queries (product codes, names) to BM25-weighted fusion and conceptual queries to dense-weighted fusion instead of one-size-fits-all.

---

### Q32. Legal research tool misses precedents a paralegal finds — diagnose
**Concept:** Dense retrieval struggles with domain-specific exact terminology, citations, and rare/specific phrasing that a keyword search would catch trivially — reranking doesn't fix a retrieval recall problem, since it only reorders what's already retrieved.

**Simple Answer:** Dense embeddings are good at *semantic* similarity but weak on exact legal terms, case citation formats, or rare phrasing (e.g., a specific statute number) — these need exact/lexical matching. If the pipeline is dense + rerank with **no lexical/BM25 component**, the correct precedent may never even make it into the top-K candidate pool for the reranker to consider (reranking can't rescue something that was never retrieved). Fix: add hybrid (BM25) retrieval specifically to catch exact citation/term matches, and consider a legal-domain-tuned embedding model, since general-purpose embeddings underperform on specialized jargon.

---

### Q33. When would you skip reranking entirely?
**Concept:** Reranking (typically cross-encoder) adds real latency/cost for a quality gain that isn't always worth it — argue against the default recommendation.

**Simple Answer:** Skip reranking when: (1) **latency is the primary constraint** (e.g., sub-200ms real-time use cases) and the marginal accuracy gain doesn't justify it; (2) the corpus is small/homogeneous enough that top-K dense retrieval is already near-perfect (little room for reranking to improve); (3) cost-sensitivity is high and query volume is massive (reranking every query at scale is expensive); (4) initial retrieval is already hybrid and well-tuned, so the "noisy top-K" problem reranking solves doesn't really exist here. In short: reranking earns its cost when retrieval is noisy/large-scale and answer quality is paramount — it's overkill for small, fast, low-stakes lookups.

---

### Q34. Design the query classifier for adaptive routing
**Concept:** A lightweight classifier routes queries to different retrieval/generation strategies (e.g., simple lookup vs. multi-hop vs. no-retrieval-needed) based on query type.

**Simple Answer:** Build a small, fast model (or even a rules + small fine-tuned classifier, not a full LLM call) that classifies a query into categories like: "factual lookup" (single retrieval), "multi-hop/complex" (needs agentic loop), "no retrieval needed" (chit-chat/greeting), "out of scope." Keep it fast/cheap by using a small distilled model or even a prompt to a tiny/local LLM, not the main generation model. **Monitoring:** log classifier decisions vs. downstream outcomes (did a "simple" classification actually get answered correctly, or did it need escalation?), and periodically audit misroutes to retrain the classifier — treat it like any production ML model with its own accuracy metric, not a "set and forget" component.

---

### Q35. Explain corrective RAG vs. answer verification — why have both?
**Concept:** Corrective RAG acts on the **retrieval** step (before generation); answer verification acts on the **output** (after generation) — they catch different failure classes.

**Simple Answer:** **Corrective RAG (CRAG)** evaluates retrieved chunks *before* generation — if they're irrelevant/low-quality, it triggers a fallback (re-query, expand search, or go to the web) so the LLM never even sees bad context. **Answer verification** checks the *generated answer* against the context afterward — catching cases where retrieval was fine but the LLM still hallucinated or misstated something. You need both because they catch different failure modes: CRAG prevents "garbage in," verification catches "garbage out" even when the input was good (LLMs can still misread correct context). Using only one leaves a gap — CRAG can't catch generation errors, and post-hoc verification can't fix a fundamentally bad context (though it can at least flag it).


## Chapter 6: RAG Evaluation, Observability, and Debugging

### Q36. Design the evaluation system for a platform serving 40 teams
**Concept:** Multi-tenant evaluation needs a shared framework with per-team customization — standard metrics + team-specific golden sets and thresholds.

**Simple Answer:**
- **Shared infrastructure:** One evaluation pipeline/tooling (metric computation, dashboards, CI hooks) used by all teams — don't let each team build their own from scratch.
- **Per-team golden sets:** Each team owns a labeled (question, expected answer/sources) dataset reflecting their corpus and use case, since quality bars differ (a legal team's bar ≠ an internal FAQ bot's bar).
- **Standard metric suite:** retrieval recall@K/MRR, faithfulness (does the answer match retrieved evidence), answer relevance, latency/cost — computed consistently so teams are comparable.
- **Gating:** Each team sets its own pass/fail thresholds for their CI pipeline, using the shared metric definitions.
- **Aggregation layer:** A central dashboard rolls up all 40 teams' health so platform owners can spot systemic issues (e.g., all teams' faithfulness dropped after a shared embedding model upgrade).

---

### Q37. How do you validate your LLM judge is trustworthy?
**Concept:** LLM-as-judge is itself an ML system with its own accuracy — it must be validated against human judgment, not assumed correct.

**Simple Answer:** Build a small set (e.g., 100–300 examples) with **human-labeled ground truth** for faithfulness/quality. Run the LLM judge on the same examples and compute **agreement rate** (e.g., Cohen's kappa) between the judge and humans. If agreement is high (e.g., >0.8), the judge is trustworthy enough for scale; if low, either improve the judge prompt (more explicit rubric, chain-of-thought reasoning, few-shot examples) or use a different/larger model as judge. Periodically re-validate (judge drift can happen if the underlying model changes) and spot-check disagreements to understand systematic judge biases (e.g., judges often favor longer or more confident-sounding answers regardless of correctness).

---

### Q38. Offline metrics improved but users complained (thumbs-down doubled)
**Concept:** Offline metrics (computed on a static eval set) can diverge from real user experience if the eval set doesn't represent actual usage, or if the metric itself is a poor proxy for user satisfaction.

**Simple Answer:** Likely causes: (1) the **eval set is stale/unrepresentative** — it doesn't cover the query patterns real users are now sending; (2) faithfulness (is the answer supported by retrieved evidence) is not the same as **usefulness/relevance** — an answer can be perfectly faithful to irrelevant retrieved chunks ("faithfully" says "I don't have that information" when it should have found the answer) and still frustrate users; (3) a recent change may have improved faithfulness at the cost of some other quality dimension (e.g., answer completeness, tone, speed) not captured by the faithfulness metric. Action: pull actual thumbs-down transcripts, categorize the complaints, and check whether they're clustering around a dimension your offline eval doesn't measure — then add that dimension to the eval suite.

---

### Q39. Retrieval metrics look excellent, answers are still wrong
**Concept:** Good retrieval/faithfulness metrics don't guarantee good answers — this points to either a mismatch between the golden set and real query distribution, or a generation-side issue not captured by faithfulness.

**Simple Answer:** Possible explanations: (1) **faithfulness measures "is the answer consistent with retrieved context," not "is the retrieved context actually correct/complete/current"** — if the source documents themselves are outdated or the question requires info from a document not in top-5, the model can be "faithful" to wrong/incomplete context; (2) the **eval golden set** may not reflect real user question distribution (evaluation looks great on the test set but users ask differently); (3) numeric/reasoning-heavy questions can have high faithfulness scores despite wrong final computation (the judge checks "is this grounded in text" not "is the math right"). Action: sample real production failures, manually trace them — check if it's a source-data quality issue, a query-distribution mismatch, or a reasoning failure the faithfulness judge doesn't catch.

---

### Q40. How much evaluation is enough before shipping?
**Concept:** Evaluation depth is a risk-vs-velocity trade-off — tiered testing lets you get fast feedback without skipping safety nets.

**Simple Answer:** Resolve with a **tiered eval strategy**, not an all-or-nothing choice: run a **fast smoke-test subset** (a small, high-signal slice of the eval set covering known critical edge cases, ~2 minutes) as a merge-blocking gate, and run the **full 90-minute suite asynchronously** post-merge (or nightly) with alerts if it fails — with an agreed policy that a full-suite failure triggers an immediate rollback/hotfix, not a "we'll get to it" ticket. This gives engineers fast iteration while still catching regressions before they compound, and reserves human review/expensive eval for changes flagged as higher-risk (e.g., prompt or retrieval logic changes vs. a UI tweak).

---

### Q41. Design the observability stack, no ground truth in production
**Concept:** Production observability relies on proxy signals and reference-free metrics since you don't have labels for real traffic.

**Simple Answer:**
- **Log:** every query, retrieved chunk IDs + scores, final answer, latency per stage, user feedback (thumbs up/down), and any explicit "I don't know" abstentions.
- **Reference-free metrics computed live:** faithfulness (LLM judge comparing answer to retrieved context — doesn't need ground truth), retrieval confidence distribution (are similarity scores trending down over time?), answer length/refusal rate trends.
- **Alert on:** sudden drop in average retrieval similarity score (signals index/embedding drift or a new query pattern), spike in refusal rate, spike in thumbs-down rate, latency SLA breaches, and faithfulness score dropping below a threshold on sampled traffic.
- **Sampling + human review:** Since you can't grade every response automatically with full confidence, sample a % of traffic daily for human/expert spot-checks to catch what automated judges miss, and feed confirmed failures back into the golden eval set.

---

### Q42. Why does claim-level faithfulness beat paragraph-level scoring?
**Concept:** Faithfulness scoring at coarse granularity (whole paragraph/answer) can hide partial hallucination — decomposing into atomic claims isolates exactly which statements are (un)supported.

**Simple Answer:** A paragraph-level judge asks "is this whole answer roughly grounded?" — but an answer with 5 claims where 4 are correct and 1 is fabricated can still score "mostly faithful," letting the hallucinated claim slip through. **Claim-level decomposition** breaks the answer into individual atomic factual statements and checks each one against the retrieved evidence independently — so a single false claim is caught precisely, rather than averaged away by surrounding correct claims. This also gives you actionable debugging info (exactly *which* claim is unsupported) instead of a vague "somewhat faithful" score, even though both approaches use the same underlying judge model.


## Chapter 7: Knowledge Graph Fundamentals

### Q43. Design the KG layer for 200,000 docs (contracts, incident reports, org charts)
**Concept:** Knowledge graph construction = entity/relationship extraction + schema design + storage + ongoing sync — different document types need different extraction strategies but a shared graph schema.

**Simple Answer:**
- **Extraction:** Use an LLM (or domain-specific NER model) to extract entities (people, companies, dates, clauses, systems) and relationships (e.g., "Company X — party_to → Contract Y", "Employee A — reports_to → Employee B", "Incident Z — affected → System C") per document type, with type-specific prompts/schemas.
- **Schema design:** Define a **controlled, constrained set of node/relationship types upfront** (don't let the LLM invent arbitrary types — see Q46) — e.g., `Person`, `Organization`, `Contract`, `Clause`, `System`, `Incident`, with relationships like `PARTY_TO`, `REPORTS_TO`, `CAUSED_BY`.
- **Storage:** A graph database (Neo4j/Neptune) for the graph itself, plus keep vector embeddings of the source chunks for hybrid graph+vector retrieval.
- **Maintenance:** Incremental extraction on document updates (Q48), entity resolution/deduplication (same "Acme Corp" mentioned differently across docs should merge to one node).
- **Access control:** Since contracts and org charts contain sensitive data, encode permissions as graph properties and filter at query time.

---

### Q44. Why is a graph database faster than SQL for multi-hop queries?
**Concept:** Graph DBs use **index-free adjacency** (each node directly stores pointers to its neighbors), while relational DBs must compute joins at query time.

**Simple Answer:** In SQL, a multi-hop query (e.g., "friends of friends of friends") requires a **join** at each hop — the database must scan/index-lookup to match foreign keys across tables, and each additional hop multiplies this cost (join cost typically grows with table size and hop count). In a graph database, each node **physically stores direct references to its connected nodes** (index-free adjacency), so traversing a relationship is a constant-time pointer-following operation regardless of overall graph size — hopping from node to node doesn't require re-scanning an index. This means multi-hop traversal in a graph DB scales with the **size of the traversed subgraph**, not the size of the entire dataset, whereas SQL joins scale with table size at every hop.

---

### Q45. Extracted graph is fragmented, traversals return nothing (80K nodes, 12K relationships)
**Concept:** A healthy knowledge graph needs relationship density proportional to node count — a ratio this low (0.15 relationships per node) signals extraction failure, not a naturally sparse domain.

**Simple Answer:** 12,000 relationships across 80,000 nodes is extremely sparse (most nodes likely have zero or one connection) — this points to an **extraction problem**, not a real characteristic of the data. Common causes: (1) the extraction prompt/model is good at pulling out entities but poor at identifying relationships between them (entity extraction and relationship extraction are different tasks — one succeeding doesn't guarantee the other); (2) **entity resolution failure** — the same real-world entity is created as many duplicate nodes (e.g., "Acme Corp," "Acme Corporation," "ACME" as 3 separate nodes), so relationships that *should* connect through a shared entity don't, fragmenting the graph into disconnected islands. Fix: audit a sample of nodes for duplicates (add entity resolution/deduplication), and strengthen the relationship-extraction step (e.g., explicit prompting for relationships, not just entities, or a two-pass extraction: entities first, then relationships between confirmed entities).

---

### Q46. LLM built a graph with 340 relationship types — recover without re-extracting
**Concept:** Unconstrained LLM extraction produces an uncontrolled/exploded schema (near-duplicate relationship types like `WORKS_AT`, `EMPLOYED_BY`, `IS_EMPLOYEE_OF`) that makes querying impossible.

**Simple Answer:** This happens because the extraction prompt let the LLM **freely invent relationship type names** per document instead of choosing from a fixed vocabulary — so semantically identical relationships get many different labels. Recovery without full re-extraction: (1) Cluster the 340 types by semantic similarity (embed the relationship type names, cluster them) to find groups that mean the same thing; (2) build a **type-mapping/normalization table** (e.g., `EMPLOYED_BY`, `WORKS_FOR`, `IS_STAFF_OF` → canonical `WORKS_AT`); (3) run a bulk relationship-type rewrite (a Cypher script) using this mapping instead of re-running extraction. Going forward, **constrain the extraction prompt to a fixed, closed set of allowed relationship types** (a schema) rather than letting the LLM freely generate labels.

---

### Q47. When is a knowledge graph not worth building? (argue against)
**Concept:** KGs add real cost (extraction pipeline, schema maintenance, graph DB ops) that's only justified when the query patterns genuinely need multi-hop/relational reasoning.

**Simple Answer:** Argue against KG adoption when: (1) most user questions are **single-hop factual lookups** ("what's our refund policy") — vector RAG alone handles these fine, and a graph adds extraction/maintenance cost for no query benefit; (2) the domain's relationships are unstable/low-value (e.g., customer support chat logs have few meaningful entity relationships worth modeling); (3) the team lacks capacity to maintain the extraction pipeline and schema over time (a stale/fragmented graph, per Q45/Q46, is worse than no graph — it creates false confidence); (4) budget/timeline don't support the multi-week upfront investment schema design and entity resolution require. Rule of thumb: only build a KG when you have concrete evidence (from failure analysis of vector RAG) that users are asking **multi-hop, relational, or aggregation** questions ("which suppliers affect which customers") that vector similarity fundamentally cannot answer.

---

### Q48. Keep the graph synchronized with changing documents, without full rebuilds
**Concept:** Incremental graph maintenance — mirrors incremental vector index maintenance (Q13), but must also handle relationship consistency.

**Simple Answer:**
- **Change detection:** Track document version/hash; on change, identify only the affected document.
- **Targeted re-extraction:** Re-run entity/relationship extraction on just the changed document, not the whole corpus.
- **Diff & merge:** Compare newly extracted entities/relationships against what's currently in the graph for that document's provenance — add new ones, remove ones no longer supported, and **carefully handle entities that appear in multiple documents** (don't delete a node just because one source document changed, if other documents still reference it — track provenance per relationship/edge, not just per node).
- **Async pipeline:** Run this as an event-driven job (triggered by document change events) rather than a scheduled full rebuild.
- This mirrors incremental vector maintenance but is trickier because deleting stale info from a shared graph requires provenance tracking (a relationship might be evidenced by multiple documents).

---

### Q49. Prevent Cypher injection when relationship types come from an LLM
**Concept:** Cypher injection — if an LLM-generated string is concatenated directly into a query (especially for relationship type or label, which can't be parameterized like values can), it's a code injection vector similar to SQL injection.

**Simple Answer:** Relationship types and labels in Cypher **cannot be parameterized with normal query parameters** (unlike values), which is exactly what makes this risky if you interpolate an LLM's raw string output directly into the query text. Secure it by: (1) **never using the LLM's raw output as a relationship type directly** — validate it against your **fixed, allow-listed schema** (see Q46) before use, rejecting/mapping anything outside the allowed set; (2) if dynamic type names are unavoidable, use a strict allow-list regex (alphanumeric + underscore only, no special Cypher syntax characters) before interpolation; (3) treat the LLM as an **untrusted input source**, exactly like user input — apply the same input-validation discipline you'd apply to a web form, since the LLM can be prompted/manipulated (or simply hallucinate) into producing malformed or malicious strings.


## Chapter 8: GraphRAG and Hybrid Graph Retrieval

### Q50. Design a GraphRAG system for supply chain risk analysis
**Concept:** GraphRAG combines a knowledge graph (for relational/multi-hop reasoning) with an LLM for synthesis — ideal when the core question is inherently relational ("who is exposed to what").

**Simple Answer:**
- **Graph schema:** Nodes = `Supplier`, `Product`, `Customer`, `Facility`; relationships = `SUPPLIES`, `DEPENDS_ON`, `LOCATED_IN`, `PURCHASES_FROM`.
- **Ingestion:** Extract these entities/relationships from supplier contracts, shipment records, and structured ERP data feeds.
- **Query flow:** User asks "which customers are exposed if Supplier X fails?" → parse into a graph traversal (find all products depending on Supplier X → find customers purchasing those products, multi-hop) → the graph engine returns the exact affected set (deterministic, auditable) → an LLM formats this into a readable risk narrative, optionally pulling supporting text context (e.g., contract terms) via vector search for extra detail.
- **Why graph, not just vector RAG:** This is fundamentally a **traversal/aggregation** question, not a similarity-search question — vector RAG can't reliably enumerate "all customers N hops away," but a graph query can do this exactly and completely.
- **Freshness:** Supply chain relationships change often — design for incremental updates (Q48) driven by ERP/contract change events.

---

### Q51. Explain local search and global search, and when each fails
**Concept:** In GraphRAG (Microsoft's framework terminology), **local search** answers questions about specific entities using their immediate neighborhood in the graph; **global search** answers broad, corpus-wide questions using pre-computed community summaries.

**Simple Answer:**
- **Local search:** Given a query mentioning specific entities, retrieve those entities' nodes plus their nearby relationships/neighbors and related text chunks, then generate an answer. Good for: "What projects is Employee X working on?" **Fails when:** the question is broad/aggregate ("What are the main themes across all incident reports?") — there's no single entity to anchor the search, so local search has nothing to start from.
- **Global search:** Uses hierarchical **community summaries** (the graph is clustered into communities via community-detection algorithms, and each community is pre-summarized by an LLM) to answer broad, thematic questions by combining summaries across relevant communities (map-reduce style). **Fails when:** the question needs precise, specific facts about one entity — community summaries are lossy abstractions, so fine-grained detail gets averaged away, and it's expensive/slow (must query across many community summaries) for what should be a simple lookup.
- **Mechanism, not just intuition:** Local search = **graph traversal from a query-matched entity**; global search = **querying pre-aggregated summaries of graph partitions**. Choosing wrong means either "no anchor to search from" (local on broad question) or "answer is vague/expensive" (global on a specific question).

---

### Q52. GraphRAG returns empty context for half of all queries
**Concept:** Entity linking failure — the pipeline needs to map a natural-language query to specific graph entities before it can traverse; if this linking step fails silently, retrieval returns nothing even though the graph itself is fine.

**Simple Answer:** "Traversals work when I test them manually" (i.e., when you supply the exact entity/node) but fail for real queries points to a broken **entity extraction/linking step**: the system can't reliably map the user's natural-language query into the correct graph node(s) to start traversal from. Likely causes: (1) the query entity-extraction step (NER on the user's question) is missing or too strict, failing on paraphrases/typos/partial names; (2) **no fallback to vector search** when entity linking fails — a good hybrid design should fall back to semantic search over node/chunk embeddings when a direct entity match isn't found, rather than returning empty. Fix: add fuzzy/semantic entity linking (embed entity names, match via similarity, not exact string match) and always fall back to vector retrieval over the graph's associated text when graph-anchor entities can't be confidently identified.

---

### Q53. A 3-hop query with a LIMIT clause brought down Neo4j — why didn't LIMIT protect you?
**Concept:** In Cypher, `LIMIT` is applied to the **final result set**, not to intermediate traversal steps — an unconstrained multi-hop `MATCH` can still explode combinatorially before the limit is ever applied.

**Simple Answer:** `LIMIT` only trims the number of rows **returned at the end** of the query — it doesn't stop the database from fully computing (and holding in memory) every possible path during the multi-hop `MATCH` traversal first. In a densely connected graph, a 3-hop traversal without additional constraints (e.g., relationship type filters, node property filters, or a path-length cap enforced *during* traversal) can produce an explosively large intermediate result set — each hop can multiply the candidate paths, so 3 hops through a moderately connected graph can generate millions of intermediate rows before `LIMIT` ever gets a chance to cut it down, exhausting memory/CPU. Fix: constrain the traversal itself (specific relationship types, node label filters, a bounded-length path pattern), and/or use `LIMIT` earlier per-hop (e.g., limit candidates at each intermediate step, not just the final output), plus set query timeout/memory guards at the database level.

---

### Q54. Hybrid GraphRAG or advanced vector RAG for customer support?
**Concept:** Customer support queries are typically single-document, single-topic lookups — the value of relational/multi-hop reasoning (GraphRAG's strength) is usually low here.

**Simple Answer:** **Pick advanced vector RAG** (hybrid dense+lexical retrieval + reranking) for most customer support assistants. Justification: support questions ("how do I reset my password," "what's your return policy") are almost always answerable from a single relevant document/FAQ entry — they don't require multi-hop relational reasoning across entities. GraphRAG's added complexity (extraction pipeline, schema maintenance, entity resolution) isn't justified unless support questions specifically involve **relationships between entities** (e.g., "which of my orders are affected by the recall on Product X" — genuinely relational). Unless failure analysis of a vector RAG system specifically shows multi-hop questions being missed, GraphRAG is over-engineering for this use case — advanced vector RAG gets better ROI per engineering-hour here.

---

### Q55. Keep community summaries fresh without full recomputation (graph changes hourly, recompute takes 6 hours)
**Concept:** Incremental community detection/summarization — full graph clustering (e.g., Louvain/Leiden algorithms) is expensive to recompute from scratch, but most hourly changes only affect a small, localized part of the graph.

**Simple Answer:**
- **Localize the update:** Most hourly changes touch a small subset of nodes/edges — instead of re-running community detection on the whole graph, identify only the **communities affected** by the changed nodes/edges (their direct neighborhood) and re-cluster/re-summarize just those.
- **Incremental clustering algorithms:** Some community-detection approaches support incremental updates (adjusting existing community assignments rather than recomputing from scratch) — use these where available.
- **Tiered freshness:** Keep community summaries at "good enough" staleness for most communities (e.g., refresh daily/on a schedule) but trigger immediate targeted refresh only for communities with high-impact or high-change-volume nodes.
- **Full recompute cadence:** Reserve the expensive full 6-hour recomputation for periodic (e.g., nightly/weekly) consistency reconciliation, not every hourly update.

---

### Q56. Why does concatenating graph and vector context often make answers worse?
**Concept:** Combining two retrieval modalities without reconciling format/redundancy/relevance creates prompt bloat and conflicting signal — more context isn't automatically better context (ties back to "lost in the middle," Q24).

**Simple Answer:** Even if both graph facts (structured, terse) and vector chunks (unstructured prose) are individually relevant, naively concatenating them: (1) creates **redundancy** — the same fact may appear in both a graph triple and a text chunk, wasting context budget and diluting attention; (2) creates a **format mismatch** — the LLM has to context-switch between terse structured facts and flowing prose, which can confuse synthesis; (3) **increases total prompt length**, pushing relevant information toward the middle (Q24's lost-in-the-middle effect) and increasing the chance of contradiction if the graph and text sources are out of sync (e.g., stale graph vs. updated document). Fix: **fuse, don't concatenate** — deduplicate overlapping facts, structure the prompt with clear sections (e.g., "Structured facts:" vs. "Supporting context:"), and only include graph facts that add information the vector chunks don't already cover.


## Chapter 9: Agentic GraphRAG and Multi-Hop Reasoning

### Q57. Design an agentic research assistant, 2–4 retrieval steps, under 15s, no infinite loops
**Concept:** Agentic RAG wraps retrieval in a reasoning loop (plan → retrieve → evaluate → decide next step or answer), requiring explicit loop-termination controls.

**Simple Answer:**
- **Loop structure:** Plan (decompose the question into sub-questions) → retrieve for the current sub-question → evaluate whether enough evidence has been gathered → either retrieve again (different sub-question/refined query) or synthesize a final answer.
- **Never loop forever:** Enforce a **hard step cap** (e.g., max 4 retrieval iterations) and a **wall-clock timeout** (e.g., abort and answer with best-available evidence at 12s, leaving buffer under the 15s budget) — both are non-negotiable guardrails, not soft suggestions.
- **Latency management:** Run independent sub-question retrievals in parallel where possible (if the plan identifies non-dependent sub-questions); use a fast, smaller model for the planning/routing steps and reserve the larger model for final synthesis.
- **Fallback:** If the step cap or timeout is hit without full resolution, return a partial answer with a clear caveat rather than erroring out or hanging.

---

### Q58. Explain LangGraph state reducers and why they matter
**Concept:** In LangGraph, state reducers define **how updates from different nodes get merged into the shared graph state**, which matters critically when multiple nodes execute concurrently and write to the same state key.

**Simple Answer:** LangGraph models an agent as a graph of nodes that read and write a shared state object. When a node returns an update to a state field, the **reducer function** determines how that update combines with the existing value — e.g., the default reducer *overwrites* the old value, while a custom reducer (like `add` for lists) *appends* to it instead. This matters for concurrency: if two nodes run in parallel and both write to the same state key with a plain overwrite reducer, one node's update **silently clobbers** the other's — you lose data non-deterministically depending on execution order. Using an appropriate reducer (e.g., a list-append reducer for something like "retrieved_chunks," so parallel retrieval branches all contribute rather than overwrite each other) makes concurrent state updates safe and deterministic.

---

### Q59. Agent loops until the cap on most queries
**Concept:** Agent looping without value-add usually means the "should I stop" decision logic isn't actually detecting sufficiency — it's a self-evaluation/stopping-criterion failure.

**Simple Answer:** Likely cause: the agent's "do I have enough information to answer" evaluation step is **poorly calibrated** — it's either too conservative (never confident enough to stop, so it always retrieves again until hitting the cap) or it's not actually reasoning about *evidence sufficiency* at all (e.g., it's just checking "did retrieval return non-empty results," which is almost always true, so it never has a real stop signal). Diagnose: log the agent's reasoning/decision at each step and check what triggers continuation. Fix: give the stopping-criterion prompt an explicit, concrete rubric ("Do you have enough evidence to answer with confidence X? Answer only YES/NO with justification") rather than an open-ended judgment call, and validate that rubric against known-answerable-in-1-hop test cases to confirm it can actually recognize "done."

---

### Q60. Agent gave different answers to the same question twice
**Concept:** LLM non-determinism (sampling temperature, race conditions in tool calls, or non-deterministic retrieval ordering) compounds in agentic loops because each step's output feeds the next step's input, amplifying small variations.

**Simple Answer:** Causes: (1) **temperature > 0** at any step in the loop (planning, retrieval query generation, or synthesis) introduces randomness, and because agentic loops are multi-step, small variations early (e.g., a slightly different sub-question phrasing) can cascade into a **different retrieval path entirely** by the final step — this is amplification, not just noise; (2) if any external state (e.g., a live database, or an index that changed between calls) differs between the two calls, that alone can cause a different answer; (3) tie-breaking in retrieval (two chunks with identical similarity scores returned in different order) can subtly shift context. Control it: set temperature to 0 (or low) for the planning/decision steps in the loop (keep it higher only for final natural-language synthesis if desired), pin/fix any non-deterministic tie-breaking in retrieval (deterministic sort), and log full traces so you can compare exactly where two runs diverged.

---

### Q61. Fixed pipeline, router, or full agent — choose and defend
**Concept:** These represent increasing complexity/flexibility trade-offs: fixed pipeline (cheap, predictable, limited), router (adds query-type-specific handling), full agent (most flexible, most expensive/unpredictable).

**Simple Answer:** For a general enterprise knowledge assistant, a **router** (query classifier + a small set of specialized pipelines) is usually the right starting point: it handles the reality that different questions need different strategies (simple lookup vs. multi-hop) without the cost/latency/non-determinism risk of a full open-ended agent. Defend it: a **fixed pipeline** underperforms on the subset of genuinely complex/multi-hop questions enterprise users do ask; a **full agent** (open-ended tool use, unbounded reasoning) is harder to test, more expensive, more latency-variable, and prone to loop/non-determinism issues (Q59, Q60) — overkill for the majority of queries which are simple lookups. A router captures most of the agent's benefit (right strategy per query type) with far more predictability and lower cost — upgrade specific routes to agentic behavior only where evaluation proves it's needed.

---

### Q62. How do you test an agentic workflow with non-deterministic outputs?
**Concept:** Testing shifts from exact-match assertions to **property-based and rubric-based evaluation** — checking that outputs satisfy certain criteria/invariants rather than matching one fixed expected string.

**Simple Answer:**
- **Property-based assertions:** Instead of "output == expected_string," check properties: does the final answer cite the correct source document(s)? Does it stay within the step/time budget? Does it avoid banned behaviors (e.g., never fabricate a citation)?
- **LLM-judge scoring:** Use an LLM judge (validated per Q37) to score answer quality/faithfulness against a rubric across many runs, rather than requiring identical text.
- **Trace-level testing:** Test individual nodes/steps in isolation (deterministic unit tests on the retrieval node, the planning node, etc.) separately from full end-to-end runs — isolate what *is* deterministic.
- **Run multiple samples:** Run the same test case N times (e.g., 5–10) and check that the **distribution** of outcomes meets a bar (e.g., ≥90% of runs correctly cite the source), rather than expecting 100% identical outputs.
- **Regression on failure modes:** Every production failure discovered becomes a new property-based test case added to the suite over time.

---

### Q63. Where does human-in-the-loop belong, and how to implement without blocking?
**Concept:** HITL should be inserted at high-risk decision points, not the whole pipeline, and should be async (non-blocking) so throughput isn't destroyed.

**Simple Answer:** Place human review at the **specific high-stakes decision points**, not the entire flow — e.g., before an action with real-world consequence (approving a refund, sending a regulated communication) or when the agent's own confidence is low. **Implementation without blocking the whole system:** design the agent to produce a "draft decision + confidence score" and **queue** low-confidence or high-risk cases for async human review (e.g., a review dashboard/ticket queue) while allowing high-confidence, low-risk cases to proceed automatically — the agent doesn't halt globally waiting on a human; only the specific flagged case sits in a queue while other requests continue processing normally. For escalation paths in regulated use cases, also log a full audit trail (what evidence, what confidence, why escalated) so human reviewers have full context to make a fast decision.


## Chapter 10: Production RAG and GraphRAG System Design Interviews

### Q64. Design a RAG platform for 50M documents, 2,000 QPS
**Concept:** Full-scale production system design — combines everything from earlier chapters (chunking, hybrid retrieval, caching, incremental indexing) at high throughput, plus infra-level concerns (sharding, autoscaling, cost control).

**Simple Answer:**
- **Ingestion:** Distributed, event-driven pipeline (queue + workers), incremental updates (Q13/Q48), dedup via content hashing.
- **Indexing:** Sharded vector index (e.g., across multiple nodes/collections by tenant or document-type partition) using ANN (HNSW), plus a lexical (BM25/Elasticsearch) index for hybrid search.
- **Retrieval path (per query):** Query embedding → parallel dense + lexical search → RRF fusion → optional reranking (cut under load) → context assembly.
- **Caching:** Semantic query cache for repeated/similar queries, embedding cache, and a CDN-like cache layer for extremely common questions — critical at 2,000 QPS to avoid recomputing everything.
- **Generation:** Load-balanced pool of LLM inference endpoints, streaming responses, request queueing/backpressure for load spikes.
- **Scale numbers:** At 2,000 QPS, budget infra for horizontal autoscaling of retrieval and generation tiers independently (they have very different bottlenecks — retrieval is I/O-bound, generation is GPU-bound).
- **Build last:** Advanced features like GraphRAG, agentic multi-hop reasoning, and fine-grained personalization — get the core retrieve-generate loop rock-solid and observable first; add complexity only where evaluation proves it's needed.

---

### Q65. p95 latency doubled overnight, no deployment, flat traffic — diagnose
**Concept:** Latency regressions without a code change point to external/environmental causes: infra degradation, data growth, or a dependency change.

**Simple Answer:** Check in order: (1) **Index growth** — did the corpus silently grow (e.g., a bulk ingestion job ran overnight), pushing the vector index past a size threshold where ANN search naturally slows (more candidates to traverse)? (2) **Upstream dependency change** — did the embedding API provider or LLM API provider have a silent latency regression on their end (check their status page)? (3) **Infra-level issue** — a node failure causing uneven load on remaining shards, a caching layer that got flushed/expired (cold cache = way slower), or a scheduled job (backup, reindex, compaction) competing for the same resources overnight. (4) **Query pattern shift** — even with flat *volume*, the *type* of queries might have shifted overnight (e.g., a batch job sending more complex queries), increasing average generation length/context size. This is a "nothing changed in code" bug, so the fault is almost certainly in data volume, external dependencies, or infra — check dashboards/logs for exactly what changed at the time latency jumped.

---

### Q66. Board asks why the RAG project costs 3x its forecast
**Concept:** Cost overruns in RAG projects usually trace to a few recurring categories: underestimated LLM token usage, embedding/reindexing costs, and infra scaling that wasn't modeled — presenting this well means quantifying each driver, not just apologizing.

**Simple Answer:** **Analysis approach:** Break down actual spend into categories (LLM inference tokens, embedding compute, vector DB/infra, engineering time) and compare each to the original forecast to identify exactly where the 3x came from — common culprits are (a) underestimating tokens-per-query (long contexts, conversational history accumulation — see Q25), (b) more frequent reindexing/re-embedding than planned (e.g., due to more frequent document updates than modeled), (c) reranking or agentic loops added post-launch that weren't in the original cost model, (d) low cache hit rates driving redundant compute. **The plan:** Present concrete cost-reduction levers tied to the diagnosis — e.g., prompt/context trimming, semantic caching, moving to a cheaper model tier for simple queries (via the router pattern, Q34/Q61), and a revised, more accurate forecasting model going forward that ties cost directly to usage metrics (queries/day, avg tokens/query) rather than a rough estimate.

---

### Q67. Enforce document-level access control in retrieval
**Concept:** Access control in RAG must happen at the **retrieval** layer (filtering which chunks are even eligible to be retrieved), not just at the UI/application layer — otherwise the LLM could be given (and could leak) unauthorized content.

**Simple Answer:** Attach **permission metadata** (e.g., allowed roles/user IDs/groups) to every chunk at ingestion time. At query time, **apply the permission filter as part of the vector search itself** (metadata pre-filtering supported by most vector DBs), so unauthorized chunks are excluded *before* they're ever retrieved or seen by the LLM — never filter *after* generation, since by then the model may have already used/leaked restricted content in its answer. For complex permission models (e.g., hierarchical roles, document-specific ACLs), resolve the user's effective permission set once per request (cache it) and pass it as a filter clause. Also test this explicitly: since a filter bug here is a security incident (like Q17, but security-critical), include automated tests that verify a user literally cannot retrieve content outside their permission scope.

---

### Q68. Local models or hosted API models for enterprise deployment?
**Concept:** This trade-off spans cost, control, latency, compliance, and operational burden — the "right" answer depends on data sensitivity, scale, and team capability, not a blanket preference.

**Simple Answer:** **Favor local/self-hosted models when:** data sensitivity/compliance requires data to never leave your infrastructure (regulated industries, contractual data residency requirements), query volume is high enough that self-hosting is cheaper at scale than per-token API pricing, or you need full control over model versioning (no surprise upstream changes). **Favor hosted API models when:** the team is small and can't justify GPU infra/ops overhead, you need access to frontier model capability that isn't available to self-host, query volume is variable/bursty (pay-per-use avoids over-provisioning idle GPUs), or time-to-market matters more than long-run unit economics. **Justify with more than preference:** run the actual cost math (self-hosting GPU/ops cost vs. API cost at your projected query volume) and weigh it against compliance requirements — for most mid-size enterprises without strict data-residency constraints, hosted APIs win on total cost of ownership until volume is very high; for regulated industries, local models may be a hard requirement regardless of cost.

---

### Q69. Design the rollout plan for replacing keyword search with RAG (10,000 users)
**Concept:** Large-scale system replacement needs a gradual, reversible rollout with clear success criteria at each stage — never a hard cutover for a system this many people depend on daily.

**Simple Answer:**
1. **Shadow mode:** Run the new RAG system alongside the existing keyword search, logging RAG's answers but showing users only the old system — compare quality/coverage offline without any user-facing risk.
2. **Opt-in beta:** Let a small, willing subset of users (e.g., power users, early adopters) opt into RAG with an easy fallback to keyword search, and collect explicit feedback plus behavioral metrics (task success, thumbs-down rate).
3. **Gradual rollout by cohort:** Expand to larger % of users incrementally (e.g., 5% → 25% → 50% → 100%), monitoring quality/latency/cost metrics at each stage with an explicit rollback trigger if metrics regress.
4. **Keep keyword search available as a fallback** throughout the rollout (e.g., an "advanced search" toggle) — this reduces risk and builds trust, since users who don't get a satisfying RAG answer can always fall back to what they know works.
5. **Full cutover + deprecation:** Only after RAG demonstrably matches/exceeds keyword search on key metrics across the full user base, sunset the old system on a communicated timeline.
6. **Change management:** Communicate the transition to users in advance, provide guidance on how RAG-based search differs (e.g., natural language questions vs. keywords) since user behavior/expectations need to adapt too.

---

### Q70. What would you monitor, and what does each alert mean? (on-call, live system)
**Concept:** A production RAG monitoring stack spans infra health, pipeline-stage health, and quality/business metrics — each alert should map to a specific likely cause so on-call response is fast.

**Simple Answer:**
- **Latency (p50/p95/p99) per pipeline stage** (embedding, retrieval, reranking, generation): alert on SLA breach → tells you *which stage* regressed, narrowing root cause immediately (compare to Q65).
- **Error rate / timeout rate:** alert on spike → likely infra issue (index node down, LLM API outage) or a bad deploy.
- **Retrieval similarity score distribution:** alert on a sustained drop in average top-1 score → signals embedding/index drift, a new unindexed query pattern, or an ingestion pipeline failure (new docs not being indexed).
- **Refusal/abstention rate:** alert on spike → could mean retrieval quality dropped (see above) or a new class of out-of-scope questions is arriving.
- **Faithfulness score (sampled):** alert on drop → generation-side regression (prompt change, model swap) — ties to Q38/Q39.
- **User feedback (thumbs-down) rate:** alert on spike → catch-all signal for any quality regression not caught by automated metrics; always investigate transcripts when this fires.
- **Cost/token usage per query:** alert on sudden increase → context bloat (Q25) or a routing bug sending simple queries to an expensive path.
- **Index size / ingestion lag:** alert if ingestion falls behind → users are querying against stale/incomplete data.
Each alert should point on-call toward a **specific stage of the pipeline**, so response starts with "which metric fired" rather than a blind investigation from scratch.

---

## Quick-Reference: Core Concepts Glossary

| Concept | One-Line Definition |
|---|---|
| Chunking | Splitting documents into retrievable units, sized empirically via recall testing |
| Embedding | Converting text to a vector; models are not interchangeable (different vector spaces) |
| HNSW | Graph-based ANN index enabling fast approximate similarity search |
| Hybrid Retrieval | Combining dense (semantic) + lexical (BM25) search for better coverage |
| RRF (Reciprocal Rank Fusion) | Combines ranked lists by rank position, avoiding score-scale mismatches |
| Reranking | Cross-encoder re-scoring of a candidate set for higher precision, at added latency cost |
| Lost in the Middle | LLMs recall info best at the start/end of context, worst in the middle |
| Faithfulness | Whether an answer is actually supported by the retrieved context |
| Abstention | Deliberately refusing to answer when evidence confidence is too low |
| Corrective RAG | Evaluates/fixes retrieval quality before generation |
| Knowledge Graph | Entities + relationships stored for multi-hop/relational queries |
| GraphRAG Local Search | Traversal from a specific matched entity for targeted questions |
| GraphRAG Global Search | Uses pre-computed community summaries for broad, thematic questions |
| Agentic RAG | A reasoning loop around retrieval (plan → retrieve → evaluate → repeat/answer) with explicit stopping criteria |
| Index-Free Adjacency | Why graph DBs beat SQL joins for multi-hop traversal |
