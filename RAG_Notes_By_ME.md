-   LLM is a Deep learning model which doesnot understand text.

# ![Preview](media/image1.png){width="6.5in" height="3.12in"}

![](media/image2.png){width="5.604166666666667in" height="5.229166666666667in"}

![](media/image3.png){width="6.302083333333333in" height="4.5in"}

![](media/image4.png){width="5.677083333333333in" height="5.666666666666667in"}

The model then samples (or picks the top) from this distribution to choose the actual next word --- and the whole process repeats with the new, longer sequence to generate the word after that.

# **Vectors 27/Jun/2026**

-   A Vector is larger dimensional point

-   Vectors are organized in a way as shown below

king - man + women = Queen

distance between king and queen ~= distance between man and women

-   A specialized databases for storing and searching data in this vector notations is called as Vector database.

![Preview](media/image5.png){width="6.541666666666667in" height="3.323167104111986in"}

![Preview](media/image6.png){width="6.5in" height="3.792413604549431in"}

![](media/image7.png){width="6.3141852580927385in" height="5.913828740157481in"}

![](media/image8.png){width="6.7977624671916015in" height="5.4867661854768155in"}

# **Chaining 29/Jun/2026**

<https://directai.blog/2026/06/29/gen-ai-developer-classroom-notes-29-jun-2026/>

<https://github.com/raffeemdai/RAG_BY_ME/blob/main/RAG-2-langchain-chaining-prompts-messages-notes.md> —NOTES BY ME

![](media/image9.png){width="6.26875in" height="2.8625in"}

-   When we interact with llm basic usage is

-   prompt

-   llm

-   we can chain these things

prompt | llm

-   I'm interacting with llm but i want response in json structure

prompt | llm | jsonstrcture

-   The above notations are referred as LCEL

## **PROMPT**

-   PROMPT is an input to the model

-   The underlying LLM deals with different types of prompts

    -   SYSTEM PROMPT => define the role

    -   USER PROMPT => this is where we ask a question

-   When we pass this prompt we get a response

### **MESSAGES**

-   When we interact with llm we will have different types of prompts or responses

    -   System Prompt

    -   User Prompt

    -   Response from llm

-   To generalize this langchain does this. we have 4 types of messages

    -   SystemMessage => (System Prompt)

    -   HumanMessage => User Prompt

    -   AIMessage => llm response

    -   ToolMessage

![](media/image10.png){width="6.26875in" height="3.183333333333333in"}

**execute the above main.pycode :**

uv run main.py

uv commands

![](media/image11.png){width="6.26875in" height="2.770138888888889in"}

![](media/image12.png){width="6.1875in" height="3.8958333333333335in"}

![](media/image13.png){width="5.989583333333333in" height="5.489583333333333in"}

# LangChain: Prompts & Messages --- Study Notes

## 1. PROMPT --- The Basics

-   A **prompt** is the input given to an LLM.

-   Under the hood, LLMs distinguish between different *types* of prompts (this is literally how chat-based APIs are structured --- OpenAI, Anthropic, Gemini all use this same role-based format):

| **Type** | **Purpose** |
| --- | --- |
| **System Prompt** | Defines the model's *role*, behavior, constraints, and persona. Sets the rules before any conversation starts. Example: "You are a helpful HR assistant. Only answer from the provided policy documents." |
| **User Prompt** | The actual question or instruction from the user. Example: "How many sick leaves am I entitled to?" |

-   When this prompt (system + user) is sent to the model, we get back a **response**.

**Key insight:** the system prompt isn't "extra" --- it's how you control tone, restrict scope, prevent hallucination, and enforce output format, all *before* the user even types anything.

## 2. MESSAGES --- LangChain's Generalized Abstraction

Real conversations have more moving parts than just "prompt in, response out" --- especially once tools/agents are involved. LangChain generalizes every piece of a conversation into a **Message** object. There are **4 core message types**:

| **LangChain Class** | **Maps to** | **Role string** | **Meaning** |
| --- | --- | --- | --- |
| SystemMessage | System Prompt | "system" | Defines LLM's role/behavior for the whole conversation |
| HumanMessage | User Prompt | "user" | What the human/user says or asks |
| AIMessage | LLM Response | "assistant" | What the model generated back |
| ToolMessage | Tool/Function output | "tool" | The **result returned by a tool** that the LLM decided to call, fed back into the conversation so the LLM can use it to form its final answer |

### About ToolMessage specifically

When an LLM is given tools (function calling / agents), it doesn't always answer directly --- it can respond with a request to call a tool (e.g., "call get_weather(city='Atlanta')"). The **actual output of running that tool** (e.g., "72°F, sunny") gets wrapped in a ToolMessage and appended back into the message history, tagged with a tool_call_id linking it to the specific call the model requested. The LLM then reads this ToolMessage on the next turn and uses it to produce a final natural-language answer.

This is the mechanism that makes **RAG's SQL agent** (structured data retrieval) and any **agentic tool-calling** workflow possible --- the LLM doesn't run code itself, it just asks for a tool call via AIMessage, your code executes it, and the result comes back as a ToolMessage.

## Why LangChain Does This (The Point)

Different LLM providers format conversation history differently under the hood (OpenAI's JSON schema ≠ Anthropic's ≠ Gemini's). LangChain's Message classes are a **provider-agnostic abstraction** --- write your conversation once using SystemMessage/HumanMessage/AIMessage/ToolMessage, and LangChain translates it into whatever format the underlying model API expects. This is what lets you swap ChatOpenAI for ChatVertexAI or ChatAnthropic without rewriting your conversation logic.

## **Prompt Templates vs. Messages**

Don't confuse these --- they solve different problems:

| **Concept** | **Purpose** |
| --- | --- |
| PromptTemplate / ChatPromptTemplate | **Reusable, parameterized** prompt with placeholders ({variable}) --- used to *generate* messages dynamically from user input |
| SystemMessage / HumanMessage / etc. | The actual **message objects** sent to the LLM --- the output of formatting a template, or manually constructed |

# **Runnable 30jun 2026**

30th june 2026 class : <https://directai.blog/2026/06/30/gen-ai-developer-classroom-notes-30-jun-2026/>

2nd July/2026 -- <https://directai.blog/2026/07/02/gen-ai-developer-classroom-notes-02-jul-2026/>

<https://github.com/raffeemdai/RAG_BY_ME/blob/main/RAG-3-4%20langchain_runnable_notes%20GCP%20setup%20with%20langchain%20n%20prompt%20template%20%20outputparser.md> — it has notes for 30jun and and 2nd jul

4th July

## **Document and Document Loaders**

<https://directai.blog/2026/07/04/gen-ai-developer-classroom-notes-04-jul-2026/>

<https://github.com/raffeemdai/RAG_BY_ME/blob/main/RAG-5%20-langchain_documents_loaders_notes.md>

<https://github.com/raffeemdai/RAG_BY_ME/blob/main/RAG_6_Chunking_Notes.md> —notes prepared by me

-   Document is an object that refers to individual doc or part of the document.

-   [Refer Here](https://reference.langchain.com/python/langchain-core/documents/base/Document) for the class reference of document object

-   Generally we dont create document objects we load the document objects from files or urls

-   Document Loaders help in reading from various sources and return document objects

-   [Refer Here](https://docs.langchain.com/oss/python/integrations/document_loaders) for document loaders

## 1. What is a Document?

Think of a Document as a simple **container with two things**:

1.  page_content --- the actual text. The actual textual data parsed from a PDF, document, or web page. This string is what gets converted into embeddings.

2.  metadata --- a dictionary of extra info *about* that text (source URL, author, page number, date, etc.) .A dictionary containing contextual information such as the source file name, page number, author, or timestamp. This is heavily used to filter documents before a similarity search is performed.

That's it. It's just a labeled box of text. Everything else in LangChain's RAG workflow (loaders, splitters, retrievers) works by producing or consuming Document objects.

![](media/image14.png){width="6.26875in" height="1.5277777777777777in"}

![](media/image15.png){width="6.26875in" height="1.9104166666666667in"}

![](media/image16.png){width="6.26875in" height="2.24375in"}

### **Lets Experiment**

-   Create a text file and store it data/intro.txt

-   Add a pacakge called as langchain_community

uv add langchain_community

-   Lets use text loader

-   Lets create a pdf file and try reading that.

-   using some loaders will work when we install external packages

-   Lets create a csv loader

-   Lets load from directory with files

-   [Refer Here](https://github.com/GenAIDevelopment/agenticai/blob/main/june26/rag/rag_learning_gcp/documents_document_loaders.ipynb) for notebook

![](media/image17.png){width="6.26875in" height="3.2465277777777777in"}

![](media/image18.png){width="6.26875in" height="2.43125in"}

![](media/image19.png){width="6.26875in" height="2.4180555555555556in"}

![](media/image20.png){width="6.26875in" height="5.051388888888889in"}

**Chunking or Splitting** 6th July:

<https://directai.blog/2026/07/06/gen-ai-developer-classroom-notes-06-jul-2026/>

<https://github.com/raffeemdai/RAG_BY_ME/blob/main/RAG_6_Chunking_Notes.md>

-   IN RAG we deal with two models

-   large language model (llm)

-   embedding model

-   Embedding models convert text into vector which we store in vector database.

-   For effective retrieval, it better if we chunk or split the document into multiple chunks.

-   chunk in langchain will be document.

-   For chunking [Refer Here](https://docs.langchain.com/oss/python/integrations/splitters)

-   Chunking stragies

![Preview](media/image21.gif){width="6.947759186351706in" height="8.912892607174102in"}

-   Chunking also uses overlap (genrally 15 to 20 percent overlap)


#### 1**) Fixed-size chunking**

The most intuitive and straightforward way to generate chunks is by splitting the text into uniform segments based on a pre-defined number of characters, words, or tokens.

![https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2026/01/https-3a-2f-2fsubstack-post-media-s3-amazonaws-com-2fpublic-2fimages-2f98c422a0-f0e2-457c-a256-4476a56a601f_943x232-1.png](media/image22.png){width="6.504650043744532in" height="1.6001902887139108in"}

Since a direct split can disrupt the semantic flow, it is recommended to maintain some overlap between two consecutive chunks (the blue part above).

This is simple to implement. Also, since all chunks are of equal size, it simplifies batch processing.

But there is a big problem. This usually breaks sentences (or ideas) in between. Thus, important information will likely get distributed between chunks.

#### **2) Semantic chunking**

The idea is simple.

![https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2026/01/https-3a-2f-2fsubstack-post-media-s3-amazonaws-com-2fpublic-2fimages-2fa6ad83a6-2879-4c77-9e49-393f16577aef_1066x288-1.gif](media/image23.gif){width="6.26875in" height="1.6936209536307962in"}

-   Segment the document based on meaningful units like sentences, paragraphs, or thematic sections.

-   Next, create embeddings for each segment.

-   Let's say I start with the first segment and its embedding.

    -   If the first segment's embedding has a high cosine similarity with that of the second segment, both segments form a chunk.

    -   This continues until cosine similarity drops significantly.

    -   The moment it does, we start a new chunk and repeat.

Here's what the output could look like:

![https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2026/01/https-3a-2f-2fsubstack-post-media-s3-amazonaws-com-2fpublic-2fimages-2f74037e11-362d-4ea2-8ee2-ee85ab013523_963x231-1.png](media/image24.png){width="6.765157480314961in" height="1.624106517935258in"}

Unlike fixed-size chunks, this maintains the natural flow of language and preserves complete ideas.

Since each chunk is richer, it improves the retrieval accuracy, which, in turn, produces more coherent and relevant responses by the LLM.

A minor problem is that it depends on a threshold to determine if cosine similarity has dropped significantly, which can vary from document to document.

#### **3) Recursive chunking**

This is also simple.

![https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2026/01/https-3a-2f-2fsubstack-post-media-s3-amazonaws-com-2fpublic-2fimages-2ff4009caa-34fc-48d6-8102-3d0f6f2c1386_1066x316-1.gif](media/image25.gif){width="6.443297244094488in" height="1.9072003499562555in"}

First, chunk based on inherent separators like paragraphs, or sections.

Next, split each chunk into smaller chunks if the size exceeds a pre-defined chunk size limit. If, however, the chunk fits the chunk-size limit, no further splitting is done.

Here's what the output could look like:

![https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2026/01/https-3a-2f-2fsubstack-post-media-s3-amazonaws-com-2fpublic-2fimages-2fb0e40cc1-996f-48f4-9306-781b112536e4_984x428-1.png](media/image26.png){width="6.64338801399825in" height="2.8908169291338583in"}

As shown above:

-   First, we define two chunks (the two paragraphs in purple).

-   Next, paragraph 1 is further split into smaller chunks.

Unlike fixed-size chunks, this approach also maintains the natural flow of language and preserves complete ideas.

However, there is some extra overhead in terms of implementation and computational complexity.

**Trade-off:** Like semantic chunking, it preserves the natural flow of language, but adds some extra implementation and computational overhead.

#### **4) Document structure-based chunking**

This is another intuitive approach.

![https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2026/01/https-3a-2f-2fsubstack-post-media-s3-amazonaws-com-2fpublic-2fimages-2fe8febecd-ee68-42ff-ab06-41a0a3a43cd3_1102x306-1.gif](media/image27.gif){width="6.63499343832021in" height="1.8445034995625547in"}

It utilizes the inherent structure of documents, like headings, sections, or paragraphs, to define chunk boundaries.

This way, it maintains structural integrity by aligning with the document's logical sections.

Here's what the output could look like:

![https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2026/01/https-3a-2f-2fsubstack-post-media-s3-amazonaws-com-2fpublic-2fimages-2f40bdaf3b-601d-4357-bc7f-89b47f812097_1025x663-1.png](media/image28.png){width="4.921346237970254in" height="3.1818963254593178in"}

That said, this approach assumes that the document has a clear structure, which may not be true.

Also, chunks may vary in length, possibly exceeding model token limits. You can try merging it with recursive splitting.

#### **5) LLM-based chunking**

![https://storage.ghost.io/c/3f/df/3fdf6ed2-17ac-4b12-a693-8078bd13e748/content/images/2026/01/https-3a-2f-2fsubstack-post-media-s3-amazonaws-com-2fpublic-2fimages-2f4d1b6d60-8956-4030-8525-d899ee61a9d5_1140x198-1.gif](media/image29.gif){width="6.26875in" height="1.0877318460192476in"}

Since every approach has upsides and downsides, why not use the LLM to create chunks?

The LLM can be prompted to generate semantically isolated and meaningful chunks.

Quite evidently, this method will ensure high semantic accuracy since the LLM can understand context and meaning beyond simple heuristics (used in the above four approaches).

The only problem is that it is the most computationally demanding chunking technique of all five techniques discussed here.

Also, since LLMs typically have a limited context window, that is something to be taken care of.

Each technique has its own advantages and trade-offs.

I have observed that semantic chunking works pretty well in many cases, but again, you need to test.

The choice will heavily depend on the nature of your content, the capabilities of the embedding model, computational resources, etc.

We shall be doing a hands-on demo of these strategies pretty soon.

In the meantime, in case you missed it, yesterday, we discussed techniques to build robust NLP systems that rely on pairwise content similarity (RAG is one of them).

![](media/image30.png){width="5.864583333333333in" height="4.291666666666667in"}

### **Experiment with chunking**

-   Prompt

Popular embeddings by different providers, their token limit and vector dimensions in a tabular form

-   Lets add a package

uv add langchain-text-splitters

-   Chunking visualizer [Refer Here](https://chunkviz.up.railway.app/)

-   Note in reality we prefer loading and splitting as one operation

-   Exercise: While we load and split, i want to add additional metadata

-   ingested_date = "Current date"

-   project = "learning"

-   [Refer Here](https://github.com/GenAIDevelopment/agenticai/blob/main/june26/rag/rag_learning_gcp/documents_document_loaders.ipynb) for the notebook

# **Embeddings 07/Jul/2026**

<https://directai.blog/2026/07/07/gen-ai-developer-classroom-notes-07-jul-2026/>

<https://github.com/raffeemdai/RAG_BY_ME/blob/main/RAG_7_Embeddings_Notes.md>

<https://www.pinecone.io/learn/vector-database/>

vvv imp note : sir suggested to gothrough python class for design patterns

in python concerete implementation n method template pattern class july 7th 20206

<https://directai.blog/2026/07/07/python-classroom-notes-07-jul-2026/>

-   Embedding embed the meaning in the form of the vector (a mathematical point)

-   Early idea of Embedding was done by a library called as [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)

-   Embedding will have the whole vocabulary

-   Prompt

IN simple timeline viewer show me evolution of embeddings

-   Embedding models

-   Opensource embedding models

-   Embedding models as service (Cloud)

-   Langchain Embeddings the base class Embedding which is in langchain-core. embeddings which has two simple methods

-   embed_query

-   embed_documents

## **Vector Database (Vector Store)**

-   Once we have vectors we need to store them [Refer Here](https://www.pinecone.io/learn/vector-database/) to understand vector databases

-   Prompt

-   Compare with Relational Database and Explain the organization and operations possible on vector databases![](media/image31.png){width="6.052083333333333in" height="5.770833333333333in"}

-   Langchain vector databases integration [Refer Here](https://docs.langchain.com/oss/python/integrations/vectorstores)

-   Exercise -> Indexing Pipeline:

    -   Create a folder with a text files

    -   Each text file contains information about some city

    -   Ensure text is atleast 200 lines organized as paragraphs

    -   Use RecursiveCharacterSplitter

    -   Load and split => documents

    -   Use text-embedding-005 and chroma to store in vector database in some folder.

    -   Ensure every chunk has the following metdata

        -   city: <>

        -   chunk-id:

    -   Vector database => ./data/example1

# **First RAG 08thJuly 2026**

<https://directai.blog/2026/07/08/gen-ai-developer-classroom-notes-08-jul-2026/>

<https://github.com/GenAIDevelopment/agenticai/blob/58564156046b7c3e3f506f6766c9f376232b0623/june26/rag/rag_learning_gcp/simple_rag.py>
