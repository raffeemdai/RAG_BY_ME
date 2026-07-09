# LangChain Basics: Chaining, Prompts, Messages & First Experiment

---

## 1. Chaining — The Core Idea

The most basic way to talk to an LLM is:

```
prompt → llm → response
```

LangChain lets you **chain** these pieces together using the pipe operator `|` — same mental model as Unix pipes, where the output of one step feeds directly into the next.

```python
chain = prompt | llm
```

**Simple explanation:** think of it like an assembly line. The prompt template fills in your question → passes it to the LLM → the LLM's answer comes out the other end.

### Wanting structured output (JSON)
If you don't just want a plain-text answer but a structured one, you add another link to the chain:

```python
chain = prompt | llm | json_parser
```

**Simple explanation:** the raw LLM response is just text. A parser sits at the end of the chain and converts that text into something usable in code — like a Python dictionary — instead of you having to manually clean up the string yourself.

### What is LCEL?
This `|` notation is called **LCEL — LangChain Expression Language**. It's LangChain's syntax for composing components (prompts, models, parsers, retrievers, tools) into a pipeline, where every piece speaks the same interface (`invoke`, `stream`, `batch`) so they can be linked together with `|`.

**Simple explanation:** LCEL is just a clean, readable way of saying "do this, then this, then this" — instead of writing lots of manual glue code, you snap components together like Lego blocks.

---

## 2. Prompt — What You Send to the Model

A **prompt** is the input given to the LLM. Underneath, LLMs recognize different *types* of prompts, based on role:

| Type | Purpose | Simple explanation |
|---|---|---|
| **System Prompt** | Defines the model's role/behavior | Like giving an employee their job description before they start work — "You are a customer support agent, only answer politely and from company docs." |
| **User Prompt** | The actual question being asked | This is you, the user, typing your actual question. |

When you send this prompt to the model, you get a **response** back.

---

## 3. Messages — LangChain's Way of Organizing a Conversation

A real conversation isn't just one prompt and one response — it can go back and forth, involve tools, etc. LangChain generalizes every piece of a conversation into a **Message** object. There are **4 core types**:

| LangChain Class | Represents | Simple explanation |
|---|---|---|
| `SystemMessage` | System Prompt | The "instructions" given once, before the conversation starts |
| `HumanMessage` | User Prompt | What the human typed |
| `AIMessage` | LLM's response | What the model replied with |
| `ToolMessage` | Result of a tool call | If the LLM asked to use a tool (like a calculator or search), this carries the tool's answer back to the LLM |

**Why bother with these classes instead of plain strings?** Different LLM providers (OpenAI, Gemini, Anthropic) each expect conversation history in slightly different formats behind the scenes. LangChain's Message classes are a common, provider-agnostic way to represent a conversation — write it once, and LangChain translates it for whichever model you're using.

---

## 4. Hands-On Experiment: "hello_llms" — Your First LLM Call

### Step 1 — Create a new project folder using `uv`
`uv` is a fast Python package/project manager (an alternative to pip + venv).

```bash
mkdir hello_llms
cd hello_llms
uv init .
```

This scaffolds a basic Python project (creates `pyproject.toml`, `.python-version`, etc.).

### Step 2 — Add the LangChain package
```bash
uv add langchain
```
(`uv add` is like `pip install`, but it also records the dependency in your project's `pyproject.toml`.)

### Step 3 — Get an API key for Gemini
Go to [Google AI Studio](https://aistudio.google.com/apps) and generate an API key — this is how your code authenticates to use Google's Gemini models.

### Step 4 — Install `python-dotenv`
```bash
uv add python-dotenv
```
This package lets your code securely load secrets (like API keys) from a `.env` file instead of hardcoding them.

### Step 5 — Create a `.env` file
```
GEMINI_API_KEY='paste your api key here'
```
**Why:** keeps your API key out of your actual source code — important for security and for not accidentally committing secrets to GitHub.

### Step 6 — Install the Gemini integration package
```bash
uv add langchain-google-genai
```
This is the specific LangChain package that knows how to talk to Google's Gemini models. (Reference: [LangChain Google Generative AI docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai))

### Step 7 — Write the first working code

Final `pyproject.toml` dependencies looked like:
```toml
[project]
name = "hello-llms"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "langchain>=1.3.11",
    "langchain-google-genai>=4.2.6",
    "python-dotenv>=1.2.2",
]
```

`main.py`:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from .env into environment variables

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

prompt = HumanMessage(content='What is capital of France?')

response = llm.invoke([prompt])
response.pretty_print()
```

**execute the above main.pycode :**
uv run main.py

**Line-by-line, simple explanation:**
1. `load_dotenv()` — loads your `.env` file so `GEMINI_API_KEY` becomes available to the code.
2. `ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")` — creates the LLM object, pointing at a specific Gemini model.
3. `HumanMessage(content='...')` — wraps your question in LangChain's standard message format (from Section 3 above).
4. `llm.invoke([prompt])` — sends the message list to the model and gets back an `AIMessage`.
5. `response.pretty_print()` — nicely formats and prints the model's answer to the console.

### Step 8 — Try chaining (LCEL)
Now that you have a working `llm` object, you can wrap it in LCEL syntax instead of calling `.invoke()` manually:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful geography assistant."),
    ("user", "What is the capital of {country}?")
])

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"country": "France"})
print(result)
```

**Simple explanation:** instead of manually building a `HumanMessage` and calling `.invoke()` yourself, the `ChatPromptTemplate` fills in the blank (`{country}`), pipes it to the `llm`, and `StrOutputParser()` extracts clean text from the response — all in one readable line: `prompt | llm | parser`.

---

## 5. Quick Recap (Interview-Ready)

- **LCEL** (`|`) chains LangChain components together — prompt → llm → parser — because they all implement the same `Runnable` interface.
- **Prompt** = input to the LLM, made of a **system** part (role/behavior) and a **user** part (the actual ask).
- **Messages** are LangChain's typed, provider-agnostic way of representing a conversation: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`.
- Practical setup for a first LLM project: `uv init` → `uv add langchain` → get API key → `python-dotenv` for secrets → `langchain-google-genai` for the Gemini integration → `load_dotenv()` + `ChatGoogleGenerativeAI` + `HumanMessage` + `.invoke()`.
- Once basic invocation works, LCEL (`prompt | llm | parser`) is the natural next step to make prompts reusable and outputs cleanly structured.
