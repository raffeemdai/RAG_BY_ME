# LangChain: Prompts & Messages — Study Notes

## 1. PROMPT — The Basics

- A **prompt** is the input given to an LLM.
- Under the hood, LLMs distinguish between different *types* of prompts (this is literally how chat-based APIs are structured — OpenAI, Anthropic, Gemini all use this same role-based format):

| Type | Purpose |
|---|---|
| **System Prompt** | Defines the model's *role*, behavior, constraints, and persona. Sets the rules before any conversation starts. Example: `"You are a helpful HR assistant. Only answer from the provided policy documents."` |
| **User Prompt** | The actual question or instruction from the user. Example: `"How many sick leaves am I entitled to?"` |

- When this prompt (system + user) is sent to the model, we get back a **response**.

**Key insight:** the system prompt isn't "extra" — it's how you control tone, restrict scope, prevent hallucination, and enforce output format, all *before* the user even types anything.

---

## 2. MESSAGES — LangChain's Generalized Abstraction

Real conversations have more moving parts than just "prompt in, response out" — especially once tools/agents are involved. LangChain generalizes every piece of a conversation into a **Message** object. There are **4 core message types**:

| LangChain Class | Maps to | Role string | Meaning |
|---|---|---|---|
| `SystemMessage` | System Prompt | `"system"` | Defines LLM's role/behavior for the whole conversation |
| `HumanMessage` | User Prompt | `"user"` | What the human/user says or asks |
| `AIMessage` | LLM Response | `"assistant"` | What the model generated back |
| `ToolMessage` | Tool/Function output | `"tool"` | The **result returned by a tool** that the LLM decided to call, fed back into the conversation so the LLM can use it to form its final answer |

### About `ToolMessage` specifically
When an LLM is given tools (function calling / agents), it doesn't always answer directly — it can respond with a request to call a tool (e.g., "call `get_weather(city='Atlanta')`"). The **actual output of running that tool** (e.g., `"72°F, sunny"`) gets wrapped in a `ToolMessage` and appended back into the message history, tagged with a `tool_call_id` linking it to the specific call the model requested. The LLM then reads this `ToolMessage` on the next turn and uses it to produce a final natural-language answer.

This is the mechanism that makes **RAG's SQL agent** (structured data retrieval) and any **agentic tool-calling** workflow possible — the LLM doesn't run code itself, it just asks for a tool call via `AIMessage`, your code executes it, and the result comes back as a `ToolMessage`.

---

## 3. Why LangChain Does This (The Point)

Different LLM providers format conversation history differently under the hood (OpenAI's JSON schema ≠ Anthropic's ≠ Gemini's). LangChain's Message classes are a **provider-agnostic abstraction** — write your conversation once using `SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`, and LangChain translates it into whatever format the underlying model API expects. This is what lets you swap `ChatOpenAI` for `ChatVertexAI` or `ChatAnthropic` without rewriting your conversation logic.

---

## 4. Code Example

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful HR assistant. Answer only from company policy."),
    HumanMessage(content="How many sick leaves do I get per year?"),
]

response = llm.invoke(messages)   # returns an AIMessage
print(response.content)

# Continuing the conversation — append the AI's reply and a new user turn
messages.append(response)
messages.append(HumanMessage(content="What about casual leaves?"))

response2 = llm.invoke(messages)
```

### With a tool call
```python
messages = [
    HumanMessage(content="What's the weather in Atlanta?"),
    AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Atlanta"}, "id": "call_123"}]),
    ToolMessage(content="72°F, sunny", tool_call_id="call_123"),
]

final_response = llm.invoke(messages)  # LLM now uses the tool result to answer
```

---

## 5. Prompt Templates vs. Messages

Don't confuse these — they solve different problems:

| Concept | Purpose |
|---|---|
| `PromptTemplate` / `ChatPromptTemplate` | **Reusable, parameterized** prompt with placeholders (`{variable}`) — used to *generate* messages dynamically from user input |
| `SystemMessage` / `HumanMessage` / etc. | The actual **message objects** sent to the LLM — the output of formatting a template, or manually constructed |

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    ("user", "{question}")
])

messages = template.format_messages(domain="HR policy", question="How many sick leaves?")
# This produces a list of SystemMessage + HumanMessage objects, ready to pass to the LLM
```

---

## 6. Quick Interview-Ready Summary

- LLMs fundamentally take a **prompt** → produce a **response**; the prompt is structured into **roles** (system, user).
- LangChain generalizes this role-based structure into 4 **Message classes**: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`.
- This abstraction exists so your code is **portable across LLM providers**.
- `ToolMessage` is the piece that enables **agentic/tool-calling workflows** — it's how tool output re-enters the conversation for the LLM to reason over.
- `PromptTemplate`/`ChatPromptTemplate` generate Messages dynamically; Messages are what actually gets sent via `.invoke()`.
- This all fits into **LCEL** chains: `prompt_template | llm | output_parser` — the template formats into Messages, the LLM returns an `AIMessage`, and the parser extracts/structures the final content.

---

## 7. Likely Interview Questions on This Topic

1. **Why does LangChain use Message objects instead of raw strings?** → Provider-agnostic abstraction; enables consistent multi-turn conversation handling and tool-calling across different LLM APIs.
2. **What's the difference between a System Prompt and a System Message?** → Conceptually the same thing; `SystemMessage` is LangChain's typed object representation of the system prompt.
3. **How does an LLM "call a tool" if it can't execute code?** → It doesn't execute anything — it returns an `AIMessage` containing a structured `tool_calls` field describing what it wants called; your application code executes the actual function and feeds the result back as a `ToolMessage`.
4. **Why maintain a full messages list instead of just the last user message?** → LLMs are stateless between calls — the entire conversation history (all messages) must be resent each time so the model has context of prior turns.
