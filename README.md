````md
# Weft

> Turning ChatGPT conversations into a searchable, structured second brain.

## What is this?

Weft is an open-source system for reconstructing, organizing, and understanding exported ChatGPT conversations.

The goal is not just to store chats.

The goal is to transform thousands of scattered AI conversations into:
- a knowledge graph
- a research archive
- a personal context engine
- a second brain
- and eventually an AI memory system

Instead of conversations dying inside chat windows, Weft tries to make them:
- searchable
- interconnected
- reusable
- analyzable
- and useful for long-term thinking.

---

# Why I’m Building This

Every serious AI user eventually hits the same problem:

- thousands of conversations
- buried ideas
- forgotten insights
- repeated questions
- no structure
- no continuity

ChatGPT remembers almost nothing across time.

Humans don’t either.

So the idea is:

> What if conversations could become persistent knowledge?

Not bookmarks.

Not notes.

Actual evolving context.

---

# Vision

Weft aims to become a system that can:

- Parse exported ChatGPT conversations
- Reconstruct conversation trees
- Extract ideas, topics, projects, and entities
- Build semantic search over conversations
- Connect related thoughts automatically
- Generate knowledge graphs
- Create long-term AI memory
- Integrate with Obsidian and local knowledge systems
- Enable AI-assisted reflection and research

Eventually:

- AI agents that understand your history
- Context-aware assistants
- Personalized reasoning systems
- Self-organizing research archives

---

# Core Idea

Chat logs are not just messages.

They are:
- thinking traces
- learning history
- project evolution
- research logs
- decision timelines
- idea graphs

Weft treats conversations like structured knowledge instead of disposable text.

---

# Current Features

## Parsing ChatGPT Export Data
- Reads exported `conversations-000.json`
- Reconstructs branching conversations
- Preserves message hierarchy and metadata

## Obsidian Integration
- Converts chats into markdown
- Builds vault-ready structures
- Enables backlinking and note navigation

## Conversation Reconstruction
- Restores message flow from node mappings
- Creates readable timelines
- Preserves assistant/user roles

## Local Knowledge Storage
- Fully local-first workflow
- Your data stays yours
- No dependency on cloud vector DBs

---

# Planned Features

## Semantic Search
Find ideas instead of keywords.

## Embedding Pipelines
Vectorize conversations for contextual retrieval.

## Knowledge Graphs
Automatically connect:
- projects
- concepts
- people
- research topics
- recurring patterns

## AI Context Engine
Provide historical context to local LLMs and agents.

## Memory Compression
Summarize years of conversations into reusable knowledge.

## Multi-Source Integration
Future support for:
- Claude exports
- Gemini chats
- emails
- notes
- PDFs
- research papers

---

# Tech Stack

## Backend
- Python 3.11
- FastAPI

## Data Processing
- Pandas
- Pydantic
- NetworkX

## NLP / AI
- Sentence Transformers
- FAISS / ChromaDB
- LangChain (possibly)
- local embeddings

## Frontend (planned)
- React
- TypeScript

## Knowledge Layer
- Obsidian
- Markdown
- Graph-based linking

---

# Philosophy

Weft is built around a few ideas:

### AI conversations are valuable data
Most people waste them.

### Knowledge should compound
Ideas should connect over time.

### AI should enhance thinking
Not replace it.

### Local-first matters
Your thoughts should belong to you.

### Context is everything
Intelligence without memory is shallow.

---

# Long-Term Goal

The long-term goal is to build something between:
- a second brain
- a research operating system
- and a persistent AI memory layer

A system where:
- your ideas evolve over years
- AI understands your projects
- context accumulates instead of disappearing

---

# Current Status

Early-stage experimental project.

Right now the focus is:
1. understanding ChatGPT export structure
2. reconstructing conversations correctly
3. building clean markdown pipelines
4. designing scalable architecture
5. preparing for semantic retrieval and AI memory systems

---

# Example Workflow

```bash
Export ChatGPT data
        ↓
Parse conversation JSON files
        ↓
Reconstruct conversation trees
        ↓
Convert to markdown
        ↓
Store in Obsidian
        ↓
Generate embeddings
        ↓
Semantic search + knowledge graph
        ↓
AI memory/context engine
````

---

# Project Goals

* Build something genuinely useful
* Learn deeply instead of tutorial-copying
* Understand AI memory systems
* Explore knowledge engineering
* Create infrastructure for future AI agents
* Open-source the entire process

---

# Inspiration

Inspired by:

* second brain systems
* knowledge graphs
* semantic search
* AI memory architectures
* personal knowledge management
* research workflows
* long-term thinking systems

---

# Contributing

Still heavily experimental.

But if you care about:

* AI memory
* knowledge systems
* semantic retrieval
* local-first AI
* personal context engines
* Obsidian workflows

feel free to explore, fork, or contribute.

---

# Final Thought

Most AI conversations disappear.

Weft exists because maybe they shouldn’t.

```
```
