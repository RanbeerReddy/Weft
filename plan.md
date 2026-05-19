 use both for Similarity and graph rag for retrival.
 use similarity for global memory like memories, custom interactions. things which would apply globally.
 use graphRAG for convo's
 
                ┌────────────────────┐
                │  AI Chat Exports   │
                │  Obsidian Vault    │
                │  GitHub Repos      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Ingestion Pipeline │
                │  Parsing + Chunk   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Local Storage     │
                │ Markdown + SQLite  │
                └──────┬─────┬───────┘
                       │     │
             ┌─────────┘     └─────────┐
             ▼                         ▼
    ┌────────────────┐       ┌─────────────────┐
    │ Vector Index   │       │ Semantic Graph  │
    │ sqlite-vec     │       │ triples/links   │
    └────────┬───────┘       └────────┬────────┘
             │                        │
             └──────────┬─────────────┘
                        ▼
              ┌──────────────────┐
              │ Retrieval Engine │
              │ Hybrid Search    │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ MCP Server/API   │
              │ Context Provider │
              └────────┬─────────┘
                       ▼
      ┌────────────────────────────────┐
      │ Cursor / Claude / VSCode / AI │
      └────────────────────────────────┘








chatgpt-export.zip
├── conversations.json        ← Full chat history (main file)
├── message_feedback.json     ← Thumbs up/down and written feedback
├── model_comparisons.json    ← Side-by-side model comparison votes
├── user.json                 ← Account info (name, email, creation date)
├── chat.html                 ← Human-readable version of all chats
├── dalle/                    ← Folder with DALL·E-generated images
└── tts/                      ← Folder with text-to-speech audio (if used)   






What a Serious AI Memory System Would Extract

A strong pipeline would derive:

Raw Layer
Original messages
Timestamps
Attachments

Semantic Layer
Topics
Entities
Intent
Tasks
Decisions

Temporal Layer
Long-term interests
Evolving goals
Recurring projects

Knowledge Layer
Concepts learned
Research threads
Skill progression

Behavioral Layer
Writing style
Thinking patterns
Preferred depth
Learning gaps

That is where your project becomes much more interesting than “just RAG over chats”.





What You Actually Need To Parse
1. conversations.json → REQUIRED

This is the core.

You need:

message content
timestamps
author role
conversation title
branching structure
attachments references
model metadata

Without this file your system is mostly useless.

2. files/ → VERY IMPORTANT

Especially for your project idea.

Why?

Because uploaded files are part of the user’s cognitive context.

Example:

research PDFs
resumes
code
datasets
notes
screenshots

You should:

hash them
extract text
embed separately
link back to conversations

This becomes HUGE for contextual memory systems.

3. custom_instructions.json → IMPORTANT

This contains:

persistent user behavior
preferences
communication style
goals

This is basically:

“global system prompt of the human”

Extremely valuable.

4. memories.json → IMPORTANT

This is critical if memory existed.

You can compare:

explicit memory
vs
inferred memory from conversations

Very interesting research direction.