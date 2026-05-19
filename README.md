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