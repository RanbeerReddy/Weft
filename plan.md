 use both for Similarity and graph rag for retrival.
 use similarity for global memory like memories, custom interactions. things which would apply globally.
 use graphRAG for convo's

think of properly utilizing the question and answer in your chunking quality

------------------------------------------------------------------------------------------------------

Research Perspective

Potentially strong.

Especially if you explore:

memory compression
memory consolidation
memory importance scoring
forgetting mechanisms
episodic memory
semantic memory

This starts looking closer to AI memory research.

Much more interesting than generic RAG.



-----------------------------------------------------------------------------------------------------------------

Why would anyone use this?

Current answer:

Search old chats.

Not enough.

Interesting answers:

Understand how I changed over time
Reconstruct decisions
Track goals
Remember commitments
Surface forgotten insights
Build a second brain
Long-term AI memory

Now we're talking.


------------------------------------------------------------------------------------------------------------




 
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


Load all conversation shards
    ↓
Merge into unified dataset
    ↓
Extract active branch
    ↓
Resolve attachments
    ↓
Normalize messages
    ↓
Embed/store/index





for reconstructing chats:
extract_data
|
reconstructing_chats.py



for the real workflow:
database.py
|
models.py ---->(.createall(bind=base))
|
parser.py














-------------------------------------------------------------------------------------------------------
Step 1: Fix Retrieval Before Building Anything Else

Not GraphRAG.

Not agents.

Not APIs.

Not frontend.

Your current retrieval quality is unknown and you've already seen weird results.

The next goal should be:

"When I ask a question about my chat history, do I consistently get the right messages back?"

What I would build this week
Add retrieval evaluation

Create a file:

tests/retrieval_eval.py

Make 20-50 test queries.

Example:

[
    {
        "query": "what do i think about RAG applications",
        "expected_conversation": "RAG discussion"
    },
    {
        "query": "what internship did i get selected for",
        "expected_keyword": "Sensovibe"
    },
]

Then measure:

Top 1 hit
Top 5 hit
Top 10 hit

Right now you don't know if retrieval is 20% accurate or 80% accurate.

Step 2: Fix Chunking

After seeing your schema, I think your biggest issue is:

Message
↓
Split into 1000 char chunks

For conversations this is often wrong.

Try:

1 message = 1 chunk

first.

Seriously.

Don't overcomplicate.

Current message sizes are probably small enough.

Test:

Approach A:
1000-char chunks

vs

Approach B:
Message chunks

Evaluate both.

I suspect message-level retrieval wins.

Step 3: Store Better Metadata

Update Chunk table.

Add:

role
timestamp
conversation_title

or derive them during retrieval.

Current retrieval only knows:

chunk_text

That's weak.

Step 4: Add Hybrid Search

After retrieval evaluation exists.

Do:

Vector Search
+
Keyword Search

In PostgreSQL.

Not GraphRAG.

Not Neo4j.

Something like:

vector_score
+
tsvector score

Hybrid retrieval is one of the biggest practical improvements you'll ever make.

Step 5: Add Reranking

This is where quality jumps.

Flow:

Query
↓
Vector Search
↓
Top 50
↓
Cross Encoder
↓
Top 10

Models:

cross-encoder/ms-marco-MiniLM-L-6-v2

or

BAAI/bge-reranker-base

This is probably worth more than GraphRAG.

Step 6: Build Memory Extraction

Now we're entering interesting territory.

Create a new table:

Memory

Example:

id
type
value
source_message_id
confidence
created_at

Example extraction:

Message:

I am preparing for GATE 2027

Memory:

{
  "type": "goal",
  "value": "GATE 2027"
}

Message:

I got selected for Sensovibe internship

Memory:

{
  "type": "experience",
  "value": "Sensovibe Internship"
}

Now you're building a memory system.

Step 7: Build Search API

Only after retrieval works.

FastAPI.

One endpoint.

/search

Input:

{
  "query": "what internships have i discussed"
}

Output:

{
  "results": [...]
}

That's enough.

Don't build 20 endpoints.

Step 8: Conversation Reconstruction

This is your hidden advantage.

You already store:

parent_id

Use it.

When retrieving a message:

Retrieved Message
+
Parent
+
Child

This gives context.

Huge improvement.