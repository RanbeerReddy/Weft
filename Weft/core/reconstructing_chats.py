from __future__ import annotations
import json

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any

from Weft.utils.exceptions import WeftException

covo_path = "Data/Extracted data/"
convo_files = ["conversations-000.json", "conversations-001.json", "conversations-002.json"]

"""
reconstructing_chats.py

Reconstruct ChatGPT exported conversations into an Obsidian-friendly structure.

Expected input:
    conversations.json

Output:
    vault/
        conversations/
            <conversation_title>_<conversation_id>/
                metadata.json
                conversation.md
                messages/
                    0001_user.md
                    0002_assistant.md
                    ...

Python:
    3.11+
"""


# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = "conversations.json"
OUTPUT_DIR = "vault/conversations"

WRITE_INDIVIDUAL_MESSAGES = True
WRITE_FULL_CONVERSATION = True



from pathlib import Path
import json

convo_path = Path("Data/Extracted data/")
convo_files = [
    "conversations-000.json",
    "conversations-001.json",
    "conversations-002.json"
]

all_conversations = []

for file_name in convo_files:

    file_path = convo_path / file_name

    print(f"[+] Loading {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:

        data = json.load(f)

        if isinstance(data, list):
            all_conversations.extend(data)

        else:
            print(f"[!] Unexpected format in {file_name}")

print(f"[+] Total conversations loaded: {len(all_conversations)}")

# After loading all_conversations
with open(INPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_conversations, f, indent=2, ensure_ascii=False)

print(f"[+] conversations.json written with {len(all_conversations)} conversations")


# =========================================================
# HELPERS
# =========================================================

def sanitize_filename(name: str) -> str:
    """
    Make safe folder/file names.
    """
    name = re.sub(r"[<>:\"/\\\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:120]


def unix_to_iso(ts: float | None) -> str:
    """
    Convert unix timestamp to ISO datetime.
    """
    if ts is None:
        return "unknown"

    try:
        return datetime.fromtimestamp(ts).isoformat()
    except Exception:
        return "unknown"


def extract_message_text(message: dict[str, Any]) -> str:
    """
    Extract readable text from ChatGPT export message structure.
    """
    if not message:
        return ""

    content = message.get("content", {})

    parts = content.get("parts", [])

    if not parts:
        return ""

    cleaned_parts = []

    for part in parts:
        if isinstance(part, str):
            cleaned_parts.append(part)

    return "\n".join(cleaned_parts).strip()


def get_author_role(message: dict[str, Any]) -> str:
    """
    Extract author role.
    """
    author = message.get("author", {})

    role = author.get("role", "unknown")

    return role


# =========================================================
# CORE LOGIC
# =========================================================

def reconstruct_conversation(conversation: dict[str, Any]) -> None:
    """
    Reconstruct a single conversation into Obsidian structure.
    """

    conv_id = conversation.get("id", "unknown_id")
    title = conversation.get("title", "untitled")

    safe_title = sanitize_filename(title)

    conversation_dir = Path(OUTPUT_DIR) / f"{safe_title}_{conv_id}"
    messages_dir = conversation_dir / "messages"

    conversation_dir.mkdir(parents=True, exist_ok=True)
    messages_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # SAVE METADATA
    # -----------------------------------------------------

    metadata = {
        "id": conv_id,
        "title": title,
        "create_time": unix_to_iso(conversation.get("create_time")),
        "update_time": unix_to_iso(conversation.get("update_time")),
        "is_archived": conversation.get("is_archived"),
        "default_model_slug": conversation.get("default_model_slug"),
    }

    metadata_path = conversation_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------
    # PARSE MESSAGE TREE
    # -----------------------------------------------------

    mapping = conversation.get("mapping", {})

    parsed_messages = []

    for node_id, node_data in mapping.items():

        message = node_data.get("message")

        if not message:
            continue

        role = get_author_role(message)

        text = extract_message_text(message)

        if not text.strip():
            continue

        create_time = unix_to_iso(message.get("create_time"))

        parsed_messages.append({
            "id": node_id,
            "role": role,
            "text": text,
            "create_time": create_time,
            "parent": node_data.get("parent"),
            "children": node_data.get("children", []),
        })

    # sort chronologically if possible
    parsed_messages.sort(key=lambda x: x["create_time"])

    # -----------------------------------------------------
    # WRITE INDIVIDUAL MESSAGE FILES
    # -----------------------------------------------------

    if WRITE_INDIVIDUAL_MESSAGES:

        for idx, msg in enumerate(parsed_messages, start=1):

            filename = f"{idx:04d}_{msg['role']}.md"

            file_path = messages_dir / filename

            content = f"""# {msg['role'].upper()}

## Metadata

- Message ID: {msg['id']}
- Created: {msg['create_time']}
- Parent: {msg['parent']}
- Children: {msg['children']}

---

{msg['text']}
"""

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    # -----------------------------------------------------
    # WRITE FULL CONVERSATION MARKDOWN
    # -----------------------------------------------------

    if WRITE_FULL_CONVERSATION:

        conversation_md = [
            f"# {title}",
            "",
            f"**Conversation ID:** `{conv_id}`",
            "",
            "---",
            "",
        ]

        for msg in parsed_messages:

            conversation_md.extend([
                f"## {msg['role'].upper()}",
                "",
                f"**Time:** {msg['create_time']}",
                "",
                msg["text"],
                "",
                "---",
                "",
            ])

        conversation_path = conversation_dir / "conversation.md"

        with open(conversation_path, "w", encoding="utf-8") as f:
            f.write("\n".join(conversation_md))


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find: {INPUT_FILE}"
        )

    conversations = all_conversations
    print(f"[+] Loaded {len(conversations)} conversations")

    for idx, conversation in enumerate(conversations, start=1):

        try:
            reconstruct_conversation(conversation)

            print(
                f"[{idx}/{len(conversations)}] "
                f"Processed: {conversation.get('title', 'untitled')}"
            )

        except Exception as e:
            print(
                f"[!] Failed conversation "
                f"{conversation.get('id')} -> {e}"
            )

    print("\n[+] Reconstruction complete")


if __name__ == "__main__":
    main()