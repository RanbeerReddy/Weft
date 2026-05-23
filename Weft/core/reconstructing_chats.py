from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any

# =========================================================
# CONFIG
# =========================================================
INPUT_FILES = [
    "Data/Extracted data/conversations-000.json",
    "Data/Extracted data/conversations-001.json",
    "Data/Extracted data/conversations-002.json",
]
MERGED_FILE = "conversations.json"
OUTPUT_DIR = Path("vault/conversations")

WRITE_INDIVIDUAL_MESSAGES = True
WRITE_FULL_CONVERSATION = True


# =========================================================
# HELPERS
# =========================================================
def sanitize_filename(name: str) -> str:
    name = re.sub(r"[<>:\"/\\\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:120]


def unix_to_iso(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts).isoformat()
    except Exception:
        return "unknown"


def extract_message_text(message: dict[str, Any]) -> str:
    if not message:
        return ""
    content = message.get("content", {})
    parts = content.get("parts", [])
    cleaned_parts = [p for p in parts if isinstance(p, str)]
    return "\n".join(cleaned_parts).strip()


def get_author_role(message: dict[str, Any]) -> str:
    return message.get("author", {}).get("role", "unknown")


# =========================================================
# CORE LOGIC
# =========================================================
def reconstruct_conversation(conversation: dict[str, Any]) -> None:
    conv_id = conversation.get("id", "unknown_id")
    title = conversation.get("title", "untitled")
    safe_title = sanitize_filename(title)

    conversation_dir = OUTPUT_DIR / f"{safe_title}_{conv_id}"
    messages_dir = conversation_dir / "messages"

    conversation_dir.mkdir(parents=True, exist_ok=True)
    messages_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata = {
        "id": conv_id,
        "title": title,
        "create_time": unix_to_iso(conversation.get("create_time")),
        "update_time": unix_to_iso(conversation.get("update_time")),
        "is_archived": conversation.get("is_archived"),
        "default_model_slug": conversation.get("default_model_slug"),
    }
    with open(conversation_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Parse messages
    mapping = conversation.get("mapping", {})
    parsed_messages = []
    for node_id, node_data in mapping.items():
        message = node_data.get("message")
        if not message:
            continue
        text = extract_message_text(message)
        if not text.strip():
            continue
        parsed_messages.append({
            "id": node_id,
            "role": get_author_role(message),
            "text": text,
            "create_time": unix_to_iso(message.get("create_time")),
            "parent": node_data.get("parent"),
            "children": node_data.get("children", []),
        })
    parsed_messages.sort(key=lambda x: x["create_time"])

    # Individual message files
    if WRITE_INDIVIDUAL_MESSAGES:
        for idx, msg in enumerate(parsed_messages, start=1):
            filename = f"{idx:04d}_{msg['role']}.md"
            file_path = messages_dir / filename

            parent_link = f"[[{msg['parent']}]]" if msg['parent'] else "None"
            children_links = ", ".join([f"[[{child}]]" for child in msg['children']]) if msg['children'] else "None"
            prev_link = f"[[{parsed_messages[idx-2]['id']}]]" if idx > 1 else "None"
            next_link = f"[[{parsed_messages[idx]['id']}]]" if idx < len(parsed_messages) else "None"

            content = f"""# {msg['role'].upper()}

## Metadata
- Message ID: {msg['id']}
- Created: {msg['create_time']}
- Parent: {parent_link}
- Children: {children_links}
- Previous: {prev_link}
- Next: {next_link}

---

{msg['text']}
"""
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    # Full conversation file
    if WRITE_FULL_CONVERSATION:
        conversation_md = [
            f"# {title}",
            "",
            f"**Conversation ID:** `{conv_id}`",
            "",
            f"[[{parsed_messages[0]['id']}|Start of conversation]]",
            "",
            f"[[{parsed_messages[-1]['id']}|End of conversation]]",
            "",
            "---",
            "",
        ]
        for idx, msg in enumerate(parsed_messages, start=1):
            filename = f"{idx:04d}_{msg['role']}.md"
            conversation_md.extend([
                f"## {msg['role'].upper()}",
                "",
                f"**Time:** {msg['create_time']}",
                "",
                f"[[{filename}]]",  # link to message file
                "",
                msg["text"],
                "",
                "---",
                "",
            ])
        with open(conversation_dir / "conversation.md", "w", encoding="utf-8") as f:
            f.write("\n".join(conversation_md))


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    # Merge input files into one
    all_conversations = []
    for file in INPUT_FILES:
        path = Path(file)
        if not path.exists():
            print(f"[!] Missing file: {file}")
            continue
        print(f"[+] Loading {file}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_conversations.extend(data)
            else:
                print(f"[!] Unexpected format in {file}")

    # Write merged file
    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)
    print(f"[+] Merged {len(all_conversations)} conversations into {MERGED_FILE}")

    # Process conversations
    for idx, conversation in enumerate(all_conversations, start=1):
        try:
            reconstruct_conversation(conversation)
            print(f"[{idx}/{len(all_conversations)}] Processed: {conversation.get('title', 'untitled')}")
        except Exception as e:
            print(f"[!] Failed conversation {conversation.get('id')} -> {e}")

    print("\n[+] Reconstruction complete")


if __name__ == "__main__":
    main()
