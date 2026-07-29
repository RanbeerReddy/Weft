import json
from datetime import datetime

from Weft.storage.database import SessionLocal
from Weft.storage.models import Conversation, Message
from Weft.utils.exceptions import WeftException


def ts_to_dt(ts):
    try:
        if ts is None:
            return None

        return datetime.fromtimestamp(ts)
    except WeftException as e:
        raise WeftException("Failed to convert timestamp to datetime", e)


def get_content(msg):
    try:
        if not msg:
            return ""

        content = msg.get("content")

        if not content:
            return ""

        parts = content.get("parts", [])

        return "\n".join(str(x) for x in parts if x)
    except WeftException as e:
        raise WeftException("Failed to extract content from message", e)


def parse_export(path):
    try:
        db = SessionLocal()

        with open(path, "r", encoding="utf8") as f:
            conversations = json.load(f)

        for convo in conversations:
            conversation = Conversation(
                id=convo["id"],
                title=convo.get("title"),
                create_time=ts_to_dt(convo.get("create_time")),
                update_time=ts_to_dt(convo.get("update_time")),
                current_node=convo.get("current_node"),
                model_slug=convo.get("default_model_slug"),
                is_archived=convo.get("is_archived", False),
            )

            db.merge(conversation)

            mapping = convo["mapping"]

            for node_id, node in mapping.items():
                message = node.get("message")

                if not message:
                    continue

                db.merge(
                    Message(
                        id=node_id,
                        conversation_id=convo["id"],
                        parent_id=node.get("parent"),
                        role=message["author"]["role"],
                        content=get_content(message),
                        create_time=ts_to_dt(message.get("create_time")),
                    )
                )

        db.commit()

        db.close()
    except WeftException:
        pass


if __name__ == "__main__":
    path = "conversations.json"
    parse_export(path)
