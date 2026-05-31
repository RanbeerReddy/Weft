from db.database import engine
from models.conversation import Conversation
from models.message import Message
from models.chunk import Chunk

from db.database import Base

Base.metadata.create_all(engine)

print("Database initialized")