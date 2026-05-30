CREATE TABLE conversation (
    id UUID PRIMARY KEY,
    title TEXT,
    create_time TIMESTAMP,
    update_time TIMESTAMP
);

CREATE TABLE message (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    role VARCHAR(50),
    content TEXT,
    create_time TIMESTAMP,

    CONSTRAINT fk_conversation
        FOREIGN KEY (conversation_id)
        REFERENCES conversation(id)
        ON DELETE CASCADE
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    message_id UUID NOT NULL,
    conversation_id UUID NOT NULL,
    chunk_order INTEGER,
    chunk_text TEXT,

    CONSTRAINT fk_message
        FOREIGN KEY (message_id)
        REFERENCES message(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_conversation_chunk
        FOREIGN KEY (conversation_id)
        REFERENCES conversation(id)
        ON DELETE CASCADE
);

CREATE INDEX idx_conversation_id
ON messages (conversation_id);

CREATE INDEX idx_chunk_conversation
ON chunks (conversation_id);