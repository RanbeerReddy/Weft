def reconstruct_conversation(
    conversation
):

    mapping = conversation["mapping"]

    current = conversation[
        "current_node"
    ]

    path = []

    while current:

        node = mapping.get(current)

        if not node:
            break

        path.append(node)

        current = node.get(
            "parent"
        )

    path.reverse()

    return path