

def create_mcp_message(sender, content, metadata=None):
    """Creates a standardized MCP message."""
    return {
        "protocol_version": "2.0 (Context Engine)",
        "sender": sender,
        "content": content, # The actual payload/context
        "metadata": metadata or {}
    }
