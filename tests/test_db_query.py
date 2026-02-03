from app.llm.embedding import get_embedding
from app.db.client import ensure_index
from app.config.settings import config
import asyncio

INDEX_NAME = config.DEFAULT_INDEX_NAME
NAMESPACE_CONTEXT = "ContextLibrary"


def test_query():
    """Test function for Pinecone query."""
    async def _test():
        index = ensure_index(INDEX_NAME)
        query_embedding = await get_embedding("ujjwal basnet")
        result = index.query(
            vector=query_embedding,
            namespace=NAMESPACE_CONTEXT,
            top_k=3,
            include_metadata=True
        )
        print("Query results:", result)

    asyncio.run(_test())


if __name__ == "__main__":
    test_query()
