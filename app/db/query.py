from app.llm.embedding import get_embedding
from app.db.client import ensure_index
from loguru import logger
import asyncio
from app.config.settings import config

INDEX_NAME = config.DEFAULT_INDEX_NAME
NAMESPACE_CONTEXT = "ContextLibrary"

async def query_pinecone(index, query_text, namespace, top_k=1):
    """Embeds the query text and searches the specified Pinecone namespace."""
    try:
        query_embedding = await get_embedding(query_text)
        response = index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        return response['matches']
    except Exception as e:
        logger.info(f"Error querying Pinecone (Namespace: {namespace}): {e}")
        raise e






# async def test_query():
#     # Ensure index exists
#     index = ensure_index(INDEX_NAME)

#     # Query Pinecone
#     result = await query_pinecone(index, "ujjwal basnet", NAMESPACE_CONTEXT)
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(test_query())
