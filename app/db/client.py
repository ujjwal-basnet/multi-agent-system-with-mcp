import time
from app.config.settings import config
from pinecone import Pinecone, ServerlessSpec
from loguru import logger

pc = Pinecone(api_key=config.PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

EMBEDDING_DIM = config.OPENAI_EMBEDDING_DIM


def ensure_index(INDEX_NAME : str ):
    if INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Index '{INDEX_NAME}' not found. Creating new serverless index")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=spec,
        )

        while not pc.describe_index(INDEX_NAME).status["ready"]:
            logger.info("Waiting for index to be ready")
            time.sleep(1)

        logger.info("Index created successfully")

    else:
        logger.info(f"Index '{INDEX_NAME}' already exists")

    return pc.Index(INDEX_NAME)


# def test_ensure_index():
#     index_name = config.DEFAULT_INDEX_NAME
#     index = ensure_index(index_name)

#     logger.info(f"Test passed: index '{index_name}' is ready")
    
    
# test_ensure_index()