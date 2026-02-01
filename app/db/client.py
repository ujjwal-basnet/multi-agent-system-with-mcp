# Import the Pinecone library
from openai import api_key
from app.config.settings import config 
from pinecone import Pinecone, ServerlessSpec 
from loguru import logger
import time 

pc= Pinecone(api_key= config.PINECONE_API_KEY)
spec= ServerlessSpec(cloud='aws', region='us-east-1')

INDEX_NAME= config.DEFAULT_INDEX_NAME
NAMESPACE_KNOWLEDGE="KnowledgeStore" 
NAMESPACE_CONTEXT= "ContextLibrary"
EMBEDDING_DIM= config.OPENAI_EMBEDDING_DIM


if INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"{INDEX_NAME} not found... creating new serverless index")
    pc.create_index(
        name=INDEX_NAME, 
        dimension=EMBEDDING_DIM,
        metric='cosine',
        spec=spec
    )
    
    # wait for spec to be ready 
    while not pc.describe_index(INDEX_NAME).status['ready']:
        logger.info(f"Waiting for {INDEX_NAME} index to be ready")
        time.sleep(1)
        
    logger.info("Index created successfully")
    
    
else:
    ## runs only if index exist 
    logger.info(f"Index {INDEX_NAME} already exist")
    logger.info(f"Clearing NewSpace for a fresh start....")
    
    #connect to the index to perform operation 
    index= pc.Index(INDEX_NAME)
    

        
    
    