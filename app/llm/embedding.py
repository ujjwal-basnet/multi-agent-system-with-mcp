from app.config.settings import config
from pydantic_ai import  Embedder
from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger

# Variables
Embedding_Model= config.OPENAI_EMBEDDING_MODEL 
Embedding_Dim= config.OPENAI_EMBEDDING_DIM 


#Embedder 
embedder = Embedder(f'openai:{Embedding_Model}')


@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(2))
async def get_embedding(data) -> list: 
    
    ## in both case returning list for predictability . i.e outputt of this will always list 
    if isinstance(data, str ):
        result = await embedder.embed_query(data) 
        logger.info(f'Embedding dimensions: {len(result.embeddings[0])}')
        return [result.embeddings[0]]
    
    if isinstance(data, list):
        result= await embedder.embed_documents(data)
        logger.info(f'Embedding dimensions: {len(result.embeddings)}')
        return result.embeddings 
    
    else : 
        raise ValueError ("Only string or list is supported currently") 
    


#################### ends #################### 



# async def main():
#     docs = ['Machine learning is a subset of AI.', 'hi hi how are you']
#     embeddings = await get_embedding(docs)
#     print(len(embeddings))


# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())