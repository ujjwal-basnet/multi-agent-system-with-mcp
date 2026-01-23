from settings import config
from openai import OpenAI 
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage , HumanMessage 
from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
from pydantic import BaseModel,Field
from typing import List


mimo_v2_flash_client = ChatOpenAI(
    model="xiaomi/mimo-v2-flash:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=config .OPENROUTER_API_KEY,
)


openai_client= OpenAI(
  api_key= config.OPENAI_API_KEY,
)

# response= openai_client.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=[
#                             {
#                                 "role": "user",
#                                 "content": "How many r's are in the word 'strawberry'?"
#                             }
#                             ],
#  )


from langchain_google_genai import ChatGoogleGenerativeAI


# Google LLM client
google_langchain_client = ChatGoogleGenerativeAI(
    model=config.GEMINI_MODEL,
    google_api_key=config.GEMINAI_API_KEY2,
)



from pinecone import Pinecone
# Initialize a Pinecone client with your API key
pine_cone_client = Pinecone(api_key=config.PINECONE_API_KEY)


from langchain_cohere import ChatCohere,CohereEmbeddings
cohere_chat_client = ChatCohere( cohere_api_key=config.COHERE_API)


from langchain_cohere import CohereEmbeddings
cohere_embedding_client = CohereEmbeddings(model="embed-english-light-v3.0" , cohere_api_key= config.COHERE_API)




## cohere does't has json output in api level so forcing it though retry , pydantic methods

class ResearcherOutput(BaseModel):
    intent_query: str = Field(description="descriptive phrase summarizing desired, style , tone or formating optimized for searching query")
    topic_query:str = Field(description="Concise phrases summarizing the factual subject matter" )



@retry(
                        stop=stop_after_attempt(3),
                        wait=wait_random_exponential(min=2, max=10))
def call_llm(system_prompt: str, user_content: str, structured=False) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    
    
    if not structured:
        return cohere_chat_client.invoke(messages).content
    
    structured_llm = cohere_chat_client.with_structured_output(ResearcherOutput)
    prompt = f"{system_prompt}\n\n{user_content}"
    return structured_llm.invoke(prompt)



# ### robust llm 

# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(min=2, max=10),
#     retry=retry_if_exception_type(Exception),
# )
# def call_primary(messages):
#     return google_langchain_client.invoke(messages)

# def call_llm(system_prompt: str, user_content: str) -> str:
#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_content),
#     ]

#     try:
#         return call_primary(messages).content
#     except Exception:
#         return cohere_embedding_client.invoke(messages).content