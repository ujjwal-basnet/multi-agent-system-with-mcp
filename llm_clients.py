from settings import config
from openai import OpenAI 
from langchain_openai import ChatOpenAI



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
