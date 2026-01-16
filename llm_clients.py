from settings import settings
from openai import OpenAI 
from langchain_openai import ChatOpenAI



mimo_v2_flash_client = ChatOpenAI(
    model="xiaomi/mimo-v2-flash:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)


openai_client= OpenAI(
  api_key= settings.OPENAI_API_KEY,
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
    model=settings.GEMINI_MODEL,
    google_api_key=settings.GEMINAI_API_KEY,
)