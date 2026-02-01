from app.config.settings import config 
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel


#variables 
llm_model_name_1= config.OPENAI_MODEL
llm_model_name_2= config.OPENAI_MODEL

# Define models : currently not using implementing retry logic
primary = OpenAIChatModel(llm_model_name_1)
secondary = OpenAIChatModel(llm_model_name_2)

# Initialize FallbackModel
# It will try 'primary' first, then 'secondary' if the first fails
fallback_model = FallbackModel(primary, secondary)
agent = Agent(fallback_model)


## todo 
## add retries logic , but  not  idk i dont want to add boilerplate code 