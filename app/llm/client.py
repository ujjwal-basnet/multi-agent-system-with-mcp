from app.config.settings import config
from pydantic_ai import Agent, NativeOutput, StructuredDict, Embedder
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger 

# Variables
llm_model_name_1 = config.OPENAI_MODEL
llm_model_name_2 = config.OPENAI_MODEL

# Models
primary = OpenAIChatModel(llm_model_name_1)
secondary = OpenAIChatModel(llm_model_name_2)
fallback_model = FallbackModel(primary, secondary)

# JSON schema
AnyJSON = StructuredDict({"type": "object"}, name="AnyJSON")


# llm calling function 
@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(2))
def call_llm_robust(system_prompt: str, user_prompt: str, json_mode: bool = False):
    
    try :
        output_type = NativeOutput(AnyJSON) if json_mode else str
        
        agent = Agent(
            fallback_model,
            instructions=system_prompt,
            output_type=output_type,
            model_settings={"api_key": config.OPENAI_API_KEY}
        )

        # Run the agent with user message
        result = agent.run_sync(user_prompt)
        return result.output
    
    except Exception as e :
        logger.info(f"Error calling llm {e}")
        raise e 
    
    

    
    
        
        





# will add others client too but not now  """








# def test():
#     system_prompt= "you are a helpful assistant "
#     user_prompt= "solve 2+2"
    
#     response_json= call_llm_robust(system_prompt, user_prompt, json_mode=True)
#     response= call_llm_robust(system_prompt, user_prompt, json_mode=False)
    
#     print(response_json)
#     print("*********************************")
#     print(response)
    
# test()