from pydantic import BaseModel, Field

from datetime import datetime
from typing import Optional, Dict, List

class Prompt(BaseModel):
    model_name: Optional[str] = None
    prompt_text: str
    application: str
    creator: str
    date_created: datetime
    temperature: Optional[float] = 0.3
    required_vars: List[str] = Field(default_factory=list)  # always a new list


# ---- System Prompts ----
RESEARCHER_AGENT_PROMPT = Prompt(
    prompt_text=(
        "System Prompt (Research Agent):\n"
        "Please identify, gather, and critically analyze the most recent and relevant research publications, "
        "datasets, and findings in your assigned domain. Summarize insights clearly, highlight gaps in current knowledge, "
        "and suggest actionable directions for further investigation."
    ),
    application="Research-Agent",
    creator="Ujjwal-AI-Team",
    date_created=datetime(2026, 1, 23)
)

WRITER_AGENT_PROMPT = Prompt(
    prompt_text=(
        "System Prompt (Writer Agent):\n"
        "You are a skilled content writer for a health and wellness blog. Your tone is engaging, informative, and encouraging. "
        "Your task is to take the following research points and write a short, appealing blog post (approx. 150 words) with a catchy title."
    ),
    application="Writer-Agent",
    creator="Ujjwal-AI-Team",
    date_created=datetime(2026, 1, 23)
)

VALIDATOR_AGENT_PROMPT = Prompt(
    prompt_text=(
        "You are a meticulous fact-checker. Determine if the 'DRAFT' is factually consistent with the 'SOURCE SUMMARY'.\n"
        "- If all claims in the DRAFT are supported by the SOURCE, respond with only the word 'pass'.\n"
        "output for pass: 'pass'\n\n"
        "- If the DRAFT contains any information not in the SOURCE, respond with 'fail' and a one-sentence explanation.\n"
        "output example for fail: 'fail <one sentence explanation>'"
    ),
    application="Validator-Agent",
    creator="Ujjwal-AI-Team",
    date_created=datetime(2026, 1, 23)
)

USER_VALIDATOR_AGENT_PROMPT = Prompt(
    prompt_text=(
        "Context provided below\n\n"
        "Task : {task}\n\n"
        "Source Summary  : {source_summary}\n\n"
        "Draft Post : {draft_post}"
    ),
    application="User-Validator-Agent",
    creator="Ujjwal-AI-Team",
    date_created=datetime(2026, 1, 23),
    required_vars=["task", "source_summary", "draft_post"],

)


RESEARCH_SYNTHESIS_AI_PROMPT = Prompt(
    prompt_text=(
        "System Prompt (Research Synthesis AI):\n"
        "You are an expert research synthesis AI. Synthesize the "
        "provided source texts into a concise, bullet-pointed summary relevant to the "
        "user's topic. Focus strictly on the facts provided in the sources. Do not add "
        "outside information."
    ),
    application="Research-Synthesis-AI",
    creator="Ujjwal-AI-Team",
    date_created=datetime(2026, 1, 23),
    required_vars=[]  # system prompt itself requires no variables
)


USER_RESEARCH_SYNTHESIS_PROMPT = Prompt(
    prompt_text=(
        "Topic: {topic}\n\nSources:\n{sources}"
    ),
    application="User-Research-Synthesis",
    creator="Ujjwal-AI-Team",
    date_created=datetime(2026, 1, 23),
    required_vars=["topic", "sources"]
)




# ---- Prompt Registry ----
PROMPTS: Dict[str, Prompt] = {
    "system:researcher_agent": RESEARCHER_AGENT_PROMPT,
    "system:writer_agent": WRITER_AGENT_PROMPT,
    "system:validator_agent": VALIDATOR_AGENT_PROMPT,
    "user:validator_agent": USER_VALIDATOR_AGENT_PROMPT,
    "system:research_synthesis_ai":RESEARCH_SYNTHESIS_AI_PROMPT,
    "user:research_synthesis_ai": USER_RESEARCH_SYNTHESIS_PROMPT

}

# ---- Helper Function ----
def get_prompt(role: str, agent_type: str, **kwargs) -> str:
    key = f"{role.lower()}:{agent_type.lower()}"
    prompt_obj = PROMPTS.get(key)
    if not prompt_obj:
        raise ValueError(f"No prompt found for role={role}, agent_type={agent_type}")
    
    return prompt_obj.prompt_text.format(**kwargs)


if __name__ == "__main__":
    prompt_text = get_prompt(
        role="user",
        agent_type="validator_agent",
        task="Check consistency of draft",
        source_summary="All facts match the source",
        draft_post="This is the draft content to be validated."
    )
    print(prompt_text)
