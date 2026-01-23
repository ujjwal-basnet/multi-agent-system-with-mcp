# -------------------------------------------------------------------------
# Helper Functions (Tokenization, Chunking, Embeddings, MCP, Pinecone, LLM)
# -------------------------------------------------------------------------

from typing import Any, Dict, List, Union
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pydantic import BaseModel, Field
import uuid
import textwrap

# Tokenization Helpers

def tokenize(text: str, model: str = "command-a-03-2025") -> List[int]:
    return cohere_chat_client.client.tokenize(text=text, model=model).tokens


def detokenize(tokens: List[int], model: str = "command-a-03-2025") -> str:
    return cohere_chat_client.client.detokenize(tokens=tokens, model=model).text


# Chunking Helper


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    tokens = tokenize(text)
    chunks: List[str] = []

    step = chunk_size - overlap
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk = detokenize(chunk_tokens).replace("\n", " ").strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# Embedding Helpers


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    texts = [t.replace("\n", " ") for t in texts]
    return cohere_embedding_client.embed_documents(texts)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text: str) -> List[float]:
    return get_embedding_batch([text])[0]




# MCP Models


class ValidatorContext(BaseModel):
    task: str
    source_summary: str
    draft_post: str


class MCPMessage(BaseModel):
    protocol_version: str = "1.0"
    sender: str = Field(min_length=3)
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


# MCP Helpers


def create_mcp_message(
    sender: str,
    content: Union[str, Dict[str, Any], ValidatorContext],
    metadata: Dict[str, Any] | None = None,
) -> MCPMessage:
    if not isinstance(content, (str, dict, ValidatorContext)):
        raise TypeError(
            f"content must be str, dict, or ValidatorContext; got {type(content).__name__}"
        )

    metadata = metadata or {}
    metadata.setdefault("task_id", str(uuid.uuid4()))
    metadata.setdefault("parents", [])

    return MCPMessage(
        sender=sender,
        content=content,
        metadata=metadata,
    )


def display_mcp(message: MCPMessage, title: str = "MCP Message") -> None:
    print(f"\n--- {title} (Sender: {message.sender}) ---")
    if isinstance(message.content, dict):
        print(f"Content Keys: {list(message.content.keys())}")
    else:
        print(f"Content: {textwrap.shorten(str(message.content), width=100)}")
    print(f"Metadata Keys: {list(message.metadata.keys())}")
    print("-" * (len(title) + 25))


# Pinecone Query Helper

def query_pinecone(
    query_text: str,
    namespace: str,
    top_k: int = 1,
):
    query_embedding = get_embedding(query_text)
    response = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=top_k,
        include_metadata=True,
    )
    return response["matches"]


def get_or_create_index(
    pc,
    index_name: str,
    embedding_dim: int= 384,
    namespaces_to_clear: List[str] | None = None,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    delete_timeout_seconds: int = 120,
    sleep_interval: int = 2,
):
    """
    - Creates serverless index if missing
    - Waits for readiness
    - Optionally clears specified namespaces (safe async delete)
    - Returns ready-to-use index
    """

    spec = ServerlessSpec(cloud=cloud, region=region)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric=metric,
            spec=spec,
        )

        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(sleep_interval)

    index = pc.Index(index_name)

    if namespaces_to_clear:
        stats = index.describe_index_stats()

        for namespace in namespaces_to_clear:
            ns_stats = stats.namespaces.get(namespace)

            if ns_stats and ns_stats.vector_count > 0:
                index.delete(delete_all=True, namespace=namespace)
                start_time = time.time()

                while True:
                    stats = index.describe_index_stats()
                    ns_stats = stats.namespaces.get(namespace)

                    if not ns_stats or ns_stats.vector_count == 0:
                        break

                    if time.time() - start_time > delete_timeout_seconds:
                        raise TimeoutError(
                            f"Timeout clearing namespace '{namespace}', "
                            f"remaining vectors: {ns_stats.vector_count}"
                        )

                    time.sleep(1)

    return index