from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


print("Initializing Ollama embedding model...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")


problem_solution_text = """
Our database queries are experiencing heavy latency spikes during peak trading hours.
The high write volume causes row-level locks on the user balances table, leading to HTTP 504 timeouts at the API gateway.
Customers are currently reporting failed transactions when trying to process payments.

To resolve this bottleneck, we are introducing a Redis write-behind caching tier.
Incoming transaction requests will be written to an in-memory queue first and acknowledged instantly.
A background worker pool will then batch write these queued entries to the primary PostgreSQL database every 500 milliseconds.
"""

doc = Document(page_content=problem_solution_text, metadata={"source": "mixed_topics.txt"})

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90.0  
)

print("Splitting document using Ollama semantic chunker...")
chunks = semantic_splitter.split_documents([doc])


print(f"\n Total Chunks Generated: {len(chunks)}\n")


for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content.strip())
    print("=" * 50)