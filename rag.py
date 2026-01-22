"""Chatfolio AI Agent."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from settings import app_settings

# model provider
model = init_chat_model(
    "google_genai:gemini-2.5-flash-lite", api_key=app_settings.google_api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=app_settings.google_api_key,
)

# inmemory vector storage for testing
vector_store = InMemoryVectorStore(embeddings)

# assume the kb file is pdf
logger.info(
    f"rag: Loading knowledge base from the path: {app_settings.kb_file_path}."
)
pdf_loader = PyMuPDFLoader(file_path=app_settings.kb_file_path)
knowledge_base_pdfs = pdf_loader.load()
logger.info(
    f"rag: Loaded {len(knowledge_base_pdfs[0].page_content)} character document."
)


logger.info("rag: Splitting the loaded document into chunks")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(knowledge_base_pdfs)
logger.info(f"rag: Split the document into {len(all_splits)} sub documents.")

logger.info("Storing sub documents in the in-memory vector store.")
document_ids = vector_store.add_documents(documents=all_splits)


# rag tool for agent
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]

# prompt
prompt = """
Act as my professional surrogate. When asked about my background, use the CV
retrieval tool to provide specific facts regarding my experience, education,
and skills.

Constraints:

No Placeholders: Never use brackets or generic "mention [x]" text. If the tool
does not provide a specific detail, omit that point entirely or pivot to a
known strength.

Fact-Driven: Only state achievements, years of experience, and degrees that are
explicitly found in the retrieved data.

Persona: Speak in the first person ("I") with a confident, executive tone.

Authenticity: Focus on delivering impactful results and driving innovation
based on the specific projects and roles listed in my CV.
"""


agent = create_agent(model, tools, system_prompt=prompt)


query = "Can you introduce yourself ?"

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
