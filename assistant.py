from typing import Optional
import streamlit as st
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.embedder.together import TogetherEmbedder
from phi.embedder.fireworks import FireworksEmbedder

firewroks_api = "oV2Ggh7ihoe1uhfC8bCiHQSGgzbg0wGTyoAuVNqvxkFjREH7"

db_url = "postgresql://neondb_owner:65WYfupDerlJ@ep-aged-dust-a5d9he9l.us-east-2.aws.neon.tech/neondb?sslmode=require"
groq_api_key="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"


# dvs
def get_groq_assistant(
    llm_model: str = "llama3-70b-8192",
    embeddings_model: str = "nomic-embed-text",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
    user_grade: Optional[str] = None,
) -> Assistant:
    """Get a Groq RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    # Define the embeddings table based on the embeddings model
    embeddings_table = (
        "groq_rag_documents_ollama" if embeddings_model == "nomic-embed-text" else "groq_rag_documents_openai"
    )

    return Assistant(
        name="groq_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model="llama3-70b-8192", api_key=groq_api_key, max_tokens=6000),
        storage=PgAssistantStorage(table_name="groq_rag_assistant", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=FireworksEmbedder(api_key=firewroks_api),
            ),
            # 2 references are added to the prompt
            num_documents=2,
        ),
        description="You are an AI called 'GroqRAG' and your task is to answer questions using the provided information",
        instructions=[
            "When a user asks a question, you will be provided with information about the question.",
            "Carefully read this information and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
            "The user is a ${user_grade}, ensure your response tailors to the user's level of understanding",
            "Whenever a user says the word - generate; Do the following:",
            "Generate summaries based on your knowledge based",
            "Create flash cards based on your knowledge base",
            "Create questions with answers based on your knowledge base",
            "Do not say anything like - Let me know if you'd like me to generate more content!"
        ],
        # This setting adds references from the knowledge_base to the user prompt
        add_references_to_prompt=True,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # This setting adds chat history to the messages
        add_chat_history_to_messages=True,
        # This setting adds 4 previous messages from chat history to the messages
        num_history_messages=4,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )