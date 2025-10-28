import streamlit as st
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Sidebar for branding
with st.sidebar:
    st.markdown(
        """
        <style>
        .css-1d391kg {
            background-color: #1f2937 !important;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image("https://static.streamlit.io/examples/chatbot.png", width=150)
    st.markdown("### 🤖 Your Smart Assistant")
    st.markdown("Use this chatbot for question-answering tasks based on retrieved documents.")

# Main interface styling
st.markdown(
    """
    <style>
    .main {
        background-color: #111827;
        color: #ffffff;
    }
    .stChatMessage {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
        color: #ffffff !important;
    }
    .stChatMessage.user {
        border-left: 4px solid #ef4444;
    }
    .stChatMessage.assistant {
        border-left: 4px solid #10b981;
    }
    .stMarkdown p {
        color: #ffffff !important;
    }
    input[type="text"] {
        background-color: #1f2937 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.title("💬 AskMyDocs")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Embeddings + Vector Store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage("You are an assistant for question-answering tasks.")
    ]

# Display message history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat input
prompt = st.chat_input("Type your question here...")

if prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # LLM initialization
    llm = ChatOpenAI(model="gpt-4o", temperature=1)

    # Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5}
    )

    docs = retriever.invoke(prompt)
    docs_text = "\n\n".join(d.page_content for d in docs)

    # System prompt with context
    system_prompt = f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Keep answers concise (max 3 sentences).
Context: {docs_text}"""

    st.session_state.messages.append(SystemMessage(system_prompt))

    # LLM response
    result = llm.invoke(st.session_state.messages).content

    with st.chat_message("assistant"):
        st.markdown(result)
    st.session_state.messages.append(AIMessage(result))
