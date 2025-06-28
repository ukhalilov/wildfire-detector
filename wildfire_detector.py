import os

# Fix OpenMP issue - Must be set before importing torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import logging
from dotenv import load_dotenv
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Dict, Any
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Set torch to use single thread to avoid OpenMP conflicts
torch.set_num_threads(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Load Environment Variables ---
load_dotenv()  # Load variables from .env file

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KNOWLEDGE_BASE_FOLDER = "./pdfs"
VECTOR_STORE_DIR = "./vector_store"
MODEL_PATH = "./wildfire_effnet.pt"  # Path to your model file
DOCS_IN_RETRIEVER = 15
RELEVANCE_THRESHOLD_DOCS = 0.7
RELEVANCE_THRESHOLD_PROMPT = 0.6

# --- RAG Imports ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_openai.chat_models import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage

    RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAG libraries not available: {e}")
    RAG_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="AI Analysis Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }

    .result-box {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Models and Stores ---

@st.cache_resource
def load_wildfire_model():
    """Load the wildfire detection model from file"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            logging.error(f"Model file not found at: {MODEL_PATH}")
            return None

        # Load the base model architecture
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 1),
            nn.Sigmoid()
        )

        # Load the trained weights with proper device mapping
        device = torch.device("cpu")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Ensure model is on CPU and in eval mode
        model = model.to(device)

        logging.info(f"Wildfire model loaded successfully from: {MODEL_PATH}")
        return model
    except Exception as e:
        error_msg = f"Failed to load wildfire model: {e}"
        st.error(error_msg)
        logging.error(error_msg)
        return None


@st.cache_resource
def get_llm_and_embeddings():
    """Initialize LLM and Embeddings for RAG"""
    if not RAG_AVAILABLE:
        return None, None

    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not found. RAG functionality will be disabled.")
        return None, None

    try:
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        logging.info("LLM and embeddings initialized successfully")
        return llm, embeddings
    except Exception as e:
        error_msg = f"Failed to initialize OpenAI models: {e}"
        st.warning(error_msg)
        logging.error(error_msg)
        return None, None


@st.cache_resource
def load_or_create_vector_store(_embeddings):
    """Loads existing FAISS store or creates a new one if needed."""
    if not _embeddings:
        return None

    index_file = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    pkl_file = os.path.join(VECTOR_STORE_DIR, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        try:
            vector_store = FAISS.load_local(
                VECTOR_STORE_DIR,
                _embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info(f"Vector store loaded from: {VECTOR_STORE_DIR}")
            return vector_store
        except Exception as e:
            logging.error(f"Error loading vector store from {VECTOR_STORE_DIR}: {e}")
            st.warning(f"Failed to load existing vector store: {e}")

    # Create new vector store
    if not os.path.isdir(KNOWLEDGE_BASE_FOLDER):
        st.warning(f"Knowledge base folder not found: {KNOWLEDGE_BASE_FOLDER}")
        return None

    documents = []
    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        st.warning(f"No PDF files found in {KNOWLEDGE_BASE_FOLDER}.")
        return None

    with st.spinner("Creating knowledge base..."):
        for filename in pdf_files:
            pdf_path = os.path.join(KNOWLEDGE_BASE_FOLDER, filename)
            try:
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
                logging.info(f"Loaded {len(pdf_docs)} pages from {filename}")
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
                continue

        if not documents:
            st.warning("No documents successfully loaded from PDF files.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        vector_store = FAISS.from_documents(split_docs, _embeddings)

        # Save the newly created store
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_DIR)
        logging.info(f"Vector store saved to: {VECTOR_STORE_DIR}")

        return vector_store


# --- Main App ---
st.markdown('<h1 class="main-header">🤖 AI Analysis Hub</h1>', unsafe_allow_html=True)


# Initialize components
@st.cache_data
def initialize_app():
    """Initialize all app components"""
    return {
        "wildfire_model": None,
        "llm": None,
        "embeddings": None,
        "vector_store": None,
        "initialization_complete": False
    }


# Initialize session state
if "app_state" not in st.session_state:
    st.session_state.app_state = initialize_app()

# Load models if not already loaded
if not st.session_state.app_state["initialization_complete"]:
    with st.spinner("Initializing AI models and knowledge base..."):
        # Load wildfire model
        st.session_state.app_state["wildfire_model"] = load_wildfire_model()

        # Load LLM and embeddings if RAG is available
        if RAG_AVAILABLE:
            llm, embeddings = get_llm_and_embeddings()
            st.session_state.app_state["llm"] = llm
            st.session_state.app_state["embeddings"] = embeddings

            # Load or create vector store
            if embeddings:
                st.session_state.app_state["vector_store"] = load_or_create_vector_store(embeddings)

        st.session_state.app_state["initialization_complete"] = True


# --- RAG Functions ---
def extract_document_info(doc):
    """Extract document name and page number from document metadata"""
    metadata = doc.metadata

    # Extract filename from source path
    source = metadata.get('source', 'Unknown Document')
    doc_name = os.path.basename(source).replace('.pdf', '') if source else 'Unknown Document'

    # Extract page number
    page_num = metadata.get('page', 0) + 1  # Convert 0-indexed to 1-indexed

    return doc_name, page_num


def query_rag_model(vector_store, user_prompt: str, chat_history_langchain: list, llm):
    """Enhanced RAG query function with document references"""
    if not vector_store or not llm:
        return "RAG system not available. Please check your configuration.", None, None

    try:
        # Document retrieval with scores
        docs_with_scores = vector_store.similarity_search_with_relevance_scores(user_prompt, k=5)
        relevant_docs = [doc for doc, score in docs_with_scores if score >= 0.7]

        if not relevant_docs:
            return "I couldn't find relevant information in the documents to answer your question.", None, None

        # Create context from documents with references
        context_parts = []
        doc_references = []

        for i, doc in enumerate(relevant_docs[:3]):
            doc_name, page_num = extract_document_info(doc)

            # Add to context with reference marker
            context_parts.append(f"[Source {i + 1}] {doc.page_content}")

            # Store reference info
            doc_references.append({
                'index': i + 1,
                'document': doc_name,
                'page': page_num
            })

        context_str = "\n\n".join(context_parts)

        # Enhanced prompt template
        system_prompt = f"""Based on the following context from multiple documents, answer the user's question. When referencing information, mention the source number in your response.

Context:
{context_str}

Question: {user_prompt}

Answer based only on the context provided. When mentioning specific information, reference the source (e.g., "According to Source 1..." or "As mentioned in Source 2..."):"""

        # Get response from LLM
        response = llm.invoke(system_prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return answer, doc_references, None

    except Exception as e:
        logging.error(f"Error in RAG query: {e}")
        return f"Error processing your question: {e}", None, None


# --- Image Analysis Functions ---
def analyze_wildfire_image(image: Image.Image) -> Dict[str, Any]:
    """Analyze image for wildfire detection"""
    model = st.session_state.app_state["wildfire_model"]
    if not model:
        return {"error": "Wildfire model not loaded"}

    try:
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Prepare input tensor
        input_tensor = transform(image).unsqueeze(0)

        # Model inference with no gradient computation
        with torch.no_grad():
            output = model(input_tensor)
            prob = float(output.item())

        prediction = "🔥 Fire Detected!" if prob < 0.5 else "✅ No Fire Detected."
        confidence = (1 - prob) if prob < 0.5 else prob

        return {
            "prediction": prediction,
            "confidence": f"{confidence:.2%}",
            "raw_score": prob
        }

    except Exception as e:
        logging.error(f"Error in wildfire analysis: {e}")
        return {"error": f"Error during analysis: {str(e)}"}


# --- Main Interface ---
col1, col2 = st.columns([1, 1])

# Image Analysis Section
with col1:
    st.markdown('<h2 class="section-header">🔥 Wildfire Detection</h2>', unsafe_allow_html=True)

    model_status = st.session_state.app_state["wildfire_model"] is not None
    if model_status:
        st.success("✅ Wildfire model loaded successfully!")
    else:
        st.error("❌ Wildfire model not available")

    # Image upload
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image for wildfire detection"
    )

    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Analysis button
        if st.button("🔍 Analyze for Wildfire", type="primary"):
            if model_status:
                with st.spinner("Analyzing image..."):
                    result = analyze_wildfire_image(image)

                    if "error" in result:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(result["error"])
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.success("Analysis completed successfully!")
                        st.write(f"**Prediction:** {result['prediction']}")
                        st.write(f"**Confidence:** {result['confidence']}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Wildfire model not available!")

# RAG Chatbot Section
with col2:
    st.markdown('<h2 class="section-header">💬 RAG Chatbot</h2>', unsafe_allow_html=True)

    rag_available = RAG_AVAILABLE and st.session_state.app_state["vector_store"] is not None

    if not RAG_AVAILABLE:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ RAG libraries not installed. Install langchain packages to enable this feature.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif rag_available:
        st.success("✅ Knowledge base loaded successfully!")
    else:
        st.warning("⚠️ Knowledge base not available. Check your PDF files and OpenAI API key.")

    if rag_available:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare chat history for LangChain
            chat_history_langchain = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history_langchain.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history_langchain.append(AIMessage(content=msg["content"]))

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    response, doc_references, references = query_rag_model(
                        st.session_state.app_state["vector_store"],
                        prompt,
                        chat_history_langchain,
                        st.session_state.app_state["llm"]
                    )

                    st.markdown(response)

                    # Display document references if available
                    if doc_references:
                        st.markdown("---")
                        st.markdown("**📚 Sources:**")
                        for ref in doc_references:
                            st.markdown(f"**Source {ref['index']}:** {ref['document']} (Page {ref['page']})")

                    st.session_state.messages.append({"role": "assistant", "content": response})

# Status Section
st.markdown("---")
st.markdown("### 📊 System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if st.session_state.app_state["wildfire_model"]:
        st.success("🔥 Wildfire Model: Ready")
    else:
        st.error("🔥 Wildfire Model: Not Available")

with status_col2:
    if st.session_state.app_state["vector_store"]:
        st.success("📚 Knowledge Base: Ready")
    else:
        st.error("📚 Knowledge Base: Not Available")

with status_col3:
    if st.session_state.app_state["llm"]:
        st.success("🤖 LLM: Ready")
    else:
        st.error("🤖 LLM: Not Available")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>🤖 AI Analysis Hub - Powered by Streamlit</p>
    <p>Ready for wildfire detection and document-based Q&A!</p>
</div>
""", unsafe_allow_html=True)