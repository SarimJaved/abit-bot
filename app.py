import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import streamlit as st
from streamlit.components.v1 import html

# Initialize Streamlit with enhanced UI
def init_streamlit():
    st.set_page_config(
        page_title="ABIT AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS with modern design
    st.markdown(
        """
        <style>
            :root {
                --primary: #4361ee;
                --secondary: #3f37c9;
                --accent: #4895ef;
                --dark: #1a1a2e;
                --light: #f8f9fa;
                --success: #4cc9f0;
                --warning: #f8961e;
                --danger: #f72585;
            }
            
            .main {
                background-color: #f5f7fa;
                padding: 2rem;
            }
            
            .sidebar .sidebar-content {
                background-color: #1a1a2e;
                color: white;
                padding: 1.5rem;
            }
            
            .header-container {
                background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
                padding: 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                color: white;
            }
            
            .header-title {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .header-subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
            }
            
            .chat-container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                padding: 1.5rem;
                height: 65vh;
                overflow-y: auto;
                margin-bottom: 1rem;
            }
            
            .user-message {
                background-color: #4361ee;
                color: white;
                padding: 1rem;
                border-radius: 12px 12px 0 12px;
                margin-bottom: 1rem;
                max-width: 80%;
                margin-left: auto;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .assistant-message {
                background-color: #f0f2f5;
                color: #333;
                padding: 1rem;
                border-radius: 12px 12px 12px 0;
                margin-bottom: 1rem;
                max-width: 80%;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .input-container {
                position: fixed;
                bottom: 2rem;
                left: 50%;
                transform: translateX(-50%);
                width: 80%;
                background: white;
                padding: 1rem;
                border-radius: 12px;
                box-shadow: 0 -4px 12px rgba(0,0,0,0.05);
                z-index: 100;
            }
            
            .stButton>button {
                background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.3s ease;
                width: 100%;
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(67, 97, 238, 0.3);
            }
            
            .spinner {
                color: #4361ee !important;
            }
            
            /* Animation for new messages */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .message-animation {
                animation: fadeIn 0.3s ease-out;
            }
            
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .header-title {
                    font-size: 1.8rem;
                }
                
                .input-container {
                    width: 90%;
                    bottom: 1rem;
                }
            }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Sidebar with app info
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #4cc9f0;">🤖 ABIT AI Assistant</h2>
            <p style="color: #a1a1a1;">Your intelligent assistant for ABIT AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### About ABIT AI:
        ABIT AI provides intelligent document processing and AI-powered solutions for businesses.
        
        ### Capabilities:
        - Answer questions about ABIT AI services
        - Provide information from our documentation
        - Assist with product inquiries
        - Offer technical support guidance
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center;">
            <small>Powered by LangChain, Groq, and Streamlit</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">ABIT AI Assistant</h1>
        <p class="header-subtitle">Ask me anything about ABIT AI services and documentation</p>
    </div>
    """, unsafe_allow_html=True)

# Load website content and documents
def load_knowledge_base():
    # Load website content
    loader = WebBaseLoader([
        "https://abit.ai/",
        "https://abit.ai/about",
        "https://abit.ai/services",
        "https://abit.ai/contact"
    ])
    website_data = loader.load()
    
    # Load PDF documents - make sure "abit.pdf" is in the same directory as your script
    pdf_loader = PyPDFLoader("abit.pdf")
    pdf_data = pdf_loader.load()
    
    # Combine all data (website + PDFs)
    all_documents = website_data + pdf_data
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    docs = text_splitter.split_documents(all_documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    
    return vector_store

# Initialize QA chain
def initialize_qa_chain(vector_store):
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY") or "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"
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

# Main application 
def main():
    init_streamlit()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm the ABIT AI Assistant. How can I help you today?"}
        ]
    
    # Load knowledge base and initialize QA chain
    if "qa_chain" not in st.session_state:
        with st.spinner("Loading ABIT AI knowledge base..."):
            try:
                vector_store = load_knowledge_base()
                st.session_state.qa_chain = initialize_qa_chain(vector_store)
            except Exception as e:
                st.error(f"Failed to initialize knowledge base: {str(e)}")
                return
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message message-animation">
                    <strong>You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message message-animation">
                    <strong>Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input container (fixed at bottom)
    input_container = st.container()
    with input_container:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        if prompt := st.chat_input("Ask me anything about ABIT AI..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get assistant response
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": prompt})
                    response = result["result"]
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Rerun to update the chat display
                    st.rerun()
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Scroll to bottom of chat
    html(
        f"""
        <script>
            var chatContainer = document.getElementById("chat-container");
            chatContainer.scrollTop = chatContainer.scrollHeight;
        </script>
        """
    )

if __name__ == "__main__":
    main()
