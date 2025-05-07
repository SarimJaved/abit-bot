import os
from flask import Flask, render_template_string, request, jsonify
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize chat history
chat_history = [
    {"role": "assistant", "content": "Hello! I'm the ABIT AI Assistant. How can I help you today?"}
]

# Load knowledge base and initialize QA chain
def initialize_knowledge_base():
    try:
        # Load website content
        print("Loading website content...")
        loader = WebBaseLoader([
            "https://abit.ai/",
            "https://abit.ai/about",
            "https://abit.ai/services",
            "https://abit.ai/contact"
        ])
        website_data = loader.load()
        
        # Load PDF documents
        print("Loading PDF documents...")
        pdf_loader = PyPDFLoader("abit.pdf")
        pdf_data = pdf_loader.load()
        
        # Combine all data
        all_documents = website_data + pdf_data
        
        # Split documents into chunks
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        docs = text_splitter.split_documents(all_documents)
        
        # Create embeddings and vector store
        print("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            tokenizer_kwargs={'clean_up_tokenization_spaces': True}
        )
        
        print("Building vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Initialize QA chain
        print("Initializing LLM...")
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        print("Creating QA chain...")
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        
    except Exception as e:
        print(f"Error initializing knowledge base: {str(e)}")
        raise
# Initialize QA chain
try:
    qa_chain = initialize_knowledge_base()
except Exception as e:
    print(f"Failed to initialize QA chain: {str(e)}")
    # Create a dummy chain that returns error messages
    class DummyChain:
        def __call__(self, *args, **kwargs):
            return {"result": f"System not properly initialized: {str(e)}"}
    qa_chain = DummyChain()

# HTML template with all the original styling and enhancements
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ABIT AI Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
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
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            margin: 0;
            padding: 0;
            color: #333;
        }
        
        .container {
            display: flex;
            min-height: 100vh;
        }
        
        .sidebar {
            width: 300px;
            background-color: #1a1a2e;
            color: white;
            padding: 1.5rem;
            position: fixed;
            height: 100%;
            overflow-y: auto;
        }
        
        .main-content {
            flex: 1;
            padding: 2rem;
            margin-left: 300px;
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
            left: calc(300px + 10%);
            right: 10%;
            background: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 -4px 12px rgba(0,0,0,0.05);
            z-index: 100;
        }
        
        .input-form {
            display: flex;
            gap: 10px;
        }
        
        #user-input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            transition: border 0.3s ease;
        }
        
        #user-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(72, 149, 239, 0.2);
        }
        
        button {
            background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(67, 97, 238, 0.3);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        /* Animation for new messages */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-animation {
            animation: fadeIn 0.3s ease-out;
        }
        
        /* Markdown-like formatting in responses */
        .assistant-message strong {
            color: var(--dark);
        }
        
        .assistant-message em {
            font-style: italic;
        }
        
        .assistant-message ul, 
        .assistant-message ol {
            padding-left: 20px;
            margin: 8px 0;
        }
        
        .assistant-message code {
            background-color: #e9ecef;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
        }
        
        /* Loading spinner */
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Responsive adjustments */
        @media (max-width: 1024px) {
            .container {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
                position: relative;
                height: auto;
            }
            
            .main-content {
                margin-left: 0;
            }
            
            .input-container {
                left: 5%;
                right: 5%;
            }
        }
        
        @media (max-width: 768px) {
            .header-title {
                font-size: 1.8rem;
            }
            
            .input-container {
                width: 90%;
                bottom: 1rem;
            }
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            padding: 8px 12px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background-color: #666;
            border-radius: 50%;
            margin: 0 2px;
            animation: typingAnimation 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typingAnimation {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #4cc9f0; display: flex; align-items: center; justify-content: center; gap: 10px;">
                    <i class="fas fa-robot"></i> ABIT AI Assistant
                </h2>
                <p style="color: #a1a1a1;">Your intelligent assistant for ABIT AI</p>
            </div>
            
            <h3><i class="fas fa-info-circle"></i> About ABIT AI:</h3>
            <p>ABIT AI provides intelligent document processing and AI-powered solutions for businesses.</p>
            
            <h3><i class="fas fa-cogs"></i> Capabilities:</h3>
            <ul>
                <li>Answer questions about ABIT AI services</li>
                <li>Provide information from our documentation</li>
                <li>Assist with product inquiries</li>
                <li>Offer technical support guidance</li>
            </ul>
            
            <h3><i class="fas fa-lightbulb"></i> Example Questions:</h3>
            <ul>
                <li>What services does ABIT AI offer?</li>
                <li>How can I contact ABIT AI?</li>
                <li>Tell me about your document processing solutions</li>
            </ul>
            
            <hr>
            
            <div style="text-align: center; margin-top: 1rem;">
                <small>Powered by LangChain, Groq, and Flask</small>
                <div style="margin-top: 0.5rem;">
                    <button id="clear-chat" style="padding: 0.5rem 1rem; font-size: 0.9rem; background: var(--danger);">
                        <i class="fas fa-trash-alt"></i> Clear Chat
                    </button>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="header-container">
                <h1 class="header-title">ABIT AI Assistant</h1>
                <p class="header-subtitle">Ask me anything about ABIT AI services and documentation</p>
            </div>
            
            <div class="chat-container" id="chat-container">
                {% for message in chat_history %}
                    {% if message.role == "user" %}
                        <div class="user-message message-animation">
                            <strong>You:</strong><br>
                            {{ message.content }}
                        </div>
                    {% else %}
                        <div class="assistant-message message-animation">
                            <strong>Assistant:</strong><br>
                            {{ message.content | safe }}
                        </div>
                    {% endif %}
                {% endfor %}
            </div>
            
            <div class="input-container">
                <form class="input-form" id="chat-form">
                    <input type="text" id="user-input" placeholder="Ask me anything about ABIT AI..." autocomplete="off" required>
                    <button type="submit" id="submit-btn">
                        <i class="fas fa-paper-plane"></i> Send
                    </button>
                </form>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatForm = document.getElementById('chat-form');
            const userInput = document.getElementById('user-input');
            const chatContainer = document.getElementById('chat-container');
            const submitBtn = document.getElementById('submit-btn');
            const clearChatBtn = document.getElementById('clear-chat');
            
            // Scroll to bottom of chat on load
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Handle form submission
            chatForm.addEventListener('submit', async function(e) {
                e.preventDefault();
                const message = userInput.value.trim();
                
                if (message) {
                    // Disable input and button while processing
                    userInput.disabled = true;
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<div class="spinner"></div> Sending...';
                    
                    // Add user message to chat
                    addMessageToChat('user', message);
                    
                    // Clear input
                    userInput.value = '';
                    
                    // Show typing indicator
                    const typingIndicator = createTypingIndicator();
                    chatContainer.appendChild(typingIndicator);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                    try {
                        // Send to server
                        const response = await fetch('/ask', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ message: message })
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        
                        // Remove typing indicator
                        chatContainer.removeChild(typingIndicator);
                        
                        // Add assistant response
                        addMessageToChat('assistant', data.response);
                    } catch (error) {
                        // Remove typing indicator
                        chatContainer.removeChild(typingIndicator);
                        
                        // Show error message
                        addMessageToChat('assistant', `Sorry, I encountered an error: ${error.message}`);
                        console.error('Error:', error);
                    } finally {
                        // Re-enable input and button
                        userInput.disabled = false;
                        submitBtn.disabled = false;
                        submitBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send';
                        userInput.focus();
                    }
                }
            });
            
            // Clear chat history
            clearChatBtn.addEventListener('click', function() {
                if (confirm('Are you sure you want to clear the chat history?')) {
                    fetch('/clear', {
                        method: 'POST'
                    })
                    .then(response => {
                        if (response.ok) {
                            // Reload the page to get fresh chat
                            window.location.reload();
                        }
                    })
                    .catch(error => {
                        console.error('Error clearing chat:', error);
                    });
                }
            });
            
            // Auto-focus input on page load
            userInput.focus();
            
            // Helper function to add message to chat
            function addMessageToChat(role, content) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `${role}-message message-animation`;
                
                if (role === 'user') {
                    messageDiv.innerHTML = `<strong>You:</strong><br>${content}`;
                } else {
                    // Process markdown-like formatting
                    let processedContent = content
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // bold
                        .replace(/\*(.*?)\*/g, '<em>$1</em>')  // italic
                        .replace(/`(.*?)`/g, '<code>$1</code>');  // code
                    
                    // Handle lists if present
                    processedContent = processedContent.replace(/\n\s*-\s*(.*?)(?=\n|$)/g, '\n<li>$1</li>');
                    processedContent = processedContent.replace(/(<li>.*<\/li>)+/g, '<ul>$&</ul>');
                    
                    messageDiv.innerHTML = `<strong>Assistant:</strong><br>${processedContent}`;
                }
                
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            // Create typing indicator
            function createTypingIndicator() {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'assistant-message';
                
                const typingContent = document.createElement('div');
                typingContent.className = 'typing-indicator';
                typingContent.innerHTML = `
                    <strong>Assistant:</strong><br>
                    <div style="display: flex; align-items: center; gap: 4px;">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                `;
                
                typingDiv.appendChild(typingContent);
                return typingDiv;
            }
            
            // Allow Shift+Enter for new line, Enter to submit
            userInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    chatForm.dispatchEvent(new Event('submit'));
                }
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    message = data['message']
    
    # Add user message to chat history
    chat_history.append({"role": "user", "content": message})
    
    try:
        if not qa_chain:
            raise Exception("QA system not properly initialized")
            
        # Get assistant response
        result = qa_chain({"query": message})
        response = result["result"]
        
        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": response})
        
        return jsonify({"response": response})
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        chat_history.append({"role": "assistant", "content": error_msg})
        return jsonify({"response": error_msg}), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    global chat_history
    # Reset chat history but keep the initial greeting
    chat_history = [
        {"role": "assistant", "content": "Hello! I'm the ABIT AI Assistant. How can I help you today?"}
    ]
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)