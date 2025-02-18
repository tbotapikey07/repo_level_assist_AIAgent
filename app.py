import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from agents import (
    DebuggingAgent,
    QnA_Agent,
    CodeChangesAgent,
    IntegrationTestAgent,
    UnitTestAgent,
    LLDAgent,
    CodeGenerationAgent,
    SecurityAgent,
    DocumentationAgent
)
from typing import List, Dict, Any
from database.models import init_db
from embeddings.vector_store import VectorStore
from knowledge_graph.graph_builder import KnowledgeGraph
from tools.code_analyzer import CodeAnalyzer
import matplotlib.pyplot as plt
import networkx as nx
import logging
import traceback
import json

# Load environment variables
load_dotenv()



# Configure Gemini API
api_key_secrectpass = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key_secrectpass
genai.configure(api_key=api_key_secrectpass)

api_key_groq_secrectpass = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = api_key_groq_secrectpass



# Initialize database
db_session = init_db()

# Initialize vector store
vector_store = VectorStore(vector_db_path="vector_dbs")

analyzer = CodeAnalyzer()

# Initialize knowledge graph
knowledge_graph = KnowledgeGraph()

# Initialize code analyzer
#code_analyzer = CodeAnalyzer()

def process_documents(file_structures: List[dict]) -> List[dict]:
    """Process files with comprehensive validation and error handling for vector store"""
    import logging
    import os
    import traceback

    # Configure logging with more detailed output
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('document_processing.log', mode='a')
        ]
    )
    logger = logging.getLogger('DocumentProcessor')

    documents = []
    
    # Detailed logging of input
    logger.info(f"Starting document processing")
    logger.info(f"Total file structures received: {len(file_structures)}")
    
    # Comprehensive logging of file structures
    for idx, file_info in enumerate(file_structures):
        logger.debug(f"File Structure {idx}: {file_info}")

    # Supported programming file extensions
    supported_extensions = {
        # Java-specific extensions
        '.java', '.jsp', '.jspf', '.jspx', 
        '.properties', '.yml', '.yaml', '.xml',
        
        # Python-specific
        '.py', '.pyi', '.pyc',
        
        # Other languages
        '.js', '.ts', '.tsx', '.jsx', 
        '.cpp', '.c', '.h', '.hpp', 
        '.cs', '.php', '.rb', '.go', '.rs', 
        '.vue', '.scala', '.kt', '.kts', '.swift',
        '.m', '.mm', '.r', '.pl', '.sh', '.bash', '.sql',
        '.html', '.css', '.scss', '.sass', '.less'
    }

    # Tracking processing details
    total_files = len(file_structures)
    processed_files = 0
    skipped_files = 0
    error_files = 0

    for idx, file_info in enumerate(file_structures):
        try:
            # Detailed validation logging
            logger.debug(f"Processing file {idx}: {file_info}")

            # Validate file path
            if not file_info.get('path'):
                logger.warning(f"Skipping file at index {idx}: No path provided")
                skipped_files += 1
                continue

            # Check file extension
            file_ext = os.path.splitext(file_info['path'])[1].lower()
            if file_ext not in supported_extensions:
                logger.info(f"Skipping unsupported file type: {file_info['path']} (extension: {file_ext})")
                skipped_files += 1
                continue

            # Validate required keys with detailed logging
            required_keys = {'path', 'is_text', 'modified', 'size'}
            missing_keys = required_keys - set(file_info.keys())
            if missing_keys:
                logger.warning(f"Skipping file at index {idx}: Missing keys {missing_keys}")
                skipped_files += 1
                continue

            # Read file content for text files
            if file_info['is_text']:
                try:
                    with open(file_info['path'], 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as read_error:
                    logger.error(f"Error reading file {file_info['path']}: {read_error}")
                    logger.error(traceback.format_exc())
                    error_files += 1
                    continue

                # Skip empty files
                if not content or len(content.strip()) < 10:
                    logger.info(f"Skipping empty file: {file_info['path']}")
                    skipped_files += 1
                    continue

                # Create document entry
                doc = {
                    'path': file_info['path'],
                    'content': content,
                    'type': file_ext,
                    'size': file_info['size']
                }
                documents.append(doc)
                processed_files += 1
            else:
                # Handle binary files with minimal information
                doc = {
                    'path': file_info['path'],
                    'content': f"Binary file: {file_info['path']}",
                    'type': 'binary',
                    'size': file_info['size']
                }
                documents.append(doc)
                processed_files += 1

        except Exception as e:
            logger.error(f"Unexpected error processing file {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            error_files += 1
            continue

    # Comprehensive processing summary
    logger.info("Document Processing Summary:")
    logger.info(f"Total files scanned: {total_files}")
    logger.info(f"Files processed successfully: {processed_files}")
    logger.info(f"Files skipped: {skipped_files}")
    logger.info(f"Files with errors: {error_files}")
    logger.info(f"Final documents ready for vector store: {len(documents)}")

    return documents

def initialize_agents(vector_store=None):
    """Initialize all available agents.
    
    Args:
        vector_store: Optional vector store for context retrieval
    """
    return {
        'Debugging Agent': DebuggingAgent(vector_store=vector_store),
        'Codebase Q&A Agent': QnA_Agent(vector_store=vector_store),
        'Code Changes Agent': CodeChangesAgent(vector_store=vector_store),
        'Integration Test Agent': IntegrationTestAgent(vector_store=vector_store),
        'Unit Test Agent': UnitTestAgent(vector_store=vector_store),
        'LLD Agent': LLDAgent(vector_store=vector_store),
        'Code Generation Agent': CodeGenerationAgent(vector_store=vector_store),
        'Security Agent': SecurityAgent(vector_store=vector_store),
        'Documentation Agent': DocumentationAgent(vector_store=vector_store)
    }

def delete_vector_store_files(vector_store: VectorStore):
    """
    Provide a Streamlit UI for deleting vector store files with logging and error handling.
    
    Args:
        vector_store (VectorStore): The vector store instance to delete files from
    """
    st.sidebar.subheader("Delete Vector Store Files")
    
    # Use consistent vector database directory
    vector_db_dir = "vector_dbs"
    
    # List all .faiss files in the vector database directory
    try:
        db_files = [f for f in os.listdir(vector_db_dir) if f.endswith('.faiss')]
        
        if not db_files:
            st.sidebar.warning("No vector database files found.")
            return
        
        # Multiselect for file deletion
        files_to_delete = st.sidebar.multiselect(
            "Select knowledge  database files to delete", 
            options=db_files,
            format_func=lambda x: f"{x} ({os.path.getsize(os.path.join(vector_db_dir, x))//1024}KB)"
        )
        
        if st.sidebar.button("Confirm Deletion"):
            if not files_to_delete:
                st.sidebar.warning("No files selected for deletion.")
                return
            
            # Delete selected files
            for file in files_to_delete:
                faiss_path = os.path.join(vector_db_dir, file)
                docs_path = faiss_path + '.docs'
                
                try:
                    # Remove FAISS index file
                    os.remove(faiss_path)
                    
                    # Remove corresponding documents file if it exists
                    if os.path.exists(docs_path):
                        os.remove(docs_path)
                    
                    st.success(f"Deleted {file}")
                except Exception as e:
                    st.error(f"Error deleting {file}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error listing vector database files: {str(e)}")

def format_json_for_users(json_data: Dict) -> str:
    """
    Convert JSON response to a user-friendly format.
    
    Handles two specific JSON response formats:
    1. {"answer":"", "context":""}
    2. {"answer":{"content":"", "type":"", "language":""}, "context":""}
    
    Returns a formatted, human-readable string representation.
    """
    import json
    
    # Ensure we're working with a dictionary
    if not isinstance(json_data, dict):
        return str(json_data)
    
    formatted_text = []
    
    # Extract answer and context
    answer = json_data.get('answer', '')
    context = json_data.get('context', '')
    
    # Handle first format: simple string answer
    if isinstance(answer, str):
        if answer:
            formatted_text.append("📝 Response:")
            formatted_text.append(answer)
    
    # Handle second format: nested answer dictionary
    elif isinstance(answer, dict):
        content = answer.get('content', '')
        content_type = answer.get('type', 'text')
        language = answer.get('language', '')
        
        # Code snippet
        if content_type == 'code':
            formatted_text.append("📝 Code Snippet:")
            formatted_text.append(f"Language: {language or 'Unknown'}")
            formatted_text.append(f"```{language or ''}\n{content}\n```")
        
        # JSON content
        elif content_type == 'json':
            formatted_text.append("📊 JSON Response:")
            if isinstance(content, (dict, list)):
                formatted_text.append(json.dumps(content, indent=2))
            else:
                formatted_text.append(str(content))
        
        # Text or other types
        else:
            formatted_text.append(f"📝 {content_type.capitalize()} Response:")
            formatted_text.append(str(content))
    
    # Handle context if present
    if context:
        formatted_text.append("\n🔍 Context:")
        formatted_text.append(str(context))
    
    # If no meaningful content was found
    if not formatted_text:
        formatted_text.append("📝 No response content available.")
    
    return "\n".join(formatted_text)

# Agent descriptions dictionary
AGENT_DESCRIPTIONS = {
    'Debugging Agent': "Automatically analyzes stacktraces and provides debugging steps specific to your codebase.",
    'Codebase Q&A Agent': "Answers questions about your codebase and explains functions, features, and architecture.",
    'Code Changes Agent': "Analyzes code changes, identifies affected APIs, and suggests improvements before merging.",
    'Integration Test Agent': "Generates integration test plans and code for flows to ensure components work together properly.",
    'Unit Test Agent': "Automatically creates unit test plan and code for individual functions to enhance test coverage.",
    'LLD Agent': "Creates a low level design for implementing a new feature by providing functional requirements.",
    'Code Generation Agent': "Generates code for new features, refactors existing code, and suggests optimizations.",
    'Documentation Agent': "Generates comprehensive documentation, including API docs, usage guides, and architecture diagrams.",
    'Security Agent': "Performs security analysis, identifies vulnerabilities, and suggests security best practices.",
}

def initialize_agents(vector_store=None):
    """
    Initialize and configure agents with vector store context.
    
    Args:
        vector_store: Optional vector store for context retrieval
    
    Returns:
        Dict[str, BaseAgent]: Configured agents
    """
    agents = {
        'Debugging Agent': DebuggingAgent(vector_store=vector_store),
        'Codebase Q&A Agent': QnA_Agent(vector_store=vector_store),
        'Code Changes Agent': CodeChangesAgent(vector_store=vector_store),
        'Integration Test Agent': IntegrationTestAgent(vector_store=vector_store),
        'Unit Test Agent': UnitTestAgent(vector_store=vector_store),
        'LLD Agent': LLDAgent(vector_store=vector_store),
        'Code Generation Agent': CodeGenerationAgent(vector_store=vector_store),
        'Security Agent': SecurityAgent(vector_store=vector_store),
        'Documentation Agent': DocumentationAgent(vector_store=vector_store)
    }
    
    return agents

def main():
    st.title('CodeCzar - AI-Powered Code Analysis Platform')
    
    # Initialize components
    vector_db_dir = "vector_dbs"
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Initialize vector store
    vector_store = VectorStore(dimension=768, vector_db_path=vector_db_dir)
    
    # Initialize agents
    agents = initialize_agents(vector_store)
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Project directory
        project_dir = st.text_input("Project Directory", 
            value="Project Folder Path",
            help="Enter the path to your project directory")
        
        # Vector Database Management
        st.subheader("knowledge  Database Management")
        
        # Create new vector database
        with st.expander("Create knowledge  Database"):
            new_db_name = st.text_input("knowledge  Database Name", 
                help="Enter a name for the new knowledge  database")
            
            # Process files button
            if st.button("Process and Save Files"):
                if os.path.exists(project_dir) and new_db_name:
                    with st.spinner("Processing files..."):
                        try:
                            # Get file structure
                            analyzer = CodeAnalyzer()
                            file_structures = analyzer.get_file_structure(project_dir)
                            
                            st.write(f"Total files found: {len(file_structures)}")
                            
                            # Process documents
                            documents = process_documents(file_structures)
                            
                            # Add to vector store
                            vector_store.add_documents(documents)
                            
                            # Save the database
                            db_path = os.path.join(vector_db_dir, f"{new_db_name}.faiss")
                            vector_store.save_index(db_path)
                            
                            st.success(f"Processed {len(documents)} files into {new_db_name}")
                            
                            # Debug information
                            st.write("Debug Information:")
                            st.write(f"Total documents in vector store: {len(vector_store.documents)}")
                            st.write(f"Vector store index size: {vector_store.index.ntotal}")
                            
                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                else:
                    st.error("Please provide a valid project directory and database name")
        
        # Vector database selection and management
        st.subheader("knowledge  Databases")
        db_files = [f for f in os.listdir(vector_db_dir) if f.endswith('.faiss')]
        
        # Select database
        if db_files:
            selected_db_path = st.selectbox(
                "Select knowledge Database",
                options=db_files,
                format_func=lambda x: f"{x} ({os.path.getsize(os.path.join(vector_db_dir, x))//1024}KB)"
            )
            
            # Ensure vector database directory exists
            vector_db_dir = "vector_dbs"
            os.makedirs(vector_db_dir, exist_ok=True)
            
            # Validate selected database path
            if not selected_db_path:
                st.warning("Please select a knowledge  database")
                return
            
            # Full path to the FAISS index file
            faiss_index_path = os.path.join(vector_db_dir, selected_db_path)
            
            # Verify the full path exists
            if not os.path.exists(faiss_index_path):
                st.error(f"Vector database file not found: {faiss_index_path}")
                return
            
            # Load selected database
            try:
                vector_store.load_index(faiss_index_path)
                st.success(f"Loaded {vector_store.index.ntotal} documents")
                
                # Debug database details
                st.write("Database Details:")
                st.write(f"Total documents: {len(vector_store.documents)}")
                st.write(f"Index size: {vector_store.index.ntotal}")
                
                # Test search functionality
                test_query = st.text_input("Test Vector Search", 
                    help="Enter a query to test vector search functionality")
                
                if st.button("Perform Search"):
                    if test_query:
                        search_results = vector_store.search(test_query, k=5)
                        
                        st.write("Search Results:")
                        for idx, result in enumerate(search_results, 1):
                            st.write(f"Result {idx}:")
                            st.write(f"  Path: {result['path']}")
                            st.write(f"  Type: {result['type']}")
                            st.write(f"  Size: {result['size']} bytes")
                            st.write(f"  Distance: {result['distance']}")
                            st.write(f"  Content Preview: {result['content'][:200]}...")
                    else:
                        st.warning("Please enter a search query")
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
        else:
            st.warning("No vector databases found. Create a new one!")
        
        # Model selection
        st.subheader("Model Configuration")
        model_type = st.selectbox(
            "Select Model",
            options=['gemini', 'groq'],
            help="Choose the AI model to use"
        )
        
        if model_type == 'groq':
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not found in environment variables")
                model_type = 'gemini'  # Fallback to gemini
    
    # Main content area
    st.header("AI Agents")
    
    # Agent selection with descriptions
    selected_agent = st.selectbox(
        "Select an AI Agent", 
        list(AGENT_DESCRIPTIONS.keys()),
        #format_func=lambda agent: f"{agent} - {AGENT_DESCRIPTIONS.get(agent, 'No description available')}"
    )
    
    # Display selected agent description
    st.info(f"**{selected_agent}**: {AGENT_DESCRIPTIONS.get(selected_agent, 'No description available')}")
    
    # User input
    user_query = st.text_area("Enter your query or request", height=100)
    
    if st.button("Process Request"):
        if user_query:
            try:
                with st.spinner(f"Processing with {selected_agent}..."):
                    # Prepare input data
                    input_data = {
                        'prompt': user_query,
                        'search_results': vector_store.search(user_query, k=5) if vector_store else [],
                        'requirements': {},  # Can be populated based on context
                        'context': {}  # Can be populated based on context
                    }
                    
                    # Process with selected agent
                    agent = agents[selected_agent]
                    response = agent.process(input_data)
                    
                    # Display response
                    st.markdown("### Response")
                    
                    try:
                        # Parse response as JSON
                        response_data = json.loads(response)
                        
                        # Create tabs for answer, context and user-friendly view
                        tabs = st.tabs(["Answer", "User-Friendly View", "Context"])
                        
                        with tabs[0]:
                            answer = response_data.get("answer", {})
                            content = answer.get("content", "")
                            content_type = answer.get("type", "text")
                            language = answer.get("language")
                            
                            if content_type == "code":
                                st.code(content, language=language or "text")
                            elif content_type == "json":
                                try:
                                    # If it's valid JSON, display it nicely formatted
                                    json_content = json.loads(content)
                                    st.json(json_content)
                                except:
                                    # If JSON parsing fails, display as code
                                    st.code(content, language="json")
                            elif content_type == "diagram":
                                # Diagrams are typically in mermaid or plantuml format
                                st.markdown(content)
                            else:
                                # Plain text
                                st.markdown(content)
                        
                        with tabs[1]:
                            # Display user-friendly version
                            st.json(response_data)
                            try:
                                answer = response_data.get("answer", {})
                                if isinstance(answer, dict):
                                    user_friendly = format_json_for_users(answer)
                                    st.markdown(user_friendly)
                                else:
                                    st.markdown(str(answer))
                            except Exception as e:
                                st.error(f"Error formatting user-friendly view: {str(e)}")
                                st.text(str(answer))
                        
                        with tabs[2]:
                            # Display the context used
                            st.subheader("Context Used")
                           # st.markdown(response_data.get("context", "No context available"))
                            
                            # Display search results if available
                            if input_data.get("search_results"):
                                st.subheader("Search Results")
                                for idx, result in enumerate(input_data["search_results"], 1):
                                    with st.expander(f"Result {idx}: {result.get('path', 'Unknown')}"):
                                        st.write(f"**Type:** {result.get('type', 'Unknown')}")
                                        st.write(f"**Size:** {result.get('size', 0)} bytes")
                                        st.write(f"**Relevance Score:** {result.get('distance', 'N/A')}")
                                        st.write("**Content Preview:**")
                                        st.code(result.get('content', 'No content available')[:500], 
                                               language=result.get('type', '').lstrip('.') or 'text')
                    
                    except json.JSONDecodeError:
                        # If not JSON, display as markdown
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"Error displaying response: {str(e)}")
                        st.json(response)  # Display raw response as fallback
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                logging.error(f"Agent processing error: {traceback.format_exc()}")
        else:
            st.warning("Please enter a query or request")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # # Chat input
    # if prompt := st.chat_input("How can I help you?"):
    #     # Add user message to chat history
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
            
    #     # Get agent response
    #     with st.chat_message("assistant"):
    #         with st.spinner("Thinking..."):
    #             agent = agents[selected_agent]
                
    #             # Prepare context from selected vector database
    #             search_results = vector_store.search(prompt, k=5) if vector_store else []
                
    #             response = agent.process({
    #                 'prompt': prompt,
    #                 'model_type': model_type,
    #                 'search_results': search_results
    #             })
    #             st.markdown(response)
    #             st.session_state.messages.append({"role": "assistant", "content": response})

    # Delete vector store files
    delete_vector_store_files(vector_store)

if __name__ == '__main__':
    main()
