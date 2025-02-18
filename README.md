# CodeCzar - AI-Powered Code Analysis Platform

CodeCzar is an advanced AI-powered code analysis platform that combines LLMs with sophisticated analysis techniques for efficient and accurate handling of complex code modifications.

## Features

- Deep Code Understanding through Knowledge Graph
- Pre-built & Custom Agents
- Seamless Integration with Development Workflows
- Support for Codebases of Any Size or Language

## Specialized Agents

1. Debugging Agent
2. Codebase Q&A Agent
3. Code Changes Agent
4. Integration Test Agent
5. Unit Test Agent
6. LLD Agent
7. Code Generation Agent

## Vector Database Features
- Upload multiple file types (Python, Java, C++, JavaScript, Go)
- Create new or load existing FAISS indices
- Automatic index persistence to disk
- File content analysis and metadata storage

## Knowledge Graph Integration
- Automatic node creation for uploaded files
- Relationship tracking between code components
- Visual exploration of code relationships

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a .env file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
codezczar/
├── agents/             # Agent implementations
├── database/          # Database models and operations
├── embeddings/        # Vector embeddings and FAISS operations
├── knowledge_graph/   # Knowledge graph implementation
├── tools/            # Agent tools implementation
├── utils/            # Utility functions
├── app.py            # Main Streamlit application
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
