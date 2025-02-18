from abc import ABC, abstractmethod
import google.generativeai as genai
from typing import Dict, List, Optional, Union
import os
import json
from groq import Groq
from embeddings.vector_store import VectorStore
import logging

class BaseAgent(ABC):
    """Base class for all AI agents."""
    
    def __init__(self, model_type: str = 'groq', model_name: Optional[str] = None, system_prompt: Optional[str] = None, vector_store: Optional[VectorStore] = None):
        """Initialize the base agent.
        
        Args:
            model_type: Type of model to use ('gemini' or 'groq')
            model_name: Name of the model to use
            system_prompt: Optional system prompt to initialize the agent
            vector_store: Optional vector store for context retrieval
        """
        self.model_type = model_type
        self.system_prompt = system_prompt
        self.vector_store = vector_store
        
        if model_type == 'gemini':
            #os.environ["GOOGLE_API_KEY"] = ''
            self.model = genai.GenerativeModel(model_name or 'models/gemini-pro')
            self.chat = self.model.start_chat(history=[])
        else:  # groq
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            self.model_name = model_name or "mixtral-8x7b-32768"
            self.messages = []  # Store chat history for Groq
            if system_prompt:
                self.messages.append({
                    "role": "system",
                    "content": system_prompt
                })
    
    def _get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from vector store"""
        if not self.vector_store:
            return ""
            
        results = self.vector_store.search(query, k=k)
        if not results:
            return ""
            
        context = []
        for result in results:
            content = result['content']
            metadata = result['metadata']
            file_path = metadata.get('path', 'Unknown file')
            context.append(f"From {file_path}:\n```\n{content}\n```\n")
            
        return "\n".join(context)
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        """Format search results for context injection.
        
        Args:
            search_results: List of search results from vector store
        
        Returns:
            Formatted string of search results
        """
        if not search_results:
            return "No relevant code context found."
        
        context = "Relevant Code Context:\n"
        for idx, result in enumerate(search_results, 1):
            # Truncate content to a reasonable length
            content = result.get('content', 'No content')[:500]
            
            context += f"\n{idx}. File: {result.get('path', 'Unknown')}\n"
            context += f"   Type: {result.get('type', 'Unknown')}\n"
            context += f"   Size: {result.get('size', 0)} bytes\n"
            context += f"   Snippet: {content}...\n"
            context += f"   Relevance Score: {result.get('distance', 'N/A')}\n"
        
        return context

    @abstractmethod
    def process(self, input_data: Dict) -> str:
        """Process input data and generate a response.
        
        Args:
            input_data: Dictionary containing:
                - prompt: User query or request
                - search_results: Optional relevant code context
                - requirements: Optional requirements context
                - context: Optional additional context
        
        Returns:
            JSON string with format:
            {
                "answer": "Generated response text",
                "context": "Relevant code context used"
            }
        """
        pass

    def generate_response(self, message: str) -> str:
        """Generate a response using the configured model.
        
        Args:
            message: Input message or prompt
            
        Returns:
            JSON string with format:
            {
                "answer": {
                    "content": "string",  # The actual response content
                    "type": "text|code|diagram|json",  # Type of content for proper display
                    "language": "string"  # Optional, programming language if type is code
                },
                "context": "string"  # Context used to generate the response
            }
        """
        try:
            if self.model_type == 'gemini':
                response = self.chat.send_message(message)
                
                # Detect if response is code
                content = response.text
                content_type = "text"
                language = None
                
                # Check if response appears to be code
                if "```" in content:
                    content_type = "code"
                    # Try to extract language and code
                    try:
                        parts = content.split("```")
                        if len(parts) >= 3:
                            language = parts[1].split('\n')[0].strip()
                            content = '\n'.join(parts[1].split('\n')[1:])
                    except:
                        # If extraction fails, treat whole content as code
                        content = content.replace("```", "").strip()
                        language = "text"
                elif content.startswith("{") and content.endswith("}"):
                    try:
                        json.loads(content)
                        content_type = "json"
                    except:
                        pass
                
                return json.dumps({
                    "answer": {
                        "content": content,
                        "type": content_type,
                        "language": language
                    },
                    "context": message
                })
            else:  # groq
                self.messages.append({
                    "role": "user",
                    "content": message
                })
                
                completion = self.client.chat.completions.create(
                    messages=self.messages,
                    model=self.model_name,
                    temperature=0.7,
                    max_tokens=1024
                )
                
                response = completion.choices[0].message.content
                self.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Use same content type detection as above
                content_type = "text"
                language = None
                
                if "```" in response:
                    content_type = "code"
                    try:
                        parts = response.split("```")
                        if len(parts) >= 3:
                            language = parts[1].split('\n')[0].strip()
                            response = '\n'.join(parts[1].split('\n')[1:])
                    except:
                        response = response.replace("```", "").strip()
                        language = "text"
                elif response.startswith("{") and response.endswith("}"):
                    try:
                        json.loads(response)
                        content_type = "json"
                    except:
                        pass
                
                return json.dumps({
                    "answer": {
                        "content": response,
                        "type": content_type,
                        "language": language
                    },
                    "context": message
                })
                
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")

    def _inject_context(self, prompt: str, search_results: List[Dict]) -> str:
        """Inject context from search results into the prompt.
        
        Args:
            prompt: Original user prompt
            search_results: List of search results from vector store
        
        Returns:
            Augmented prompt with context
        """
        context = self._format_search_results(search_results)
        
        augmented_prompt = f"""Context-Aware Prompt:
{context}

User Query:
{prompt}

Instructions:
1. Carefully review the provided code context
2. Use the context to inform and enhance your response
3. If the context is not directly relevant, still consider its implications
4. Provide a comprehensive and context-aware answer
"""
        return augmented_prompt
    
    def send_message(self, message: str, search_results: List[Dict] = None) -> str:
        """Send a message to the agent and get a response, including search results.
        
        Args:
            message: Input message string.
            search_results: List of search results from the vector database.
        
        Returns:
            JSON string with format:
            {
                "answer": "Generated response text",
                "context": "Relevant code context used"
            }
        """
        if search_results:
            return self.generate_response(self._inject_context(message, search_results))
        else:
            context = self._get_relevant_context(message)
            if context:
                message = f"""Context from codebase:
{context}

User Query: {message}

Please provide a response considering the above context."""
            return self.generate_response(message)

    def _get_tools(self) -> List[str]:
        """Get the list of tools available to this agent.
        
        Returns:
            List of tool names available to the agent
        """
        return []
    
    def _validate_input(self, input_data: Dict) -> bool:
        """Validate the input data.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            True if input is valid, False otherwise
        """
        return True
