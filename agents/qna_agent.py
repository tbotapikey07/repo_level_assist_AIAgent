from typing import Dict, Optional, List
from .base_agent import BaseAgent
import json
import logging

class QnA_Agent(BaseAgent):
    """Agent specialized in answering questions about the codebase."""
    
    def __init__(self, vector_store=None):
        system_prompt = """You are an expert Q&A AI assistant.
        Your role is to answer questions about the codebase.
        Consider:
        1. Code structure
        2. Implementation details
        3. Design decisions
        4. Dependencies
        5. Best practices
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)
        
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
    
    def process(self, input_data: Dict) -> str:
        """
        Process the input question and return a response.
        
        Args:
            input_data: Dictionary containing the question and context
        
        Returns:
            JSON string containing the answer in the format:
            {"answer":{"content":"","type":"","language":""},"context":""}
        """
        try:
            # Validate input
            if not self._validate_input(input_data):
                return json.dumps({
                    "answer": {
                        "content": "Invalid input. Please provide a question or prompt.",
                        "type": "text",
                        "language": ""
                    },
                    "context": ""
                })
            
            # Prepare context from search results
            search_results = input_data.get('search_results', [])
            context = self._format_search_results(search_results)
            
            # Prepare the message
            prompt = input_data.get('question', input_data.get('prompt', ''))
            message = f"""
Question: {prompt}

Relevant Code Context:
{context}

Please provide a detailed answer based on the code context above. Include:
1. Direct references to relevant code
2. Explanations of implementation details
3. Any potential improvements or best practices

If no context is available, provide a general response based on the question.
"""
            
            try:
                # Use generate_response instead of send_message
                response_json = self.generate_response(message)
                
                self.logger.debug(f"Response from generate_response: {response_json}")
            except Exception as e:
                self.logger.error(f"Error with generate_response: {e}")
                return json.dumps({
                    "answer": {
                        "content": f"Could not generate a response. Error: {str(e)}",
                        "type": "text",
                        "language": ""
                    },
                    "context": context
                })
            
            # Attempt to parse the response
            try:
                # First try parsing as a dictionary
                if isinstance(response_json, str):
                    response_data = json.loads(response_json)
                else:
                    response_data = response_json
                
                # Determine the content and type
                answer = response_data.get('answer', {})
                answer_content = answer.get('content', '')
                answer_type = answer.get('type', 'text')
                answer_language = answer.get('language', '')
                
                # Ensure we have a valid response
                if not answer_content:
                    answer_content = "Could not generate a meaningful response."
                
                # Return JSON-serialized response with our structure
                return json.dumps({
                    "answer": {
                        "content": answer_content,
                        "type": answer_type,
                        "language": answer_language
                    },
                    "context": context
                })
            
            except Exception as e:
                self.logger.error(f"Error parsing response: {e}")
                self.logger.error(f"Raw response: {response_json}")
                return json.dumps({
                    "answer": {
                        "content": f"Error parsing response: {str(e)}",
                        "type": "text",
                        "language": ""
                    },
                    "context": context
                })
        
        except Exception as e:
            self.logger.error(f"Unexpected error in process method: {e}")
            return json.dumps({
                "answer": {
                    "content": f"Unexpected error: {str(e)}",
                    "type": "text",
                    "language": ""
                },
                "context": ""
            })
    
    def _validate_input(self, input_data: Dict) -> bool:
        """Validate the input data.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            True if input contains required fields, False otherwise
        """
        return "prompt" in input_data or "question" in input_data

    def _format_search_results(self, search_results: List[Dict]) -> str:
        """Format search results into readable context"""
        # Log the raw search results for debugging
        self.logger.debug(f"Raw search results: {search_results}")
        
        if not search_results:
            # If no search results, try to provide a more informative message
            return """No relevant code context was found. 
This could be due to:
1. The search query was too broad or vague
2. No matching documents exist in the vector store
3. The vector store is not properly initialized
4. The search mechanism failed to retrieve results

Please provide more specific details or check the vector store configuration."""
        
        formatted_results = []
        for idx, result in enumerate(search_results, 1):
            try:
                # Safely extract content and metadata
                content = result.get('content', '').strip()
                metadata = result.get('metadata', {})
                file_path = metadata.get('path', 'Unknown file')
                file_type = metadata.get('type', 'Unknown type')
                
                # Log each result for debugging
                self.logger.debug(f"Result {idx}: File={file_path}, Type={file_type}")
                
                # Only add non-empty content
                if content:
                    formatted_results.append(f"[{idx}] File: {file_path} (Type: {file_type})\n```\n{content}\n```\n")
            except Exception as e:
                self.logger.error(f"Error formatting search result {idx}: {e}")
        
        # If no formatted results after processing, return a default message
        if not formatted_results:
            return """No processable code context was found. 
The search results exist but could not be formatted. 
Possible reasons:
1. Search results are in an unexpected format
2. Content extraction failed
3. All results have empty content

Please check the vector store and search mechanism."""
        
        return "\n".join(formatted_results)
