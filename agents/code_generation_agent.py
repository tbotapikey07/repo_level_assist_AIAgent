from typing import Dict, Optional, List
from .base_agent import BaseAgent

class CodeGenerationAgent(BaseAgent):
    """Agent specialized in generating code for new features, refactoring, and optimization."""

    def __init__(self, vector_store=None):
        system_prompt = """You are an expert code generation AI assistant.
        Your role is to generate high-quality code based on requirements.
        Consider:
        1. Code organization and structure
        2. Best practices and patterns
        3. Error handling
        4. Documentation
        5. Testing considerations
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)

    def process(self, input_data: Dict) -> str:
        """Process code generation request and provide code solutions.

        Args:
            input_data: Dictionary containing requirements and context

        Returns:
            Dictionary containing generated code and recommendations
        """
        if not self._validate_input(input_data):
            return {"error": "Invalid input data"}

        prompt = input_data.get('prompt', '')
        search_results = input_data.get('search_results', [])

        # Format search results for context
        context = self._format_search_results(search_results)

        # Construct message with context
        message = f"""
Requirements: {prompt}

Similar Code Examples:
{context}

Please generate code that:
1. Follows the project's coding style
2. Includes proper error handling
3. Is well-documented
4. Is optimized for performance
5. Includes usage examples
"""
        return self.generate_response(message)

    def _format_search_results(self, search_results: List[Dict]) -> str:
        """Format search results for context.

        Args:
            search_results: List of dictionaries containing search results

        Returns:
            Formatted string of search results
        """
        if not search_results:
            return "No similar code examples found."

        formatted_results = []
        for idx, result in enumerate(search_results, 1):
            content = result.get('content', '').strip()
            metadata = result.get('metadata', {})
            file_path = metadata.get('path', 'Unknown file')

            formatted_results.append(f"Example {idx} from {file_path}:\n```\n{content}\n```\n")

        return "\n".join(formatted_results)

    def _validate_input(self, input_data: Dict) -> bool:
        """Validate the input data.

        Args:
            input_data: Dictionary containing input parameters

        Returns:
            True if input contains required fields, False otherwise
        """
        return "prompt" in input_data
