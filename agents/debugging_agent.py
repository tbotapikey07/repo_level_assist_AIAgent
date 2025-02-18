from typing import Dict, Optional, List
from .base_agent import BaseAgent

class DebuggingAgent(BaseAgent):
    def __init__(self, vector_store=None):
        system_prompt = """You are an expert debugging AI assistant.
        Your role is to analyze code issues and provide detailed debugging insights.
        For each issue, consider:
        1. Root cause analysis
        2. Error patterns and stack traces
        3. Potential fixes and solutions
        4. Prevention strategies
        5. Testing recommendations
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)
    
    def process(self, input_data: Dict) -> str:
        prompt = input_data.get('prompt', '')
        search_results = input_data.get('search_results', [])
        error_message = input_data.get('error_message', '')
        stack_trace = input_data.get('stack_trace', '')
        
        # Format search results and context
        context = self._format_search_results(search_results)
        
        # Construct debugging message
        message = f"""
Error Description: {prompt}

Error Message: {error_message}
Stack Trace: {stack_trace}

Related Code Context:
{context}

Please provide a detailed debugging analysis including:
1. Root cause identification
2. Step-by-step debugging process
3. Recommended fixes
4. Prevention measures
5. Testing suggestions
"""
        return self.generate_response(message)
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        if not search_results:
            return "No relevant code context found."
            
        formatted_results = []
        for idx, result in enumerate(search_results, 1):
            content = result.get('content', '').strip()
            metadata = result.get('metadata', {})
            file_path = metadata.get('path', 'Unknown file')
            
            formatted_results.append(f"File {idx}: {file_path}\n```\n{content}\n```\n")
            
        return "\n".join(formatted_results)
    
    def _analyze_error(self, error_message: str, stack_trace: str) -> str:
        """Analyze error message and stack trace for patterns"""
        if not error_message and not stack_trace:
            return "No error information provided."
            
        analysis = []
        if error_message:
            analysis.append(f"Error Type: {error_message.split(':')[0]}")
        if stack_trace:
            # Extract relevant frames from stack trace
            frames = stack_trace.split('\n')
            analysis.extend([frame.strip() for frame in frames if frame.strip()])
            
        return "\n".join(analysis)
