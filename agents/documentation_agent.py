from typing import Dict, List
from .base_agent import BaseAgent
import json

class DocumentationAgent(BaseAgent):
    def __init__(self, vector_store=None):
        system_prompt = """You are an expert documentation AI assistant.
        Your role is to generate and improve code documentation.
        Consider:
        1. API documentation
        2. Code examples
        3. Architecture diagrams
        4. Implementation details
        5. Usage guidelines
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)
    
    def process(self, input_data: Dict) -> str:
        """
        Process input data and generate documentation.
        
        Args:
            input_data: Dictionary containing:
                - prompt: User documentation request
                - search_results: Optional search results
                - code_context: Optional code context for documentation
        
        Returns:
            JSON string with documentation response
        """
        # Extract input parameters
        prompt = input_data.get('prompt', '')
        search_results = input_data.get('search_results', [])
        code_context = input_data.get('code_context', {})
        
        # Format context from search results
        context = self._format_search_results(search_results)
        
        # Handle case with no code context
        if not code_context:
            # Generate a generic documentation request response
            response = {
                "content": {
                    "overview": "No specific code context provided. Please provide more details about the code or project you want to document.",
                    "api_documentation": "No API documentation available. Please provide code context.",
                    "usage_guide": "No usage guide available. Please provide code context.",
                    "architecture": "No architecture diagram available. Please provide code context.",
                    "dependencies": [],
                    "configuration": {}
                },
                "type": "json"
            }
            
            return json.dumps({
                "answer": response,
                "context": f"Search Results:\n{context}\n\nNote: No code context was provided for documentation."
            })
        
        # Analyze code context
        code_analysis = self._analyze_code_context(code_context)
        
        # Generate documentation components
        api_docs = self.generate_api_docs(code_context)
        usage_docs = self.generate_usage_docs(code_context)
        architecture_docs = self.generate_architecture_diagram(code_context)
        
        # Format the response
        response = {
            "content": {
                "overview": code_analysis,
                "api_documentation": api_docs,
                "usage_guide": usage_docs,
                "architecture": architecture_docs,
                "dependencies": code_context.get('dependencies', []),
                "configuration": code_context.get('configuration', {})
            },
            "type": "json"
        }
        
        return json.dumps({
            "answer": response,
            "context": f"Code Context:\n{context}\n\nCode Analysis:\n{code_analysis}"
        })
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        if not search_results:
            return "No relevant code context found."
            
        formatted_results = []
        for idx, result in enumerate(search_results, 1):
            content = result.get('content', '').strip()
            metadata = result.get('metadata', {})
            file_path = metadata.get('path', 'Unknown file')
            
            formatted_results.append(f"File {idx}: {file_path}\n```python\n{content}\n```\n")
            
        return "\n".join(formatted_results)
    
    def _analyze_code_context(self, code_context: Dict) -> str:
        """Analyze code context for documentation purposes"""
        if not code_context:
            return "No code context provided for analysis."
            
        analysis = []
        analysis.append("Code Analysis:")
        
        # Analyze classes
        if 'classes' in code_context:
            analysis.append("\nClasses:")
            for class_info in code_context['classes']:
                analysis.append(f"- {class_info['name']}")
                if 'methods' in class_info:
                    for method in class_info['methods']:
                        analysis.append(f"  - {method['name']}")
                        
        # Analyze functions
        if 'functions' in code_context:
            analysis.append("\nFunctions:")
            for func in code_context['functions']:
                analysis.append(f"- {func['name']}")
                
        # Analyze dependencies
        if 'dependencies' in code_context:
            analysis.append("\nDependencies:")
            for dep in code_context['dependencies']:
                analysis.append(f"- {dep}")
                
        return "\n".join(analysis)
    
    def generate_api_docs(self, code_context: Dict) -> str:
        """Generate API documentation from code context"""
        docs = []
        docs.append("# API Documentation\n")
        
        if 'classes' in code_context:
            for class_info in code_context['classes']:
                docs.append(f"## {class_info['name']}\n")
                docs.append(class_info.get('docstring', 'No description available.') + "\n")
                
                if 'methods' in class_info:
                    docs.append("### Methods\n")
                    for method in class_info['methods']:
                        docs.append(f"#### {method['name']}\n")
                        docs.append(method.get('docstring', 'No description available.') + "\n")
                        docs.append("```python\n" + method.get('signature', '') + "\n```\n")
                        
        return "\n".join(docs)
    
    def generate_usage_docs(self, code_context: Dict) -> str:
        """Generate usage documentation from code context"""
        docs = []
        docs.append("# Usage Guide\n")
        
        if 'classes' in code_context:
            for class_info in code_context['classes']:
                docs.append(f"## {class_info['name']}\n")
                docs.append(class_info.get('docstring', 'No description available.') + "\n")
                
                if 'methods' in class_info:
                    docs.append("### Methods\n")
                    for method in class_info['methods']:
                        docs.append(f"#### {method['name']}\n")
                        docs.append(method.get('docstring', 'No description available.') + "\n")
                        docs.append("```python\n" + method.get('signature', '') + "\n```\n")
                        
        return "\n".join(docs)
    
    def generate_architecture_diagram(self, code_context: Dict) -> str:
        """Generate architecture diagram in mermaid format"""
        diagram = []
        diagram.append("```mermaid")
        diagram.append("classDiagram")
        
        if 'classes' in code_context:
            # Add classes
            for class_info in code_context['classes']:
                diagram.append(f"class {class_info['name']}")
                
            # Add relationships
            for class_info in code_context['classes']:
                if 'relationships' in class_info:
                    for rel in class_info['relationships']:
                        diagram.append(f"{class_info['name']} {rel['type']} {rel['target']}")
                        
        diagram.append("```")
        return "\n".join(diagram)
