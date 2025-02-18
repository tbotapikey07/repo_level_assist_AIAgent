from typing import Dict, Optional, List
from .base_agent import BaseAgent

class CodeChangesAgent(BaseAgent):
    def __init__(self, vector_store=None):
        system_prompt = """You are an expert code changes AI assistant.
        Your role is to analyze code and suggest or implement changes based on requirements.
        Consider:
        1. Code quality and best practices
        2. Performance implications
        3. Maintainability
        4. Testing requirements
        5. Documentation needs
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)
    
    def process(self, input_data: Dict) -> str:
        prompt = input_data.get('prompt', '')
        search_results = input_data.get('search_results', [])
        code_changes = input_data.get('code_changes', '')
        
        # Format search results and context
        context = self._format_search_results(search_results)
        
        # Analyze code changes
        changes_analysis = self._analyze_changes(code_changes)
        
        # Construct review message
        message = f"""
Change Request: {prompt}

Code Changes:
{code_changes}

Analysis of Changes:
{changes_analysis}

Related Code Context:
{context}

Please provide a detailed code review including:
1. Quality assessment
2. Potential issues
3. Performance impact
4. Security considerations
5. Testing requirements
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
            
            formatted_results.append(f"Related File {idx}: {file_path}\n```\n{content}\n```\n")
            
        return "\n".join(formatted_results)
    
    def _analyze_changes(self, code_changes: str) -> str:
        """Analyze code changes for patterns and impact"""
        if not code_changes:
            return "No code changes provided."
            
        analysis = []
        
        # Count lines changed
        lines = code_changes.split('\n')
        added = sum(1 for line in lines if line.startswith('+'))
        removed = sum(1 for line in lines if line.startswith('-'))
        
        analysis.append(f"Changes Summary:")
        analysis.append(f"- Lines Added: {added}")
        analysis.append(f"- Lines Removed: {removed}")
        analysis.append(f"- Net Change: {added - removed} lines")
        
        # Analyze change patterns
        if '++' in code_changes:
            analysis.append("- Contains file additions")
        if '--' in code_changes:
            analysis.append("- Contains file deletions")
        
        return "\n".join(analysis)
