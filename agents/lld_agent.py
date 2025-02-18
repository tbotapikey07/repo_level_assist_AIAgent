from typing import Dict, List, Optional
from .base_agent import BaseAgent
import json

class LLDAgent(BaseAgent):
    def __init__(self, vector_store=None):
        system_prompt = """You are an expert low-level design AI assistant.
        Your role is to create and analyze low-level design specifications.
        Consider:
        1. Class diagrams
        2. Sequence diagrams
        3. Component interactions
        4. Data structures
        5. Design patterns
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)
    
    def process(self, input_data: Dict) -> str:
        prompt = input_data.get('prompt', '')
        search_results = input_data.get('search_results', [])
        requirements = input_data.get('requirements', {})
        context = input_data.get('context', {})
        
        # Analyze requirements and context
        design_analysis = self._analyze_requirements(requirements, context)
        code_context = self._format_search_results(search_results)
        
        message = f"""
Design Request: {prompt}

Requirements Analysis:
{design_analysis}

Code Context:
{code_context}

Please provide a detailed low-level design in JSON format including:
1. Component breakdown
2. Class/interface definitions
3. Data structures
4. Algorithms
5. Sequence diagrams
6. Error handling
7. Performance considerations

Format the response as a JSON object with the following structure:
{{
    "components": [{{
        "name": "string",
        "type": "class|interface|module",
        "description": "string",
        "attributes": ["string"],
        "methods": ["string"]
    }}],
    "relationships": [{{
        "source": "string",
        "target": "string",
        "type": "inheritance|composition|aggregation|dependency"
    }}],
    "sequence_flows": [{{
        "name": "string",
        "participants": ["string"],
        "steps": ["string"]
    }}],
    "data_structures": [{{
        "name": "string",
        "type": "string",
        "fields": ["string"]
    }}],
    "error_handling": [{{
        "component": "string",
        "error_type": "string",
        "handling_strategy": "string"
    }}],
    "performance_considerations": ["string"]
}}
"""
        # Get the response in JSON format
        #print("#################################")
        #print(message)
        response = self.generate_response(message)
        
        try:
            # Parse the response as JSON
            design_data = json.loads(response)
            
            # Generate diagrams from the design data
            class_diagram = self.generate_class_diagram(design_data)
            sequence_diagrams = []
            for flow in design_data.get('sequence_flows', []):
                sequence_diagrams.append(self.generate_sequence_diagram(flow))
            
            # Format the final response with diagrams
            formatted_response = {
                "design_data": design_data,
                "diagrams": {
                    "class_diagram": class_diagram,
                    "sequence_diagrams": sequence_diagrams
                }
            }
            
            return json.dumps(formatted_response, indent=2)
            
        except json.JSONDecodeError:
            # If response is not valid JSON, return it as is
            return response
    
    def _analyze_requirements(self, requirements: Dict, context: Dict) -> str:
        """Analyze requirements and context for design implications"""
        print("#################################")
        print("Requirements:", requirements)
        

        analysis = []
        analysis.append("Requirements Analysis:")
        
        # Functional requirements
        if 'functional' in requirements:
            analysis.append("\nFunctional Requirements:")
            for req in requirements['functional']:
                analysis.append(f"- {req}")
                
        # Non-functional requirements
        if 'non_functional' in requirements:
            analysis.append("\nNon-Functional Requirements:")
            for req in requirements['non_functional']:
                analysis.append(f"- {req}")
                
        # Technical constraints
        if 'constraints' in requirements:
            analysis.append("\nTechnical Constraints:")
            for constraint in requirements['constraints']:
                analysis.append(f"- {constraint}")
                
        # System context
        if 'system' in context:
            analysis.append("\nSystem Context:")
            for item in context['system']:
                analysis.append(f"- {item}")
                
        return "\n".join(analysis)
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        """Format code search results for context"""
        if not search_results:
            return "No relevant code context found."
            
        formatted_results = []
        for idx, result in enumerate(search_results, 1):
            content = result.get('content', '').strip()
            metadata = result.get('metadata', {})
            file_path = metadata.get('path', 'Unknown file')
            
            formatted_results.append(f"Component {idx}: {file_path}\n```python\n{content}\n```\n")
            
        return "\n".join(formatted_results)
    
    def generate_class_diagram(self, design_data: Dict) -> str:
        """Generate class diagram in Mermaid format.
        
        Args:
            design_data: Dictionary containing components and relationships
            
        Returns:
            Mermaid formatted class diagram
        """
        components = design_data.get('components', [])
        relationships = design_data.get('relationships', [])
        
        # Start Mermaid class diagram
        diagram = ["```mermaid", "classDiagram"]
        
        # Add classes and interfaces
        for component in components:
            name = component.get('name', '')
            comp_type = component.get('type', 'class')
            
            # Add class/interface definition
            if comp_type == 'interface':
                diagram.append(f"class {name} {{\n<<interface>>\n}}")
            else:
                diagram.append(f"class {name}")
            
            # Add attributes
            for attr in component.get('attributes', []):
                diagram.append(f"{name} : +{attr}")
            
            # Add methods
            for method in component.get('methods', []):
                diagram.append(f"{name} : +{method}()")
        
        # Add relationships
        for rel in relationships:
            source = rel.get('source', '')
            target = rel.get('target', '')
            rel_type = rel.get('type', '')
            
            # Map relationship types to Mermaid syntax
            mermaid_rel = {
                'inheritance': '-->', 
                'composition': '*--',
                'aggregation': 'o--',
                'dependency': '..>'
            }.get(rel_type, '-->')
            
            diagram.append(f"{source} {mermaid_rel} {target}")
        
        diagram.append("```")
        return "\n".join(diagram)
    
    def generate_sequence_diagram(self, flow: Dict) -> str:
        """Generate sequence diagram in Mermaid format.
        
        Args:
            flow: Dictionary containing sequence flow information
            
        Returns:
            Mermaid formatted sequence diagram
        """
        name = flow.get('name', 'Unnamed Flow')
        participants = flow.get('participants', [])
        steps = flow.get('steps', [])
        
        # Start Mermaid sequence diagram
        diagram = [
            "```mermaid",
            "sequenceDiagram",
            f"title {name}"
        ]
        
        # Add participants
        for participant in participants:
            diagram.append(f"participant {participant}")
        
        # Add sequence steps
        for step in steps:
            # Check if step is a dictionary with detailed info
            if isinstance(step, dict):
                source = step.get('from', '')
                target = step.get('to', '')
                message = step.get('message', '')
                msg_type = step.get('type', '->>') # Default to solid arrow with open head
                
                # Map message types to Mermaid syntax if needed
                mermaid_msg_type = {
                    'sync': '->>', 
                    'async': '-->>', 
                    'response': '-->', 
                    'note': 'Note'
                }.get(msg_type, msg_type)
                
                if msg_type == 'note':
                    diagram.append(f"Note over {source}: {message}")
                else:
                    diagram.append(f"{source}{mermaid_msg_type}{target}: {message}")
            else:
                # If step is a string, add it directly (assuming proper Mermaid syntax)
                diagram.append(step)
        
        diagram.append("```")
        return "\n".join(diagram)

    def suggest_data_structures(self, requirements: Dict) -> List[Dict]:
        """Suggest appropriate data structures based on requirements"""
        suggestions = []
        
        # Analyze requirements for data structure needs
        if 'data_requirements' in requirements:
            for req in requirements['data_requirements']:
                if 'search' in req.lower():
                    suggestions.append({
                        'type': 'Hash Table',
                        'purpose': 'Fast key-based lookups',
                        'complexity': 'O(1) average case',
                        'use_case': req
                    })
                elif 'order' in req.lower():
                    suggestions.append({
                        'type': 'Binary Search Tree',
                        'purpose': 'Ordered data storage and retrieval',
                        'complexity': 'O(log n)',
                        'use_case': req
                    })
                elif 'priority' in req.lower():
                    suggestions.append({
                        'type': 'Priority Queue',
                        'purpose': 'Priority-based processing',
                        'complexity': 'O(log n)',
                        'use_case': req
                    })
                    
        return suggestions
