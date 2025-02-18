from typing import Dict, Optional, List
from .base_agent import BaseAgent
import json

class IntegrationTestAgent(BaseAgent):
    def __init__(self, vector_store=None):
        system_prompt = """You are an expert integration testing AI assistant.
        Your role is to design and implement integration tests.
        Consider:
        1. Component interactions
        2. Data flow testing
        3. Error scenarios
        4. Performance testing
        5. System boundaries
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)
    
    def process(self, input_data: Dict) -> str:
        prompt = input_data.get('prompt', '')
        search_results = input_data.get('search_results', [])
        requirements = input_data.get('requirements', {})
        context = input_data.get('context', {})
        
        # Analyze code context
        code_context = self._format_search_results(search_results)
        
        # Suggest test scenarios
        test_scenarios = self._suggest_test_scenarios(code_context)
        
        # Generate test cases
        test_cases = self._generate_test_cases(test_scenarios)
        
        # Format the response
        response = {
            "content": {
                "test_scenarios": test_scenarios,
                "test_cases": test_cases,
                "recommendations": [
                    "Ensure all endpoints have proper error handling tests",
                    "Include performance testing for critical paths",
                    "Test data consistency across services",
                    "Verify service dependencies and fallbacks"
                ]
            },
            "type": "json"
        }
        
        return json.dumps({
            "answer": response,
            "context": f"Code Context:\n{code_context}\n\nTest Scenarios:\n{json.dumps(test_scenarios, indent=2)}"
        })
    
    def _format_search_results(self, search_results: List[Dict]) -> str:
        if not search_results:
            return "No relevant code context found."
            
        formatted_results = []
        for idx, result in enumerate(search_results, 1):
            content = result.get('content', '').strip()
            metadata = result.get('metadata', {})
            file_path = metadata.get('path', 'Unknown file')
            
            formatted_results.append(f"Component {idx}: {file_path}\n```\n{content}\n```\n")
            
        return "\n".join(formatted_results)
    
    def _suggest_test_scenarios(self, code_context: str) -> List[Dict]:
        """Suggest test scenarios based on code context."""
        message = f"""
Based on the following code context, suggest integration test scenarios:

{code_context}

Format the response as a list of test scenarios, each containing:
1. Name of the scenario
2. Components involved
3. Test objective
4. Expected outcome
"""
        try:
            response = self.generate_response(message)
            scenarios_data = json.loads(response)
            return scenarios_data.get("answer", {}).get("content", [])
        except:
            # Fallback to basic scenarios if parsing fails
            return [{
                "name": "Basic Integration Test",
                "components": ["All Components"],
                "objective": "Verify basic integration",
                "expected_outcome": "All components work together"
            }]
    
    def _generate_test_cases(self, scenarios: List[Dict]) -> List[Dict]:
        """Generate detailed test cases from scenarios."""
        test_cases = []
        
        for scenario in scenarios:
            test_case = {
                "name": f"Test_{scenario['name'].replace(' ', '_')}",
                "description": scenario['objective'],
                "components": scenario['components'],
                "steps": [
                    "Setup test environment",
                    "Initialize components",
                    "Execute test actions",
                    "Verify results",
                    "Cleanup"
                ],
                "assertions": [
                    {
                        "check": scenario['expected_outcome'],
                        "type": "assertion"
                    }
                ]
            }
            test_cases.append(test_case)
        
        return test_cases
    
    def _analyze_components(self, components: List[Dict]) -> str:
        """Analyze components for integration points"""
        if not components:
            return "No components provided for analysis."
            
        analysis = []
        analysis.append("Component Integration Analysis:")
        
        for idx, component in enumerate(components, 1):
            name = component.get('name', f'Component {idx}')
            dependencies = component.get('dependencies', [])
            interfaces = component.get('interfaces', [])
            
            analysis.append(f"\n{name}:")
            if dependencies:
                analysis.append("Dependencies:")
                for dep in dependencies:
                    analysis.append(f"- {dep}")
            
            if interfaces:
                analysis.append("Interfaces:")
                for interface in interfaces:
                    analysis.append(f"- {interface}")
                    
        return "\n".join(analysis)
