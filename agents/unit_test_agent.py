from typing import Dict, List, Optional
from .base_agent import BaseAgent

class UnitTestAgent(BaseAgent):
    def __init__(self, vector_store=None):
        system_prompt = """You are an expert unit testing AI assistant.
        Your role is to create and analyze unit tests.
        Consider:
        1. Test coverage
        2. Edge cases
        3. Mocking/Stubbing
        4. Test organization
        5. Testing best practices
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)
    
    def process(self, input_data: Dict) -> str:
        prompt = input_data.get('prompt', '')
        search_results = input_data.get('search_results', [])
        code_context = input_data.get('code_context', {})
        
        # Analyze code and generate tests
        test_analysis = self._analyze_code(code_context)
        code_content = self._format_search_results(search_results)
        
        message = f"""
Test Request: {prompt}

Code Analysis:
{test_analysis}

Code Context:
{code_content}

Please provide comprehensive unit tests including:
1. Test cases with clear descriptions
2. Test data and setup
3. Mock/stub requirements
4. Assertions and verifications
5. Edge cases
6. Error scenarios
7. Test organization
"""
        return self.generate_response(message)
    
    def _analyze_code(self, code_context: Dict) -> str:
        """Analyze code for test requirements"""
        analysis = []
        analysis.append("Test Requirements Analysis:")
        
        # Analyze functions/methods
        if 'functions' in code_context:
            analysis.append("\nFunctions to Test:")
            for func in code_context['functions']:
                analysis.append(f"- {func['name']}")
                if 'parameters' in func:
                    analysis.append("  Parameters:")
                    for param in func['parameters']:
                        analysis.append(f"    - {param['name']}: {param['type']}")
                if 'return_type' in func:
                    analysis.append(f"  Returns: {func['return_type']}")
                    
        # Analyze classes
        if 'classes' in code_context:
            analysis.append("\nClasses to Test:")
            for class_info in code_context['classes']:
                analysis.append(f"- {class_info['name']}")
                if 'methods' in class_info:
                    analysis.append("  Methods:")
                    for method in class_info['methods']:
                        analysis.append(f"    - {method['name']}")
                        
        # Analyze dependencies
        if 'dependencies' in code_context:
            analysis.append("\nDependencies to Mock:")
            for dep in code_context['dependencies']:
                analysis.append(f"- {dep}")
                
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
            
            formatted_results.append(f"File {idx}: {file_path}\n```python\n{content}\n```\n")
            
        return "\n".join(formatted_results)
    
    def generate_test_cases(self, function_info: Dict) -> List[Dict]:
        """Generate test cases for a function"""
        test_cases = []
        
        # Basic functionality test
        test_cases.append({
            'name': f"test_{function_info['name']}_basic_functionality",
            'description': "Test basic functionality with valid inputs",
            'setup': self._generate_test_setup(function_info),
            'input': self._generate_test_input(function_info, 'valid'),
            'expected': self._generate_expected_output(function_info, 'valid')
        })
        
        # Edge cases
        test_cases.append({
            'name': f"test_{function_info['name']}_edge_cases",
            'description': "Test edge cases and boundary conditions",
            'setup': self._generate_test_setup(function_info),
            'input': self._generate_test_input(function_info, 'edge'),
            'expected': self._generate_expected_output(function_info, 'edge')
        })
        
        # Error cases
        test_cases.append({
            'name': f"test_{function_info['name']}_error_cases",
            'description': "Test error handling and invalid inputs",
            'setup': self._generate_test_setup(function_info),
            'input': self._generate_test_input(function_info, 'invalid'),
            'expected': self._generate_expected_output(function_info, 'invalid')
        })
        
        return test_cases
    
    def _generate_test_setup(self, function_info: Dict) -> str:
        """Generate test setup code"""
        setup = []
        
        # Import statements
        setup.append("import pytest")
        setup.append(f"from {function_info.get('module', 'module')} import {function_info['name']}")
        
        # Mock setup if needed
        if 'dependencies' in function_info:
            setup.append("from unittest.mock import Mock, patch")
            for dep in function_info['dependencies']:
                setup.append(f"@patch('{dep}')")
                
        # Fixture setup
        setup.append("@pytest.fixture")
        setup.append("def setup():")
        setup.append("    # Setup test environment")
        setup.append("    yield")
        setup.append("    # Cleanup test environment")
        
        return "\n".join(setup)
    
    def _generate_test_input(self, function_info: Dict, input_type: str) -> Dict:
        """Generate test input data"""
        inputs = {}
        
        if 'parameters' in function_info:
            for param in function_info['parameters']:
                param_type = param.get('type', 'str')
                
                if input_type == 'valid':
                    inputs[param['name']] = self._generate_valid_input(param_type)
                elif input_type == 'edge':
                    inputs[param['name']] = self._generate_edge_input(param_type)
                else:  # invalid
                    inputs[param['name']] = self._generate_invalid_input(param_type)
                    
        return inputs
    
    def _generate_valid_input(self, param_type: str) -> any:
        """Generate valid test input based on parameter type"""
        type_map = {
            'str': 'test_string',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'dict': {'key': 'value'}
        }
        return type_map.get(param_type, None)
    
    def _generate_edge_input(self, param_type: str) -> any:
        """Generate edge case test input"""
        type_map = {
            'str': '',
            'int': 0,
            'float': 0.0,
            'bool': False,
            'list': [],
            'dict': {}
        }
        return type_map.get(param_type, None)
    
    def _generate_invalid_input(self, param_type: str) -> any:
        """Generate invalid test input"""
        type_map = {
            'str': None,
            'int': 'invalid',
            'float': 'invalid',
            'bool': None,
            'list': 'invalid',
            'dict': 'invalid'
        }
        return type_map.get(param_type, None)
    
    def _generate_expected_output(self, function_info: Dict, output_type: str) -> any:
        """Generate expected output based on function return type"""
        return_type = function_info.get('return_type', 'None')
        
        if output_type == 'valid':
            return self._generate_valid_input(return_type)
        elif output_type == 'edge':
            return self._generate_edge_input(return_type)
        else:  # invalid
            return self._generate_invalid_input(return_type)
