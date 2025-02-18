from typing import Dict, List, Optional
from .base_agent import BaseAgent
import json

class SecurityAgent(BaseAgent):
    def __init__(self, vector_store=None):
        system_prompt = """You are an expert security AI assistant.
        Your role is to identify and address security concerns.
        Consider:
        1. Vulnerability analysis
        2. Code injection risks
        3. Authentication/Authorization
        4. Data protection
        5. Security best practices
        """
        super().__init__(system_prompt=system_prompt, vector_store=vector_store)
        
        self.vulnerability_patterns = {
            'sql_injection': [
                r'execute\s*\(',
                r'executemany\s*\(',
                r'raw\s*\(',
                r'\.raw\s*\('
            ],
            'xss': [
                r'innerHTML\s*=',
                r'outerHTML\s*=',
                r'document\.write\(',
                r'eval\('
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'open\s*\(',
                r'file\s*\('
            ],
            'command_injection': [
                r'exec\s*\(',
                r'spawn\s*\(',
                r'system\s*\(',
                r'popen\s*\('
            ]
        }
    
    def process(self, input_data: Dict) -> str:
        prompt = input_data.get('prompt', '')
        search_results = input_data.get('search_results', [])
        code_context = input_data.get('code_context', {})
        
        # Analyze code for vulnerabilities
        vulnerabilities = self._scan_for_vulnerabilities(search_results)
        auth_issues = self._analyze_authentication(code_context)
        security_analysis = self._analyze_security(search_results, code_context)
        
        # Format the response
        response = {
            "content": {
                "vulnerabilities": vulnerabilities,
                "authentication_issues": auth_issues,
                "security_analysis": security_analysis,
                "recommendations": [
                    "Implement input validation",
                    "Use parameterized queries",
                    "Enable CSRF protection",
                    "Implement rate limiting",
                    "Use secure session management"
                ],
                "risk_levels": {
                    "sql_injection": "HIGH",
                    "xss": "HIGH",
                    "path_traversal": "MEDIUM",
                    "command_injection": "HIGH"
                }
            },
            "type": "json"
        }
        
        return json.dumps({
            "answer": response,
            "context": f"Security Analysis:\n{security_analysis}"
        })
    
    def _analyze_security(self, search_results: List[Dict], code_context: Dict) -> str:
        """Perform security analysis on code"""
        analysis = []
        analysis.append("Security Analysis Results:")
        
        # Analyze search results for vulnerability patterns
        vulnerabilities = self._scan_for_vulnerabilities(search_results)
        if vulnerabilities:
            analysis.append("\nPotential Vulnerabilities Found:")
            for vuln_type, instances in vulnerabilities.items():
                analysis.append(f"\n{vuln_type.upper()}:")
                for instance in instances:
                    analysis.append(f"- {instance}")
        
        # Analyze dependencies
        if 'dependencies' in code_context:
            analysis.append("\nDependency Analysis:")
            for dep in code_context['dependencies']:
                analysis.append(f"- Checking {dep} for known vulnerabilities")
        
        # Analyze authentication
        auth_issues = self._analyze_authentication(code_context)
        if auth_issues:
            analysis.append("\nAuthentication Issues:")
            for issue in auth_issues:
                analysis.append(f"- {issue}")
        
        return "\n".join(analysis)
    
    def _scan_for_vulnerabilities(self, search_results: List[Dict]) -> Dict[str, List[str]]:
        """Scan code for common vulnerability patterns"""
        import re
        vulnerabilities = {}
        
        for result in search_results:
            content = result.get('content', '')
            file_path = result.get('metadata', {}).get('path', 'Unknown file')
            
            for vuln_type, patterns in self.vulnerability_patterns.items():
                matches = []
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        matches.append(f"Found {vuln_type} risk in {file_path}")
                
                if matches:
                    if vuln_type not in vulnerabilities:
                        vulnerabilities[vuln_type] = []
                    vulnerabilities[vuln_type].extend(matches)
        
        return vulnerabilities
    
    def _analyze_authentication(self, code_context: Dict) -> List[str]:
        """Analyze authentication and authorization mechanisms"""
        issues = []
        
        # Check for basic auth issues
        auth_keywords = ['login', 'authenticate', 'authorize', 'permission']
        for keyword in auth_keywords:
            if keyword in str(code_context).lower():
                issues.append(f"Review {keyword} implementation for security")
        
        return issues
    
    def generate_security_report(self, vulnerabilities: Dict[str, List[str]]) -> str:
        """Generate a detailed security report"""
        report = []
        report.append("# Security Analysis Report\n")
        
        if not vulnerabilities:
            report.append("No immediate security concerns identified.\n")
            return "\n".join(report)
        
        report.append("## Identified Vulnerabilities\n")
        for vuln_type, instances in vulnerabilities.items():
            report.append(f"### {vuln_type.upper()}\n")
            report.append("#### Description")
            report.append(self._get_vulnerability_description(vuln_type))
            report.append("\n#### Instances")
            for instance in instances:
                report.append(f"- {instance}")
            report.append("\n#### Remediation")
            report.append(self._get_remediation_steps(vuln_type))
            report.append("\n")
        
        return "\n".join(report)
    
    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type"""
        descriptions = {
            'sql_injection': "SQL injection occurs when untrusted data is sent to an interpreter as part of a command or query.",
            'xss': "Cross-site scripting (XSS) attacks occur when untrusted data is sent to a web browser without proper validation or escaping.",
            'path_traversal': "Path traversal attacks attempt to access files and directories stored outside the intended directory.",
            'command_injection': "Command injection occurs when untrusted data is sent to a system shell."
        }
        return descriptions.get(vuln_type, "No description available.")
    
    def _get_remediation_steps(self, vuln_type: str) -> str:
        """Get remediation steps for vulnerability type"""
        remediation = {
            'sql_injection': "1. Use parameterized queries\n2. Implement input validation\n3. Use ORM frameworks\n4. Apply principle of least privilege",
            'xss': "1. Encode output\n2. Use Content Security Policy\n3. Validate input\n4. Use modern framework XSS protections",
            'path_traversal': "1. Validate file paths\n2. Use path canonicalization\n3. Implement proper access controls\n4. Use secure file APIs",
            'command_injection': "1. Avoid shell commands\n2. Use secure APIs\n3. Validate and sanitize input\n4. Implement proper access controls"
        }
        return remediation.get(vuln_type, "No remediation steps available.")
