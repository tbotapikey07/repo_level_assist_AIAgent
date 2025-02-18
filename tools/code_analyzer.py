import os
import ast
from typing import List, Dict
import logging
import traceback
import datetime

# Supported programming file extensions
PROGRAMMING_EXTENSIONS = {
    '.py', '.java', '.cpp', '.c', '.h', '.hpp',
    '.js', '.ts', '.go', '.rs', '.rb', '.php',
    '.swift', '.kt', '.scala', '.hs', '.lua'
}


class CodeAnalyzer:
    """Analyzes code files to extract information about functions, classes, and dependencies."""

    def __init__(self):
        pass

    def get_code_structure(self, file_path: str) -> dict:
        """Extracts the code structure from a Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            Dictionary containing information about functions, classes, and imports.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read())

            functions = []
            classes = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    methods = []
                    for body_node in node.body:
                        if isinstance(body_node, ast.FunctionDef):
                            methods.append({
                                'name': body_node.name,
                                'args': [arg.arg for arg in body_node.args.args],
                                'docstring': ast.get_docstring(body_node)
                            })
                    classes.append({
                        'name': node.name,
                        'methods': methods,
                        'docstring': ast.get_docstring(node)
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'module': alias.name,
                            'alias': alias.asname
                        })
                elif isinstance(node, ast.ImportFrom):
                    imports.append({
                        'module': node.module,
                        'names': [alias.name for alias in node.names]
                    })

            return {
                'functions': functions,
                'classes': classes,
                'imports': imports
            }
        except Exception as e:
            return {'error': str(e)}

    def get_file_structure(self, directory: str) -> List[Dict]:
        """Get the structure of files in the directory with enhanced language support and validation"""
        import os
        import logging
        import traceback
        import datetime

        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('file_structure.log', mode='a')
            ]
        )
        logger = logging.getLogger('FileStructureAnalyzer')

        # Comprehensive list of programming file extensions
        programming_extensions = {
            # Java-specific extensions
            '.java', '.jsp', '.jspf', '.jspx', 
            # Spring Boot specific
            '.properties', '.yml', '.yaml', '.xml',
            
            # Python-specific
            '.py', '.pyi', '.pyc',
            
            # Other languages
            '.js', '.ts', '.tsx', '.jsx', 
            '.cpp', '.c', '.h', '.hpp', 
            '.cs', '.php', '.rb', '.go', '.rs', 
            '.vue', '.scala', '.kt', '.kts', '.swift',
            '.m', '.mm', '.r', '.pl', '.sh', '.bash', '.sql',
            '.html', '.css', '.scss', '.sass', '.less'
        }

        # Logging and tracking variables
        file_structures = []
        total_files_scanned = 0
        files_processed = 0
        files_skipped = 0
        files_with_errors = 0

        logger.info(f"Starting file structure analysis for directory: {directory}")

        # Walk through directory
        for root, _, files in os.walk(directory):
            for filename in files:
                total_files_scanned += 1
                
                try:
                    # Full path to the file
                    filepath = os.path.join(root, filename)
                    
                    # Get file extension
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    # Skip files with unsupported extensions
                    if file_ext not in programming_extensions:
                        logger.debug(f"Skipping unsupported file: {filepath}")
                        files_skipped += 1
                        continue
                    
                    # Get file stats
                    file_stats = os.stat(filepath)
                    
                    # Determine if file is text or binary
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            # Try to read a small portion of the file
                            f.read(1024)
                        is_text = True
                    except UnicodeDecodeError:
                        is_text = False
                    except Exception as text_check_error:
                        logger.warning(f"Error checking file type for {filepath}: {text_check_error}")
                        is_text = False
                    
                    # Prepare file structure dictionary
                    file_structure = {
                        'path': filepath,
                        'filename': filename,
                        'extension': file_ext,
                        'is_text': is_text,
                        'size': file_stats.st_size,
                        'modified': datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        'created': datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                    }
                    
                    file_structures.append(file_structure)
                    files_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {filepath}: {str(e)}")
                    logger.error(traceback.format_exc())
                    files_with_errors += 1

        # Comprehensive logging of results
        logger.info("File Structure Analysis Summary:")
        logger.info(f"Total files scanned: {total_files_scanned}")
        logger.info(f"Files processed successfully: {files_processed}")
        logger.info(f"Files skipped: {files_skipped}")
        logger.info(f"Files with errors: {files_with_errors}")

        return file_structures
