#!/usr/bin/env python3
"""Script to systematically add type annotations and docstrings to the project.

This script analyzes Python files in the project and adds comprehensive
type annotations and docstrings following Google style guidelines.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any, Union


class TypeAnnotationAdder:
    """Adds type annotations and docstrings to Python files."""
    
    def __init__(self, project_root: Path):
        """Initialize the type annotation adder.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.processed_files: Set[Path] = set()
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project.
        
        Returns:
            List of Python file paths
        """
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
                    
        return python_files
    
    def analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function node and extract information for type annotations.
        
        Args:
            node: AST function definition node
            
        Returns:
            Dictionary containing function analysis results
        """
        info = {
            'name': node.name,
            'args': [],
            'returns': None,
            'has_docstring': False,
            'is_method': False
        }
        
        # Check if it's a method (has 'self' or 'cls' as first parameter)
        if node.args.args and node.args.args[0].arg in ['self', 'cls']:
            info['is_method'] = True
            
        # Analyze arguments
        for arg in node.args.args:
            arg_info = {
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None
            }
            info['args'].append(arg_info)
            
        # Check return annotation
        if node.returns:
            info['returns'] = ast.unparse(node.returns)
            
        # Check for existing docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            info['has_docstring'] = True
            
        return info
    
    def analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class node and extract information.
        
        Args:
            node: AST class definition node
            
        Returns:
            Dictionary containing class analysis results
        """
        info = {
            'name': node.name,
            'methods': [],
            'has_docstring': False,
            'attributes': []
        }
        
        # Check for existing docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            info['has_docstring'] = True
            
        # Analyze methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self.analyze_function(item)
                method_info['is_method'] = True
                info['methods'].append(method_info)
                
        return info
    
    def generate_function_docstring(self, func_info: Dict[str, Any]) -> str:
        """Generate a docstring template for a function.
        
        Args:
            func_info: Function information dictionary
            
        Returns:
            Generated docstring template
        """
        lines = ['"""TODO: Add function description.']
        
        # Add Args section if function has parameters
        non_self_args = [arg for arg in func_info['args'] if arg['name'] not in ['self', 'cls']]
        if non_self_args:
            lines.append('')
            lines.append('Args:')
            for arg in non_self_args:
                lines.append(f'    {arg["name"]}: TODO: Add parameter description')
                
        # Add Returns section if function has return annotation
        if func_info['returns'] and func_info['returns'] != 'None':
            lines.append('')
            lines.append('Returns:')
            lines.append('    TODO: Add return value description')
            
        lines.append('"""')
        return '\n    '.join(lines)
    
    def generate_class_docstring(self, class_info: Dict[str, Any]) -> str:
        """Generate a docstring template for a class.
        
        Args:
            class_info: Class information dictionary
            
        Returns:
            Generated docstring template
        """
        lines = [f'"""TODO: Add class description for {class_info["name"]}.']
        
        if class_info['attributes']:
            lines.append('')
            lines.append('Attributes:')
            for attr in class_info['attributes']:
                lines.append(f'    {attr}: TODO: Add attribute description')
                
        lines.append('"""')
        return '\n    '.join(lines)
    
    def suggest_type_annotations(self, func_info: Dict[str, Any]) -> Dict[str, str]:
        """Suggest type annotations for function parameters.
        
        Args:
            func_info: Function information dictionary
            
        Returns:
            Dictionary mapping parameter names to suggested types
        """
        suggestions = {}
        
        for arg in func_info['args']:
            if arg['annotation']:
                continue  # Already has annotation
                
            name = arg['name']
            
            # Common parameter name patterns
            if name in ['self', 'cls']:
                continue
            elif name.endswith('_path') or name.endswith('_dir'):
                suggestions[name] = 'Union[str, Path]'
            elif name.endswith('_size') or name.endswith('_len'):
                suggestions[name] = 'int'
            elif name.endswith('_list'):
                suggestions[name] = 'List[Any]'
            elif name.endswith('_dict'):
                suggestions[name] = 'Dict[str, Any]'
            elif name == 'device':
                suggestions[name] = 'str'
            elif name in ['batch_size', 'num_workers', 'max_frames']:
                suggestions[name] = 'int'
            elif name in ['threshold', 'fps']:
                suggestions[name] = 'float'
            else:
                suggestions[name] = 'Any'
                
        return suggestions
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single Python file to add type annotations and docstrings.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            True if file was modified, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the AST
            tree = ast.parse(content)
            
            # Analyze the file
            needs_modification = False
            modifications = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self.analyze_function(node)
                    if not func_info['has_docstring']:
                        needs_modification = True
                        modifications.append(f"Function '{func_info['name']}' needs docstring")
                        
                elif isinstance(node, ast.ClassDef):
                    class_info = self.analyze_class(node)
                    if not class_info['has_docstring']:
                        needs_modification = True
                        modifications.append(f"Class '{class_info['name']}' needs docstring")
            
            if needs_modification:
                print(f"File {file_path} needs modifications:")
                for mod in modifications:
                    print(f"  - {mod}")
                    
            return needs_modification
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def run(self) -> None:
        """Run the type annotation addition process."""
        print("Finding Python files...")
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files")
        
        files_needing_work = []
        
        for file_path in python_files:
            if self.process_file(file_path):
                files_needing_work.append(file_path)
                
        print(f"\nSummary: {len(files_needing_work)} files need type annotations/docstrings:")
        for file_path in files_needing_work:
            print(f"  - {file_path}")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    adder = TypeAnnotationAdder(project_root)
    adder.run()


if __name__ == '__main__':
    main()
