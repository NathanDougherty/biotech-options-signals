#!/usr/bin/env python3
"""
Script to replace existing copyright headers with a simplified version.

Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied.
"""

import os
import glob
import re

NEW_COPYRIGHT = '''"""
Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied.
"""

'''

def replace_copyright_in_file(file_path):
    """Replace existing copyright header with simplified version."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First check if the file starts with a docstring
    docstring_pattern = re.compile(r'^""".*?"""', re.DOTALL)
    match = docstring_pattern.match(content)
    
    if match and "Copyright" in match.group(0):
        # Replace only the first docstring that contains "Copyright"
        new_content = NEW_COPYRIGHT + content[match.end():]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Updated copyright in {file_path}")
    else:
        print(f"No valid copyright header found in {file_path}")

def main():
    """Main function to process all Python files."""
    # Get all Python files
    python_files = glob.glob("*.py")
    
    # Skip this script itself
    python_files = [f for f in python_files if f != "replace_headers.py"]
    
    print(f"Found {len(python_files)} Python files")
    
    # Replace copyright in all files
    for py_file in python_files:
        replace_copyright_in_file(py_file)
    
    print("Copyright headers updated in all files.")

if __name__ == "__main__":
    main() 