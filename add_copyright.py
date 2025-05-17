#!/usr/bin/env python3
"""
Script to add copyright header to all Python files in the repository
that don't already have a copyright notice.

Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied without explicit permission.
"""

import os
import glob
import re

COPYRIGHT_HEADER = '''"""
Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied.
"""

'''

def file_has_copyright(file_path):
    """Check if file already has a copyright notice."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(500)  # Read first 500 chars, copyright should be at the top
        return "Copyright" in content

def add_copyright_to_file(file_path):
    """Add copyright header to a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # If file starts with a docstring, insert copyright before it
    docstring_pattern = re.compile(r'^""".*?"""', re.DOTALL)
    if docstring_pattern.match(content):
        content = docstring_pattern.sub(COPYRIGHT_HEADER + docstring_pattern.match(content).group(0), content)
    else:
        # Otherwise, insert at the beginning
        content = COPYRIGHT_HEADER + content
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Added copyright to {file_path}")

def main():
    """Main function to process all Python files."""
    # Get all Python files
    python_files = glob.glob("*.py")
    
    # Skip this script itself
    python_files = [f for f in python_files if f != "add_copyright.py"]
    
    print(f"Found {len(python_files)} Python files")
    
    # Add copyright to files that don't have it
    for py_file in python_files:
        if not file_has_copyright(py_file):
            add_copyright_to_file(py_file)
        else:
            print(f"Skipping {py_file} - already has copyright")
    
    print("Copyright headers added to all files.")

if __name__ == "__main__":
    main() 