#!/usr/bin/env python3
"""
Script to execute all Jupyter notebooks in the project.
This converts notebooks to Python scripts and runs them.
"""

import os
import subprocess
import sys

def run_notebook(notebook_path):
    """
    Convert and run a single notebook.
    """
    try:
        # Convert notebook to script
        script_path = notebook_path.replace('.ipynb', '.py')
        subprocess.run([
            'jupyter', 'nbconvert', '--to', 'script', 
            '--output', script_path, notebook_path
        ], check=True)
        
        # Run the script
        print(f"Running {script_path}...")
        subprocess.run([sys.executable, script_path], check=True)
        print(f"Finished {script_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {notebook_path}: {e}")
        return False
    return True

def main():
    notebooks_dir = 'notebooks'
    if not os.path.exists(notebooks_dir):
        print(f"Directory {notebooks_dir} not found.")
        return
    
    notebooks = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]
    notebooks.sort()  # Run in order
    
    for nb in notebooks:
        nb_path = os.path.join(notebooks_dir, nb)
        if not run_notebook(nb_path):
            print(f"Failed to run {nb}")
            # Continue with others or stop?
            # break

if __name__ == '__main__':
    main()