#!/usr/bin/env python3
"""
Startup script for the Pump Anomaly Detection Dashboard
"""
import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.absolute()
src_dir = project_root / 'src'

# Add both project root and src to Python path
for path in [str(project_root), str(src_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Change to project root directory
os.chdir(project_root)

# Load environment variables from .env file if it exists
env_file = project_root / '.env'
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

if __name__ == "__main__":
    # Import and run streamlit
    import streamlit.web.cli as stcli
    
    # Run the dashboard
    sys.argv = ["streamlit", "run", "src/dashboard.py"]
    sys.exit(stcli.main())
