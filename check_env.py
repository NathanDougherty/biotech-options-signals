"""
Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied.
"""



"""
FILE: check_env.py
PURPOSE: API key and environment variable validation utility.

This simple utility script verifies that the required environment variables 
(specifically the Polygon.io API key) are properly set and accessible to the application.
It loads variables from the .env file and prints the API key to confirm its presence.

This is primarily used during development and setup to ensure that:
1. The .env file exists and is properly formatted
2. The required API keys are set with valid values
3. The dotenv package is correctly configured

The script should be run before attempting to use any data fetching functionality
to prevent runtime errors due to missing API credentials.
"""

import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("POLYGON_API_KEY")) 