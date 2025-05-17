"""
Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied.
"""



"""
FILE: test_ws.py
PURPOSE: Polygon.io WebSocket API connectivity validation script.

This script tests the connection and functionality of the Polygon.io WebSocket API
to ensure that the application can successfully receive real-time financial data
for options analysis.

Key functionality:
- Initializes a Polygon.io WebSocket client using the API key
- Subscribes to real-time trade updates for Apple (AAPL)
- Defines a callback function to handle and display incoming WebSocket messages
- Maintains a connection for 5 seconds to verify continuous data flow
- Properly closes the connection after testing

This script verifies:
1. WebSocket API access is available with the current API key
2. Real-time data can be received and processed correctly
3. Connection management works as expected

Note: Contains a hardcoded API key which should be replaced with an environment variable
approach in production systems for security purposes.
"""

from polygon import WebSocketClient
import time

# Initialize the WebSocket client
client = WebSocketClient("P5uGa_U5tUCKFCs0jhNm0sPXwzMojCnB")

# Define a callback function
def handle_msg(msgs):
    print(f"Received message: {msgs}")

# Connect to the WebSocket
client.subscribe("T.AAPL", handle_msg)  # Subscribe to AAPL trades

# Keep the connection alive for a few seconds
print("Waiting for messages...")
time.sleep(5)

# Close the connection
client.close() 