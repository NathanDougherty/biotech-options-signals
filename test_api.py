"""
Copyright (c) 2024 Nathan Dougherty
ALL RIGHTS RESERVED.
This code cannot be copied.
"""



"""
FILE: test_api.py
PURPOSE: Polygon.io REST API connectivity validation script.

This script tests the connection and functionality of the Polygon.io REST API to ensure
that the application can successfully retrieve financial data for options analysis.

Key functionality:
- Initializes a Polygon.io REST client using the API key
- Performs a basic stock quote request for Apple (AAPL) as a test case
- Fetches the last trade to confirm price data access
- Retrieves daily aggregate data (OHLCV) to verify historical data access
- Provides clear error feedback if any API requests fail

This script should be run before beginning analysis to verify:
1. API key validity and active subscription status
2. Network connectivity to Polygon.io services
3. Correct response formatting for downstream processing

Note: Contains a hardcoded API key which should be replaced with an environment variable
approach in production systems for security purposes.
"""

from polygon import RESTClient
import os

# Initialize the client
client = RESTClient("P5uGa_U5tUCKFCs0jhNm0sPXwzMojCnB")

# Try to get a simple stock quote
try:
    # Test with a major stock like AAPL
    ticker = "AAPL"
    print(f"Testing API with {ticker}...")
    
    # Get the last trade
    last_trade = client.get_last_trade(ticker)
    print(f"Last trade price: ${last_trade.price}")
    
    # Get the current day's aggregates
    aggs = client.get_aggs(ticker, 1, "day", "2024-03-22", "2024-03-22")
    if aggs.results:
        print(f"Today's data:")
        print(f"Open: ${aggs.results[0].o}")
        print(f"High: ${aggs.results[0].h}")
        print(f"Low: ${aggs.results[0].l}")
        print(f"Close: ${aggs.results[0].c}")
        print(f"Volume: {aggs.results[0].v}")
    
    print("\nAPI test successful!")
    
except Exception as e:
    print(f"Error testing API: {str(e)}") 