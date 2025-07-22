#!/usr/bin/env python3
"""
Basic usage examples for the FinChat SEC QA SDK.

This script demonstrates the core functionality of the SDK including
querying SEC filings, risk analysis, and error handling.
"""

import os
from typing import List, Dict, Any

# Try importing the SDK - gracefully handle missing dependencies
try:
    from finchat_sec_qa.sdk import (
        FinChatClient,
        FinChatValidationError,
        FinChatNotFoundError,
        FinChatAPIError,
        FinChatConnectionError,
    )
    SDK_AVAILABLE = True
except ImportError as e:
    print(f"SDK not available: {e}")
    print("Install with: pip install finchat-sec-qa[sdk]")
    SDK_AVAILABLE = False


def basic_query_example():
    """Demonstrate basic query functionality."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping basic query example")
        return
    
    print("=== Basic Query Example ===")
    
    # Initialize client - can use environment variables
    base_url = os.getenv("FINCHAT_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("FINCHAT_API_KEY")  # Optional
    
    with FinChatClient(base_url=base_url, api_key=api_key) as client:
        try:
            # Query Apple's 10-K filing
            result = client.query(
                question="What was Apple's total net sales in fiscal 2022?",
                ticker="AAPL",
                form_type="10-K",
                limit=1
            )
            
            print(f"Question: What was Apple's total net sales in fiscal 2022?")
            print(f"Answer: {result.answer}")
            print(f"Citations found: {len(result.citations)}")
            
            for i, citation in enumerate(result.citations, 1):
                print(f"\nCitation {i}:")
                print(f"  Text: {citation.text[:100]}...")
                print(f"  Source: {citation.source}")
                print(f"  Page: {citation.page}")
                
        except FinChatValidationError as e:
            print(f"Validation error: {e.message}")
        except FinChatNotFoundError as e:
            print(f"Filing not found: {e.message}")
        except FinChatConnectionError as e:
            print(f"Connection error: {e.message}")
            print("Make sure the FinChat service is running")
        except FinChatAPIError as e:
            print(f"API error: {e.message}")


def risk_analysis_example():
    """Demonstrate risk analysis functionality."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping risk analysis example")
        return
        
    print("\n=== Risk Analysis Example ===")
    
    base_url = os.getenv("FINCHAT_BASE_URL", "http://localhost:8000")
    
    with FinChatClient(base_url=base_url) as client:
        try:
            # Analyze different types of financial text
            texts_to_analyze = [
                "The company reported strong quarterly earnings with record revenue growth.",
                "Supply chain disruptions and increased competition pose significant challenges.",
                "The company maintains a strong balance sheet with minimal debt.",
            ]
            
            for text in texts_to_analyze:
                result = client.analyze_risk(text)
                
                print(f"\nText: {text}")
                print(f"Sentiment: {result.sentiment}")
                print(f"Risk flags: {result.flags}")
                
        except FinChatAPIError as e:
            print(f"Risk analysis error: {e.message}")


def health_check_example():
    """Demonstrate health check functionality."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping health check example")
        return
        
    print("\n=== Health Check Example ===")
    
    base_url = os.getenv("FINCHAT_BASE_URL", "http://localhost:8000")
    
    with FinChatClient(base_url=base_url) as client:
        try:
            health = client.health_check()
            
            print(f"Service Status: {health.status}")
            print(f"Version: {health.version}")
            print(f"Timestamp: {health.timestamp}")
            
            if health.services:
                print("Individual Services:")
                for service, status in health.services.items():
                    status_emoji = "✅" if status == "ready" else "❌"
                    print(f"  {service}: {status} {status_emoji}")
            
            if health.is_healthy:
                print("✅ Service is healthy and ready for queries")
            else:
                print("⚠️  Service is not fully healthy")
                
        except FinChatConnectionError as e:
            print(f"❌ Cannot connect to FinChat service: {e.message}")
            print("Please check that the service is running and accessible")


def error_handling_examples():
    """Demonstrate comprehensive error handling."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping error handling examples")
        return
        
    print("\n=== Error Handling Examples ===")
    
    base_url = os.getenv("FINCHAT_BASE_URL", "http://localhost:8000")
    
    with FinChatClient(base_url=base_url, timeout=5) as client:
        # Example 1: Invalid ticker
        try:
            result = client.query("What is revenue?", "INVALID_TICKER", "10-K")
        except FinChatValidationError as e:
            print(f"✅ Caught validation error for invalid ticker: {e.message}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        # Example 2: Very long question (should trigger validation)
        try:
            very_long_question = "What is the revenue? " * 100  # Very long question
            result = client.query(very_long_question, "AAPL", "10-K")
        except FinChatValidationError as e:
            print(f"✅ Caught validation error for long question: {e.message}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        # Example 3: Non-existent ticker
        try:
            result = client.query("What is revenue?", "NONEXISTENT", "10-K")
        except FinChatNotFoundError as e:
            print(f"✅ Caught not found error: {e.message}")
        except Exception as e:
            print(f"Unexpected error: {e}")


def multi_company_comparison():
    """Demonstrate comparing multiple companies."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping multi-company comparison")
        return
        
    print("\n=== Multi-Company Comparison Example ===")
    
    base_url = os.getenv("FINCHAT_BASE_URL", "http://localhost:8000")
    companies = ["AAPL", "MSFT", "GOOGL"]
    question = "What is the company's primary source of revenue?"
    
    results = {}
    
    with FinChatClient(base_url=base_url) as client:
        for ticker in companies:
            try:
                print(f"Querying {ticker}...")
                result = client.query(question, ticker, "10-K")
                results[ticker] = result.answer[:200] + "..." if len(result.answer) > 200 else result.answer
            except Exception as e:
                results[ticker] = f"Error: {str(e)}"
    
    print(f"\nQuestion: {question}")
    print("=" * 60)
    
    for ticker, answer in results.items():
        print(f"\n{ticker}:")
        print(f"  {answer}")


def configuration_examples():
    """Demonstrate different client configuration options."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping configuration examples")
        return
        
    print("\n=== Configuration Examples ===")
    
    # Example 1: Basic configuration
    print("1. Basic configuration:")
    client1 = FinChatClient(
        base_url="http://localhost:8000",
        timeout=30
    )
    print(f"   Base URL: {client1.base_url}")
    print(f"   Timeout: {client1.timeout}")
    
    # Example 2: With API key
    print("\n2. With API key:")
    client2 = FinChatClient(
        base_url="https://api.finchat.example.com",
        api_key="your-api-key",
        timeout=60
    )
    print(f"   Base URL: {client2.base_url}")
    print(f"   Has API key: {client2.api_key is not None}")
    
    # Example 3: Custom retry configuration
    print("\n3. Custom retry configuration:")
    client3 = FinChatClient(
        base_url="http://localhost:8000",
        max_retries=5,
        retry_delay=2.0,
        user_agent="MyApp/1.0"
    )
    print(f"   Max retries: {client3.config.max_retries}")
    print(f"   Retry delay: {client3.config.retry_delay}")
    print(f"   User agent: {client3.config.user_agent}")


def main():
    """Run all examples."""
    print("FinChat SEC QA SDK Examples")
    print("=" * 50)
    
    if not SDK_AVAILABLE:
        print("❌ SDK is not available. Please install with:")
        print("   pip install finchat-sec-qa[sdk]")
        return
    
    # Run examples
    health_check_example()
    basic_query_example()
    risk_analysis_example()
    error_handling_examples()
    multi_company_comparison()
    configuration_examples()
    
    print("\n" + "=" * 50)
    print("Examples completed! 🎉")
    print("\nNext steps:")
    print("- Try modifying the examples with different companies")
    print("- Explore the async client for better performance")
    print("- Check out the SDK documentation for advanced features")


if __name__ == "__main__":
    main()