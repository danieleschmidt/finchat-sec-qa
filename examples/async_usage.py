#!/usr/bin/env python3
"""
Asynchronous usage examples for the FinChat SEC QA SDK.

This script demonstrates how to use the async client for better performance
when making multiple concurrent requests.
"""

import asyncio
import time
from typing import List, Dict, Tuple, Any

# Try importing the SDK - gracefully handle missing dependencies
try:
    from finchat_sec_qa.sdk import (
        AsyncFinChatClient,
        FinChatValidationError,
        FinChatNotFoundError,
        FinChatAPIError,
        FinChatConnectionError,
        QueryResponse,
    )
    SDK_AVAILABLE = True
except ImportError as e:
    print(f"SDK not available: {e}")
    print("Install with: pip install finchat-sec-qa[sdk]")
    SDK_AVAILABLE = False


async def basic_async_example():
    """Demonstrate basic async functionality."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping async example")
        return
    
    print("=== Basic Async Example ===")
    
    async with AsyncFinChatClient(base_url="http://localhost:8000") as client:
        try:
            # Single async query
            result = await client.query(
                question="What is Microsoft's primary business segment?",
                ticker="MSFT",
                form_type="10-K"
            )
            
            print(f"Question: What is Microsoft's primary business segment?")
            print(f"Answer: {result.answer}")
            print(f"Citations: {len(result.citations)}")
            
        except FinChatConnectionError:
            print("Could not connect to FinChat service - make sure it's running")
        except FinChatAPIError as e:
            print(f"API error: {e.message}")


async def concurrent_queries_example():
    """Demonstrate concurrent queries for better performance."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping concurrent queries example")
        return
    
    print("\n=== Concurrent Queries Example ===")
    
    companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    question = "What was the total revenue in the most recent fiscal year?"
    
    async with AsyncFinChatClient(base_url="http://localhost:8000") as client:
        # Start timing
        start_time = time.time()
        
        # Create tasks for all queries
        tasks = []
        for ticker in companies:
            task = client.query(question, ticker, "10-K")
            tasks.append((ticker, task))
        
        # Execute all queries concurrently
        results = {}
        print("Executing concurrent queries...")
        
        for ticker, task in tasks:
            try:
                result = await task
                results[ticker] = {
                    "answer": result.answer[:150] + "..." if len(result.answer) > 150 else result.answer,
                    "citations": len(result.citations),
                    "success": True
                }
            except Exception as e:
                results[ticker] = {
                    "answer": f"Error: {str(e)}",
                    "citations": 0,
                    "success": False
                }
        
        end_time = time.time()
        
        # Display results
        print(f"\nCompleted {len(companies)} queries in {end_time - start_time:.2f} seconds")
        print(f"Question: {question}")
        print("=" * 70)
        
        for ticker, result in results.items():
            status = "✅" if result["success"] else "❌"
            print(f"\n{ticker} {status}:")
            print(f"  {result['answer']}")
            if result["success"]:
                print(f"  Citations: {result['citations']}")


async def batch_risk_analysis():
    """Demonstrate batch risk analysis."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping batch risk analysis")
        return
    
    print("\n=== Batch Risk Analysis Example ===")
    
    # Sample financial texts to analyze
    texts = [
        "The company reported record quarterly earnings with strong growth across all segments.",
        "Supply chain disruptions and inflationary pressures continue to impact operations.",
        "New product launches are expected to drive significant revenue growth next quarter.",
        "Regulatory changes may require substantial compliance investments.",
        "The company maintains a strong cash position and low debt levels.",
    ]
    
    async with AsyncFinChatClient(base_url="http://localhost:8000") as client:
        print("Analyzing sentiment for multiple texts concurrently...")
        
        # Create tasks for all risk analyses
        tasks = []
        for i, text in enumerate(texts):
            task = client.analyze_risk(text)
            tasks.append((i, text, task))
        
        # Execute all analyses concurrently
        results = []
        for i, text, task in tasks:
            try:
                result = await task
                results.append({
                    "text": text,
                    "sentiment": result.sentiment,
                    "flags": result.flags,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "text": text,
                    "error": str(e),
                    "success": False
                })
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\nText {i}:")
            print(f"  {result['text']}")
            if result["success"]:
                print(f"  Sentiment: {result['sentiment']}")
                print(f"  Flags: {result['flags']}")
            else:
                print(f"  Error: {result['error']}")


async def performance_comparison():
    """Compare async vs sync performance (simulated)."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping performance comparison")
        return
    
    print("\n=== Performance Comparison Example ===")
    
    companies = ["AAPL", "MSFT", "GOOGL"]
    question = "What are the main risk factors?"
    
    async with AsyncFinChatClient(base_url="http://localhost:8000") as client:
        # Simulate async execution
        print("Simulating async execution...")
        start_time = time.time()
        
        # Create all tasks at once
        tasks = [
            client.query(question, ticker, "10-K") 
            for ticker in companies
        ]
        
        # Wait for all to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            async_time = time.time() - start_time
            
            successful_results = sum(1 for r in results if isinstance(r, QueryResponse))
            
            print(f"Async execution: {async_time:.2f} seconds")
            print(f"Successful queries: {successful_results}/{len(companies)}")
            
            # Simulate sequential execution time (would be much longer)
            estimated_sequential_time = async_time * len(companies)
            print(f"Estimated sequential time: {estimated_sequential_time:.2f} seconds")
            print(f"Performance improvement: {estimated_sequential_time / async_time:.1f}x faster")
            
        except Exception as e:
            print(f"Error in performance test: {e}")


async def advanced_error_handling():
    """Demonstrate advanced async error handling patterns."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping advanced error handling")
        return
    
    print("\n=== Advanced Async Error Handling ===")
    
    async with AsyncFinChatClient(base_url="http://localhost:8000", timeout=5) as client:
        # Test various error scenarios
        test_cases = [
            ("Valid query", "AAPL", "What is the revenue?", "10-K"),
            ("Invalid ticker", "INVALID", "What is the revenue?", "10-K"),
            ("Empty question", "AAPL", "", "10-K"),
            ("Invalid form type", "AAPL", "What is the revenue?", "INVALID"),
        ]
        
        for description, ticker, question, form_type in test_cases:
            print(f"\nTesting: {description}")
            try:
                result = await client.query(question, ticker, form_type)
                print(f"  ✅ Success: {len(result.answer)} characters in answer")
            
            except FinChatValidationError as e:
                print(f"  ⚠️  Validation error: {e.message}")
            
            except FinChatNotFoundError as e:
                print(f"  ⚠️  Not found: {e.message}")
            
            except FinChatConnectionError as e:
                print(f"  ❌ Connection error: {e.message}")
            
            except FinChatAPIError as e:
                print(f"  ❌ API error: {e.message}")
            
            except Exception as e:
                print(f"  ❌ Unexpected error: {str(e)}")


async def timeout_and_retry_example():
    """Demonstrate timeout and retry behavior."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping timeout example")
        return
    
    print("\n=== Timeout and Retry Example ===")
    
    # Create client with short timeout for demonstration
    async with AsyncFinChatClient(
        base_url="http://localhost:8000",
        timeout=2,  # Very short timeout
        max_retries=2,
        retry_delay=0.5
    ) as client:
        
        print("Testing with short timeout (2 seconds)...")
        
        try:
            # This might timeout if the query takes too long
            result = await client.query(
                "Provide a detailed analysis of all business segments and their performance",
                "AAPL",
                "10-K"
            )
            print("✅ Query completed despite short timeout")
            
        except FinChatConnectionError as e:
            print(f"⚠️  Connection/timeout error: {e.message}")
            print("This is expected with a very short timeout")
        
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")


async def health_monitoring_example():
    """Demonstrate async health monitoring."""
    if not SDK_AVAILABLE:
        print("SDK not available - skipping health monitoring")
        return
    
    print("\n=== Async Health Monitoring Example ===")
    
    async with AsyncFinChatClient(base_url="http://localhost:8000") as client:
        try:
            # Check health before making requests
            health = await client.health_check()
            
            print(f"Service Status: {health.status}")
            print(f"Service Version: {health.version}")
            
            if health.is_healthy:
                print("✅ Service is healthy - proceeding with queries")
                
                # Make a test query since service is healthy
                result = await client.query(
                    "What is the company name?", 
                    "AAPL", 
                    "10-K"
                )
                print(f"Test query successful: {len(result.answer)} characters")
                
            else:
                print("⚠️  Service is not fully healthy")
                print("Individual service status:")
                if health.services:
                    for service, status in health.services.items():
                        emoji = "✅" if status == "ready" else "❌"
                        print(f"  {service}: {status} {emoji}")
        
        except FinChatConnectionError as e:
            print(f"❌ Cannot reach service: {e.message}")


async def main():
    """Run all async examples."""
    print("FinChat SEC QA SDK - Async Examples")
    print("=" * 50)
    
    if not SDK_AVAILABLE:
        print("❌ SDK is not available. Please install with:")
        print("   pip install finchat-sec-qa[sdk]")
        return
    
    # Run async examples
    await health_monitoring_example()
    await basic_async_example()
    await concurrent_queries_example()
    await batch_risk_analysis()
    await performance_comparison()
    await advanced_error_handling()
    await timeout_and_retry_example()
    
    print("\n" + "=" * 50)
    print("Async examples completed! 🚀")
    print("\nKey benefits of async usage:")
    print("- Much better performance for multiple queries")
    print("- Non-blocking operations")
    print("- Efficient resource utilization")
    print("- Better scalability for high-throughput applications")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())