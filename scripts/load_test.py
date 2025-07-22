#!/usr/bin/env python3
"""Load testing script for FinChat SEC QA API performance measurement."""

import asyncio
import aiohttp
import time
import statistics
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    endpoint: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    errors: List[str]


class LoadTester:
    """Async load tester for API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> tuple[float, bool, Optional[str]]:
        """Make a single request and measure response time."""
        start_time = time.time()
        
        try:
            async with self.session.request(method, f"{self.base_url}{endpoint}", **kwargs) as response:
                await response.text()  # Consume response body
                response_time = time.time() - start_time
                success = response.status < 400
                error = None if success else f"HTTP {response.status}"
                return response_time, success, error
                
        except Exception as e:
            response_time = time.time() - start_time
            return response_time, False, str(e)
    
    async def run_concurrent_requests(
        self, 
        method: str, 
        endpoint: str, 
        num_requests: int, 
        concurrency: int,
        **kwargs
    ) -> List[tuple[float, bool, Optional[str]]]:
        """Run multiple concurrent requests."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request():
            async with semaphore:
                return await self.make_request(method, endpoint, **kwargs)
        
        tasks = [bounded_request() for _ in range(num_requests)]
        return await asyncio.gather(*tasks)
    
    def analyze_results(
        self, 
        endpoint: str, 
        results: List[tuple[float, bool, Optional[str]]], 
        duration: float
    ) -> LoadTestResult:
        """Analyze load test results."""
        response_times = [r[0] for r in results]
        successes = [r[1] for r in results]
        errors = [r[2] for r in results if r[2] is not None]
        
        successful_requests = sum(successes)
        failed_requests = len(results) - successful_requests
        
        return LoadTestResult(
            endpoint=endpoint,
            total_requests=len(results),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=statistics.mean(response_times),
            median_response_time=statistics.median(response_times),
            p95_response_time=self._percentile(response_times, 95),
            p99_response_time=self._percentile(response_times, 99),
            requests_per_second=len(results) / duration,
            error_rate=failed_requests / len(results) * 100,
            errors=errors[:10]  # Limit error samples
        )
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def test_health_endpoint(self, requests: int = 100, concurrency: int = 10) -> LoadTestResult:
        """Load test the health endpoint."""
        start_time = time.time()
        results = await self.run_concurrent_requests("GET", "/health", requests, concurrency)
        duration = time.time() - start_time
        return self.analyze_results("/health", results, duration)
    
    async def test_metrics_endpoint(self, requests: int = 100, concurrency: int = 10) -> LoadTestResult:
        """Load test the metrics endpoint."""  
        start_time = time.time()
        results = await self.run_concurrent_requests("GET", "/metrics", requests, concurrency)
        duration = time.time() - start_time
        return self.analyze_results("/metrics", results, duration)
    
    async def test_query_endpoint(
        self, 
        requests: int = 50, 
        concurrency: int = 5,
        ticker: str = "AAPL",
        question: str = "What is the company's revenue?"
    ) -> LoadTestResult:
        """Load test the query endpoint."""
        payload = {
            "question": question,
            "ticker": ticker,
            "form_type": "10-K",
            "limit": 1
        }
        
        start_time = time.time()
        results = await self.run_concurrent_requests(
            "POST", 
            "/query", 
            requests, 
            concurrency,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        duration = time.time() - start_time
        return self.analyze_results("/query", results, duration)
    
    async def test_risk_endpoint(
        self, 
        requests: int = 100, 
        concurrency: int = 10,
        text: str = "The company shows strong financial performance."
    ) -> LoadTestResult:
        """Load test the risk analysis endpoint."""
        payload = {"text": text}
        
        start_time = time.time()
        results = await self.run_concurrent_requests(
            "POST", 
            "/risk", 
            requests, 
            concurrency,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        duration = time.time() - start_time
        return self.analyze_results("/risk", results, duration)


def print_results(result: LoadTestResult):
    """Print formatted load test results."""
    print(f"\n=== {result.endpoint} Load Test Results ===")
    print(f"Total Requests: {result.total_requests}")
    print(f"Successful: {result.successful_requests}")
    print(f"Failed: {result.failed_requests}")
    print(f"Success Rate: {100 - result.error_rate:.1f}%")
    print(f"Requests/sec: {result.requests_per_second:.2f}")
    print(f"Avg Response Time: {result.avg_response_time * 1000:.1f}ms")
    print(f"Median Response Time: {result.median_response_time * 1000:.1f}ms")
    print(f"95th Percentile: {result.p95_response_time * 1000:.1f}ms")  
    print(f"99th Percentile: {result.p99_response_time * 1000:.1f}ms")
    
    if result.errors:
        print(f"Sample Errors: {result.errors[:3]}")


async def run_load_tests(base_url: str = "http://localhost:8000"):
    """Run comprehensive load tests."""
    print(f"Starting load tests against {base_url}")
    
    async with LoadTester(base_url) as tester:
        # Test different endpoints with appropriate load levels
        results = []
        
        print("Testing health endpoint...")
        results.append(await tester.test_health_endpoint(requests=200, concurrency=20))
        
        print("Testing metrics endpoint...")
        results.append(await tester.test_metrics_endpoint(requests=200, concurrency=20))
        
        print("Testing risk analysis endpoint...")
        results.append(await tester.test_risk_endpoint(requests=100, concurrency=10))
        
        # Note: Query endpoint test is lighter due to potential for actual API calls
        print("Testing query endpoint (light load)...")
        results.append(await tester.test_query_endpoint(requests=10, concurrency=2))
        
        # Print all results
        for result in results:
            print_results(result)
        
        # Generate summary report
        print("\n=== Performance Summary ===")
        avg_rps = statistics.mean([r.requests_per_second for r in results])
        avg_response_time = statistics.mean([r.avg_response_time for r in results])
        overall_success_rate = statistics.mean([100 - r.error_rate for r in results])
        
        print(f"Overall Average RPS: {avg_rps:.2f}")
        print(f"Overall Average Response Time: {avg_response_time * 1000:.1f}ms")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        
        # Save detailed results
        report = {
            "timestamp": time.time(),
            "base_url": base_url,
            "results": [asdict(r) for r in results],
            "summary": {
                "avg_rps": avg_rps,
                "avg_response_time": avg_response_time,
                "success_rate": overall_success_rate
            }
        }
        
        with open("load_test_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed results saved to load_test_results.json")


if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    try:
        asyncio.run(run_load_tests(base_url))
    except KeyboardInterrupt:
        print("\nLoad test interrupted by user")
    except Exception as e:
        print(f"Load test failed: {e}")
        sys.exit(1)