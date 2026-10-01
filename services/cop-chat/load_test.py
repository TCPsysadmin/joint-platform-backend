#!/usr/bin/env python3
"""
Simple load testing script for the RAG chatbot streaming endpoint.
Tests concurrent connections and rate limiting behavior.

Usage:
    python load_test.py --url http://localhost:8000 --concurrent 10 --requests 50
"""

import asyncio
import aiohttp
import time
import json
import argparse
from typing import List, Dict, Any


async def test_streaming_endpoint(
    session: aiohttp.ClientSession,
    url: str,
    message: str,
    request_id: int
) -> Dict[str, Any]:
    """Test a single streaming request."""
    start_time = time.time()
    
    try:
        payload = {
            "message": f"{message} (request {request_id})",
            "max_tokens": 100
        }
        
        async with session.post(
            f"{url}/chat/stream",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status != 200:
                return {
                    "request_id": request_id,
                    "success": False,
                    "status": response.status,
                    "error": await response.text(),
                    "duration": time.time() - start_time
                }
            
            # Read streaming response
            content_chunks = []
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if data.get('type') == 'content':
                            content_chunks.append(data.get('content', ''))
                    except json.JSONDecodeError:
                        pass
            
            return {
                "request_id": request_id,
                "success": True,
                "status": response.status,
                "content_length": len(''.join(content_chunks)),
                "chunks_received": len(content_chunks),
                "duration": time.time() - start_time
            }
            
    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
            "duration": time.time() - start_time
        }


async def run_load_test(
    base_url: str,
    concurrent_requests: int,
    total_requests: int,
    message: str = "What is The Collaborative Process?"
) -> None:
    """Run concurrent load test against the streaming endpoint."""
    
    print(f"🚀 Starting load test:")
    print(f"   URL: {base_url}")
    print(f"   Concurrent requests: {concurrent_requests}")
    print(f"   Total requests: {total_requests}")
    print(f"   Message: {message}")
    print()
    
    # Create session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=concurrent_requests * 2,
        limit_per_host=concurrent_requests * 2
    )
    
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout
    ) as session:
        
        # Test health endpoint first
        try:
            async with session.get(f"{base_url}/health/simple") as response:
                if response.status == 200:
                    print("✅ Health check passed")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return
        
        # Run load test in batches
        results = []
        start_time = time.time()
        
        for batch_start in range(0, total_requests, concurrent_requests):
            batch_end = min(batch_start + concurrent_requests, total_requests)
            batch_size = batch_end - batch_start
            
            print(f"📊 Running batch {batch_start + 1}-{batch_end}...")
            
            # Create tasks for this batch
            tasks = [
                test_streaming_endpoint(session, base_url, message, i)
                for i in range(batch_start, batch_end)
            ]
            
            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({
                        "success": False,
                        "error": str(result),
                        "duration": 0
                    })
                else:
                    results.append(result)
            
            # Brief pause between batches
            if batch_end < total_requests:
                await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        if successful:
            avg_duration = sum(r["duration"] for r in successful) / len(successful)
            min_duration = min(r["duration"] for r in successful)
            max_duration = max(r["duration"] for r in successful)
        else:
            avg_duration = min_duration = max_duration = 0
        
        print("\n📈 Load Test Results:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"   Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Requests/second: {len(results)/total_time:.2f}")
        print(f"   Avg response time: {avg_duration:.2f}s")
        print(f"   Min response time: {min_duration:.2f}s")
        print(f"   Max response time: {max_duration:.2f}s")
        
        # Show error breakdown
        if failed:
            print("\n❌ Error breakdown:")
            error_counts = {}
            for result in failed:
                error_key = f"HTTP {result.get('status', 'Unknown')}" if result.get('status') else result.get('error', 'Unknown')
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            for error, count in error_counts.items():
                print(f"   {error}: {count}")


async def test_rate_limiting(base_url: str) -> None:
    """Test rate limiting behavior."""
    print("🔒 Testing rate limiting...")
    
    async with aiohttp.ClientSession() as session:
        # Send requests rapidly to trigger rate limiting
        tasks = []
        for i in range(70):  # Exceed default 60 RPM limit
            task = test_streaming_endpoint(
                session, base_url, "Rate limit test", i
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        rate_limited = sum(
            1 for r in results 
            if isinstance(r, dict) and r.get("status") == 429
        )
        
        print(f"   Rate limited responses: {rate_limited}/70")
        if rate_limited > 0:
            print("   ✅ Rate limiting is working")
        else:
            print("   ⚠️  Rate limiting may not be configured")


def main():
    parser = argparse.ArgumentParser(description="Load test the RAG chatbot")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=50, help="Total requests")
    parser.add_argument("--message", default="What is The Collaborative Process?", help="Test message")
    parser.add_argument("--test-rate-limit", action="store_true", help="Test rate limiting")
    
    args = parser.parse_args()
    
    async def run_tests():
        if args.test_rate_limit:
            await test_rate_limiting(args.url)
            print()
        
        await run_load_test(
            args.url,
            args.concurrent,
            args.requests,
            args.message
        )
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()