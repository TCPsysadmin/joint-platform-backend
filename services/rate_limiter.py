# services/rate_limiter.py
import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Optional
from fastapi import HTTPException


class TokenBucketRateLimiter:
    """
    Production-ready token bucket rate limiter for handling concurrent users.
    Prevents abuse while allowing burst traffic patterns.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        cleanup_interval: int = 300  # 5 minutes
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.cleanup_interval = cleanup_interval
        
        # Token buckets per client IP
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            'tokens': burst_size,
            'last_refill': time.time(),
            'last_access': time.time()
        })
        
        # Start cleanup task
        self._cleanup_task = None
        
    async def start_cleanup(self):
        """Start the cleanup task for expired buckets."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_buckets())
    
    async def stop_cleanup(self):
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _cleanup_expired_buckets(self):
        """Periodically clean up expired rate limit buckets."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                current_time = time.time()
                expired_keys = [
                    key for key, bucket in self.buckets.items()
                    if current_time - bucket['last_access'] > self.cleanup_interval
                ]
                for key in expired_keys:
                    del self.buckets[key]
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue cleanup
                pass
    
    def _refill_bucket(self, bucket: Dict) -> None:
        """Refill tokens in the bucket based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - bucket['last_refill']
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = current_time
        bucket['last_access'] = current_time
    
    async def check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if the client is within rate limits.
        Returns True if allowed, raises HTTPException if rate limited.
        """
        bucket = self.buckets[client_ip]
        self._refill_bucket(bucket)
        
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        else:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": 60 / self.requests_per_minute,
                    "limit": self.requests_per_minute,
                    "window": "1 minute"
                }
            )


class ConnectionManager:
    """
    Manages active streaming connections for monitoring and graceful shutdown.
    """
    
    def __init__(self, max_connections: int = 1000):
        self.max_connections = max_connections
        self.active_connections: Dict[str, Dict] = {}
        
    def add_connection(self, connection_id: str, client_ip: str) -> None:
        """Add a new streaming connection."""
        if len(self.active_connections) >= self.max_connections:
            raise HTTPException(
                status_code=503,
                detail="Server at capacity. Please try again later."
            )
            
        self.active_connections[connection_id] = {
            'client_ip': client_ip,
            'start_time': time.time(),
            'last_activity': time.time()
        }
    
    def update_activity(self, connection_id: str) -> None:
        """Update last activity time for a connection."""
        if connection_id in self.active_connections:
            self.active_connections[connection_id]['last_activity'] = time.time()
    
    def remove_connection(self, connection_id: str) -> None:
        """Remove a streaming connection."""
        self.active_connections.pop(connection_id, None)
    
    def get_connection_count(self) -> int:
        """Get current number of active connections."""
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics for monitoring."""
        if not self.active_connections:
            return {
                'active_connections': 0,
                'avg_duration': 0,
                'oldest_connection': 0
            }
        
        current_time = time.time()
        durations = [
            current_time - conn['start_time'] 
            for conn in self.active_connections.values()
        ]
        
        return {
            'active_connections': len(self.active_connections),
            'avg_duration': sum(durations) / len(durations),
            'oldest_connection': max(durations),
            'max_connections': self.max_connections
        }