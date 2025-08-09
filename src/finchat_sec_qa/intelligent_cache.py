"""Intelligent caching system with adaptive patterns and machine learning-based optimization."""
import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import weakref

from .metrics import (
    cache_operations_total,
    cache_hit_ratio,
    cache_size_bytes,
    cache_entries_total,
    get_business_tracker
)

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"     # ML-based adaptive policy
    ARC = "arc"              # Adaptive Replacement Cache
    W_TINYLFU = "w_tinylfu"  # Window Tiny LFU


class CacheType(Enum):
    """Types of cached data."""
    QUERY_RESULT = "query_result"
    DOCUMENT_CHUNK = "document_chunk"
    EMBEDDINGS = "embeddings"
    RISK_ANALYSIS = "risk_analysis"
    QUANTUM_CIRCUIT = "quantum_circuit"
    API_RESPONSE = "api_response"
    COMPUTATION = "computation"


@dataclass
class CacheEntry:
    """Enhanced cache entry with comprehensive metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    cache_type: CacheType = CacheType.QUERY_RESULT
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Learning features
    access_pattern: List[float] = field(default_factory=list)
    seasonal_score: float = 0.0
    importance_score: float = 0.0
    computation_cost: float = 1.0  # Relative cost to regenerate
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of the entry in seconds."""
        return time.time() - self.created_at
    
    @property
    def time_since_access(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed
    
    def record_access(self):
        """Record an access to this entry."""
        current_time = time.time()
        self.last_accessed = current_time
        self.access_count += 1
        
        # Maintain access pattern for learning (last 100 accesses)
        self.access_pattern.append(current_time)
        if len(self.access_pattern) > 100:
            self.access_pattern.pop(0)
    
    def calculate_value_score(self) -> float:
        """Calculate the value score for this entry (higher = more valuable)."""
        current_time = time.time()
        
        # Base score components
        frequency_score = min(self.access_count / 100.0, 1.0)
        recency_score = max(0, 1 - (self.time_since_access / 3600))  # Decay over 1 hour
        importance_score = self.importance_score
        computation_cost_score = min(self.computation_cost / 10.0, 1.0)
        
        # Seasonal/pattern score
        pattern_score = self._calculate_pattern_score()
        
        # Combine scores with weights
        weights = {
            'frequency': 0.3,
            'recency': 0.25,
            'importance': 0.2,
            'cost': 0.15,
            'pattern': 0.1
        }
        
        total_score = (
            weights['frequency'] * frequency_score +
            weights['recency'] * recency_score +
            weights['importance'] * importance_score +
            weights['cost'] * computation_cost_score +
            weights['pattern'] * pattern_score
        )
        
        return total_score
    
    def _calculate_pattern_score(self) -> float:
        """Calculate score based on access patterns."""
        if len(self.access_pattern) < 3:
            return 0.0
        
        current_time = time.time()
        recent_accesses = [t for t in self.access_pattern if current_time - t < 3600]  # Last hour
        
        if not recent_accesses:
            return 0.0
        
        # Calculate access frequency in recent period
        frequency = len(recent_accesses) / 3600  # Accesses per second
        
        # Detect regular patterns (simplified)
        if len(self.access_pattern) >= 10:
            intervals = []
            for i in range(1, min(11, len(self.access_pattern))):
                intervals.append(self.access_pattern[i] - self.access_pattern[i-1])
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                regularity = 1.0 / (1.0 + variance / (avg_interval ** 2)) if avg_interval > 0 else 0
                return min(frequency * regularity, 1.0)
        
        return min(frequency, 1.0)


class IntelligentCache:
    """Intelligent cache with adaptive eviction and machine learning optimization."""
    
    def __init__(self, name: str, max_size: int = 10000, max_memory_mb: int = 1024,
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 persistence_path: Optional[str] = None):
        self.name = name
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.persistence_path = persistence_path
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()  # For LRU
        self._frequency_counter = defaultdict(int)  # For LFU
        
        # Adaptive policy state
        self._policy_performance = defaultdict(lambda: {'hits': 0, 'misses': 0, 'evictions': 0})
        self._current_adaptive_policy = CachePolicy.LRU
        self._policy_switch_threshold = 100  # Requests before reconsidering policy
        self._requests_since_switch = 0
        
        # Learning and statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0,
            'avg_computation_cost': 1.0,
            'policy_switches': 0
        }
        
        # Pattern recognition
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._temporal_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Background maintenance
        self._maintenance_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        logger.info(f"Initialized intelligent cache '{name}' with policy {policy.value}, "
                   f"max_size={max_size}, max_memory_mb={max_memory_mb}")
    
    async def get(self, key: str, cache_type: CacheType = CacheType.QUERY_RESULT) -> Optional[Any]:
        """Get a value from the cache with intelligent access tracking."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None or entry.is_expired:
                self.stats['misses'] += 1
                self._record_miss(key, cache_type)
                
                if entry and entry.is_expired:
                    await self._remove_entry(key)
                
                return None
            
            # Record hit and update access patterns
            entry.record_access()
            self.stats['hits'] += 1
            self._record_hit(key, cache_type)
            self._update_access_order(key)
            
            # Update metrics
            cache_operations_total.labels(
                operation='get',
                cache_type=cache_type.value,
                status='hit'
            ).inc()
            
            return entry.value
    
    async def set(self, key: str, value: Any, cache_type: CacheType = CacheType.QUERY_RESULT,
                 ttl_seconds: Optional[float] = None, importance_score: float = 0.5,
                 computation_cost: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set a value in the cache with intelligent storage optimization."""
        async with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if we need to make space
            while (len(self._cache) >= self.max_size or 
                   self.stats['size_bytes'] + size_bytes > self.max_memory_bytes):
                if not await self._evict_entry():
                    logger.warning(f"Cache '{self.name}' cannot evict entries to make space")
                    return False
            
            # Create cache entry
            current_time = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                cache_type=cache_type,
                importance_score=importance_score,
                computation_cost=computation_cost,
                metadata=metadata or {}
            )
            
            # Store entry
            old_entry = self._cache.get(key)
            if old_entry:
                self.stats['size_bytes'] -= old_entry.size_bytes
            
            self._cache[key] = entry
            self.stats['size_bytes'] += size_bytes
            self._update_access_order(key)
            
            # Update learning metrics
            if computation_cost > 0:
                self.stats['avg_computation_cost'] = (
                    (self.stats['avg_computation_cost'] * 0.9) +
                    (computation_cost * 0.1)
                )
            
            # Update metrics
            cache_operations_total.labels(
                operation='set',
                cache_type=cache_type.value,
                status='success'
            ).inc()
            
            cache_entries_total.labels(cache_type=self.name).set(len(self._cache))
            cache_size_bytes.labels(cache_type=self.name).set(self.stats['size_bytes'])
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete a specific key from the cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
            return False
    
    async def clear(self):
        """Clear all entries from the cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_counter.clear()
            self.stats['size_bytes'] = 0
            
            cache_entries_total.labels(cache_type=self.name).set(0)
            cache_size_bytes.labels(cache_type=self.name).set(0)
    
    async def _evict_entry(self) -> bool:
        """Evict an entry based on the current policy."""
        if not self._cache:
            return False
        
        if self.policy == CachePolicy.ADAPTIVE:
            # Use the current adaptive policy
            policy = self._current_adaptive_policy
        else:
            policy = self.policy
        
        victim_key = None
        
        if policy == CachePolicy.LRU:
            victim_key = self._find_lru_victim()
        elif policy == CachePolicy.LFU:
            victim_key = self._find_lfu_victim()
        elif policy == CachePolicy.TTL:
            victim_key = self._find_ttl_victim()
        elif policy == CachePolicy.ARC:
            victim_key = self._find_arc_victim()
        else:
            # Default to intelligent scoring
            victim_key = self._find_intelligent_victim()
        
        if victim_key:
            await self._remove_entry(victim_key)
            self.stats['evictions'] += 1
            return True
        
        return False
    
    def _find_lru_victim(self) -> Optional[str]:
        """Find least recently used entry."""
        while self._access_order:
            key = self._access_order.popleft()
            if key in self._cache:
                return key
        return None
    
    def _find_lfu_victim(self) -> Optional[str]:
        """Find least frequently used entry."""
        if not self._cache:
            return None
        
        min_freq = float('inf')
        victim_key = None
        
        for key, entry in self._cache.items():
            if entry.access_count < min_freq:
                min_freq = entry.access_count
                victim_key = key
        
        return victim_key
    
    def _find_ttl_victim(self) -> Optional[str]:
        """Find expired entry or oldest entry."""
        current_time = time.time()
        
        # First, find expired entries
        for key, entry in self._cache.items():
            if entry.is_expired:
                return key
        
        # If no expired entries, find oldest
        oldest_time = current_time
        victim_key = None
        
        for key, entry in self._cache.items():
            if entry.created_at < oldest_time:
                oldest_time = entry.created_at
                victim_key = key
        
        return victim_key
    
    def _find_intelligent_victim(self) -> Optional[str]:
        """Find victim using intelligent scoring."""
        if not self._cache:
            return None
        
        min_score = float('inf')
        victim_key = None
        
        for key, entry in self._cache.items():
            score = entry.calculate_value_score()
            if score < min_score:
                min_score = score
                victim_key = key
        
        return victim_key
    
    def _find_arc_victim(self) -> Optional[str]:
        """Simplified ARC (Adaptive Replacement Cache) victim selection."""
        # This is a simplified version; full ARC is more complex
        return self._find_intelligent_victim()
    
    async def _remove_entry(self, key: str):
        """Remove an entry from the cache."""
        entry = self._cache.pop(key, None)
        if entry:
            self.stats['size_bytes'] -= entry.size_bytes
            
            # Remove from access structures
            try:
                while key in self._access_order:
                    self._access_order.remove(key)
            except ValueError:
                pass
            
            if key in self._frequency_counter:
                del self._frequency_counter[key]
            
            # Update metrics
            cache_operations_total.labels(
                operation='evict',
                cache_type=entry.cache_type.value,
                status='success'
            ).inc()
    
    def _update_access_order(self, key: str):
        """Update access order for LRU policy."""
        # Remove existing occurrences
        try:
            while key in self._access_order:
                self._access_order.remove(key)
        except ValueError:
            pass
        
        # Add to end (most recent)
        self._access_order.append(key)
        
        # Update frequency
        self._frequency_counter[key] += 1
    
    def _record_hit(self, key: str, cache_type: CacheType):
        """Record cache hit for learning."""
        current_time = time.time()
        hour = int(current_time // 3600)
        
        # Update temporal patterns
        self._temporal_patterns[cache_type.value][str(hour)] += 1
        
        # Update hit ratio metrics
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests > 0:
            hit_ratio = self.stats['hits'] / total_requests
            cache_hit_ratio.labels(cache_type=self.name).set(hit_ratio)
        
        # Adaptive policy learning
        if self.policy == CachePolicy.ADAPTIVE:
            self._policy_performance[self._current_adaptive_policy]['hits'] += 1
            self._requests_since_switch += 1
            
            if self._requests_since_switch >= self._policy_switch_threshold:
                # Schedule async evaluation
                asyncio.create_task(self._evaluate_policy_performance())
    
    def _record_miss(self, key: str, cache_type: CacheType):
        """Record cache miss for learning."""
        # Update hit ratio metrics
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests > 0:
            hit_ratio = self.stats['hits'] / total_requests
            cache_hit_ratio.labels(cache_type=self.name).set(hit_ratio)
        
        # Adaptive policy learning
        if self.policy == CachePolicy.ADAPTIVE:
            self._policy_performance[self._current_adaptive_policy]['misses'] += 1
            self._requests_since_switch += 1
    
    async def _evaluate_policy_performance(self):
        """Evaluate and potentially switch adaptive policy."""
        current_policy_stats = self._policy_performance[self._current_adaptive_policy]
        current_hit_rate = current_policy_stats['hits'] / (
            current_policy_stats['hits'] + current_policy_stats['misses']
        ) if (current_policy_stats['hits'] + current_policy_stats['misses']) > 0 else 0
        
        # Try different policies and compare performance
        best_policy = self._current_adaptive_policy
        best_hit_rate = current_hit_rate
        
        for policy in [CachePolicy.LRU, CachePolicy.LFU]:
            if policy == self._current_adaptive_policy:
                continue
                
            policy_stats = self._policy_performance[policy]
            if policy_stats['hits'] + policy_stats['misses'] > 50:  # Sufficient data
                hit_rate = policy_stats['hits'] / (
                    policy_stats['hits'] + policy_stats['misses']
                )
                if hit_rate > best_hit_rate * 1.05:  # 5% improvement threshold
                    best_policy = policy
                    best_hit_rate = hit_rate
        
        if best_policy != self._current_adaptive_policy:
            logger.info(f"Cache '{self.name}' switching adaptive policy from "
                       f"{self._current_adaptive_policy.value} to {best_policy.value} "
                       f"(hit rate improvement: {current_hit_rate:.3f} -> {best_hit_rate:.3f})")
            
            self._current_adaptive_policy = best_policy
            self.stats['policy_switches'] += 1
        
        self._requests_since_switch = 0
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value[:100])  # Sample first 100
            elif isinstance(value, dict):
                size = 0
                for k, v in list(value.items())[:100]:  # Sample first 100
                    size += self._estimate_size(k) + self._estimate_size(v)
                return size
            else:
                # Use pickle for other objects
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimation
            return 1024  # 1KB default
    
    async def start_maintenance(self):
        """Start background maintenance task."""
        if self._maintenance_task and not self._maintenance_task.done():
            return
        
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info(f"Started maintenance task for cache '{self.name}'")
    
    async def stop_maintenance(self):
        """Stop background maintenance task."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
            logger.info(f"Stopped maintenance task for cache '{self.name}'")
    
    async def _maintenance_loop(self):
        """Background maintenance loop."""
        try:
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                async with self._lock:
                    # Remove expired entries
                    expired_keys = []
                    current_time = time.time()
                    
                    for key, entry in self._cache.items():
                        if entry.is_expired:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        await self._remove_entry(key)
                    
                    if expired_keys:
                        logger.debug(f"Cache '{self.name}' removed {len(expired_keys)} expired entries")
                    
                    # Update importance scores based on access patterns
                    await self._update_importance_scores()
                    
                    # Adaptive policy evaluation (if not done recently)
                    if (self.policy == CachePolicy.ADAPTIVE and 
                        self._requests_since_switch < self._policy_switch_threshold // 2):
                        await self._evaluate_policy_performance()
                
        except asyncio.CancelledError:
            logger.debug(f"Maintenance loop cancelled for cache '{self.name}'")
    
    async def _update_importance_scores(self):
        """Update importance scores based on learned patterns."""
        current_time = time.time()
        current_hour = int(current_time // 3600)
        
        for key, entry in self._cache.items():
            # Update seasonal score based on temporal patterns
            cache_type = entry.cache_type.value
            if cache_type in self._temporal_patterns:
                hour_accesses = self._temporal_patterns[cache_type].get(str(current_hour), 0)
                total_accesses = sum(self._temporal_patterns[cache_type].values())
                if total_accesses > 0:
                    entry.seasonal_score = hour_accesses / total_accesses
            
            # Update importance based on access frequency and recency
            if entry.access_count > 0:
                frequency_factor = min(entry.access_count / 100.0, 1.0)
                recency_factor = max(0, 1 - (entry.time_since_access / 3600))
                entry.importance_score = (frequency_factor * 0.7 + recency_factor * 0.3)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests) if total_requests > 0 else 0
        
        # Policy performance summary
        policy_performance = {}
        for policy, stats in self._policy_performance.items():
            policy_total = stats['hits'] + stats['misses']
            if policy_total > 0:
                policy_performance[policy.value] = {
                    'hit_rate': stats['hits'] / policy_total,
                    'total_requests': policy_total,
                    'evictions': stats['evictions']
                }
        
        return {
            'name': self.name,
            'size': len(self._cache),
            'max_size': self.max_size,
            'size_bytes': self.stats['size_bytes'],
            'max_memory_bytes': self.max_memory_bytes,
            'memory_utilization': self.stats['size_bytes'] / self.max_memory_bytes,
            'hit_rate': hit_rate,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'policy': self.policy.value,
            'current_adaptive_policy': (self._current_adaptive_policy.value 
                                      if self.policy == CachePolicy.ADAPTIVE else None),
            'policy_switches': self.stats['policy_switches'],
            'policy_performance': policy_performance,
            'avg_computation_cost': self.stats['avg_computation_cost']
        }
    
    async def optimize_for_workload(self, workload_characteristics: Dict[str, Any]):
        """Optimize cache configuration for specific workload characteristics."""
        # Adjust policy based on workload
        if workload_characteristics.get('access_pattern') == 'sequential':
            if self.policy == CachePolicy.ADAPTIVE:
                self._current_adaptive_policy = CachePolicy.LRU
        elif workload_characteristics.get('access_pattern') == 'random':
            if self.policy == CachePolicy.ADAPTIVE:
                self._current_adaptive_policy = CachePolicy.LFU
        
        # Adjust size based on hit rate goals
        target_hit_rate = workload_characteristics.get('target_hit_rate', 0.8)
        current_hit_rate = (self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) 
                           if (self.stats['hits'] + self.stats['misses']) > 0 else 0)
        
        if current_hit_rate < target_hit_rate and len(self._cache) < self.max_size * 0.8:
            # Could benefit from larger cache
            logger.info(f"Cache '{self.name}' hit rate {current_hit_rate:.3f} below target {target_hit_rate:.3f}")


class CacheManager:
    """Manages multiple intelligent caches with global optimization."""
    
    def __init__(self):
        self._caches: Dict[str, IntelligentCache] = {}
        self._global_memory_limit = 2048 * 1024 * 1024  # 2GB default
        self._lock = asyncio.Lock()
    
    async def create_cache(self, name: str, max_size: int = 10000, 
                          max_memory_mb: int = 512,
                          policy: CachePolicy = CachePolicy.ADAPTIVE) -> IntelligentCache:
        """Create a new intelligent cache."""
        async with self._lock:
            if name in self._caches:
                logger.warning(f"Cache '{name}' already exists")
                return self._caches[name]
            
            cache = IntelligentCache(name, max_size, max_memory_mb, policy)
            self._caches[name] = cache
            await cache.start_maintenance()
            
            logger.info(f"Created intelligent cache '{name}'")
            return cache
    
    async def get_cache(self, name: str) -> Optional[IntelligentCache]:
        """Get an existing cache."""
        return self._caches.get(name)
    
    async def remove_cache(self, name: str) -> bool:
        """Remove a cache."""
        async with self._lock:
            cache = self._caches.pop(name, None)
            if cache:
                await cache.stop_maintenance()
                await cache.clear()
                logger.info(f"Removed cache '{name}'")
                return True
            return False
    
    async def get_global_statistics(self) -> Dict[str, Any]:
        """Get global cache statistics."""
        total_size = 0
        total_memory = 0
        total_hits = 0
        total_misses = 0
        total_evictions = 0
        
        cache_stats = {}
        
        for name, cache in self._caches.items():
            stats = cache.get_statistics()
            cache_stats[name] = stats
            
            total_size += stats['size']
            total_memory += stats['size_bytes']
            total_hits += stats['hits']
            total_misses += stats['misses']
            total_evictions += stats['evictions']
        
        total_requests = total_hits + total_misses
        global_hit_rate = (total_hits / total_requests) if total_requests > 0 else 0
        
        return {
            'global_statistics': {
                'total_caches': len(self._caches),
                'total_entries': total_size,
                'total_memory_bytes': total_memory,
                'memory_limit_bytes': self._global_memory_limit,
                'memory_utilization': total_memory / self._global_memory_limit,
                'global_hit_rate': global_hit_rate,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_evictions': total_evictions
            },
            'cache_details': cache_stats
        }
    
    async def optimize_all_caches(self):
        """Perform global cache optimization."""
        async with self._lock:
            total_memory = sum(cache.stats['size_bytes'] for cache in self._caches.values())
            
            if total_memory > self._global_memory_limit * 0.9:  # 90% memory usage
                # Implement global memory management
                await self._rebalance_memory()
            
            # Share learning between caches
            await self._share_learning_insights()
    
    async def _rebalance_memory(self):
        """Rebalance memory usage across caches."""
        caches_by_efficiency = sorted(
            self._caches.values(),
            key=lambda c: (c.stats['hits'] / (c.stats['hits'] + c.stats['misses']) 
                          if (c.stats['hits'] + c.stats['misses']) > 0 else 0),
            reverse=True
        )
        
        # Reduce memory for least efficient caches
        for cache in caches_by_efficiency[-2:]:  # Bottom 2 caches
            if cache.max_memory_bytes > 64 * 1024 * 1024:  # If larger than 64MB
                cache.max_memory_bytes = int(cache.max_memory_bytes * 0.8)
                logger.info(f"Reduced memory limit for cache '{cache.name}' due to global pressure")
    
    async def _share_learning_insights(self):
        """Share learning insights between caches."""
        # Aggregate temporal patterns
        global_patterns = defaultdict(lambda: defaultdict(int))
        
        for cache in self._caches.values():
            for cache_type, patterns in cache._temporal_patterns.items():
                for hour, count in patterns.items():
                    global_patterns[cache_type][hour] += count
        
        # Share insights back to caches
        for cache in self._caches.values():
            cache._temporal_patterns.update(global_patterns)


# Global cache manager
_cache_manager = CacheManager()


async def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    return _cache_manager


async def create_intelligent_cache(name: str, **kwargs) -> IntelligentCache:
    """Create a new intelligent cache."""
    return await _cache_manager.create_cache(name, **kwargs)


async def get_intelligent_cache(name: str) -> Optional[IntelligentCache]:
    """Get an existing intelligent cache."""
    return await _cache_manager.get_cache(name)


# Convenience decorators
def cached(cache_name: str, ttl_seconds: Optional[float] = None, 
           cache_type: CacheType = CacheType.COMPUTATION,
           key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key_data = {'func': func.__name__, 'args': args, 'kwargs': kwargs}
                key = hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
            
            # Try to get from cache
            cache = await get_intelligent_cache(cache_name)
            if not cache:
                cache = await create_intelligent_cache(cache_name)
            
            result = await cache.get(key, cache_type)
            if result is not None:
                return result
            
            # Compute and cache result
            start_time = time.time()
            result = await func(*args, **kwargs)
            computation_cost = time.time() - start_time
            
            await cache.set(key, result, cache_type, ttl_seconds, 
                          computation_cost=computation_cost)
            return result
        
        def sync_wrapper(*args, **kwargs):
            async def async_func():
                # Generate cache key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    key_data = {'func': func.__name__, 'args': args, 'kwargs': kwargs}
                    key = hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
                
                # Try to get from cache
                cache = await get_intelligent_cache(cache_name)
                if not cache:
                    cache = await create_intelligent_cache(cache_name)
                
                result = await cache.get(key, cache_type)
                if result is not None:
                    return result
                
                # Compute and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                computation_cost = time.time() - start_time
                
                await cache.set(key, result, cache_type, ttl_seconds,
                              computation_cost=computation_cost)
                return result
            
            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_func())
            except RuntimeError:
                return asyncio.run(async_func())
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator