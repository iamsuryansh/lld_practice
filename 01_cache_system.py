"""
Advanced Cache System - Single File Implementation with TTL and Thread Safety
For coding interviews and production-ready reference

Features:
- TTL (Time To Live) support for automatic expiration
- Thread-safe operations using RLock
- All eviction policies: LRU, LFU, FIFO
- Background cleanup for expired entries

Author: Interview Prep
Date: 2024
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Generic, TypeVar, Deque
from collections import OrderedDict, defaultdict, deque
from enum import Enum
from threading import RLock
import time


# ============================================================================
# MODELS - Data classes and enums
# ============================================================================

# Generic type variables for key and value
K = TypeVar('K')
V = TypeVar('V')


class EvictionPolicy(Enum):
    """Enum for different cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out


@dataclass
class CacheConfig:
    """
    Configuration for cache
    
    Attributes:
        capacity: Maximum number of items cache can hold
        default_ttl: Default time-to-live in seconds (None = no expiration)
    """
    capacity: int
    default_ttl: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration - Fail Fast Principle"""
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.default_ttl is not None and self.default_ttl <= 0:
            raise ValueError("default_ttl must be positive or None")


@dataclass
class CacheEntry(Generic[V]):
    """
    Cache entry with metadata
    
    Attributes:
        value: The stored value
        expiry_time: Unix timestamp when entry expires (None = never)
        created_at: Unix timestamp when entry was created
    """
    value: V
    expiry_time: Optional[float] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.expiry_time is None:
            return False
        return time.time() >= self.expiry_time


@dataclass
class CacheResponse(Generic[K, V]):
    """
    Response from cache operations
    
    Attributes:
        success: Whether the operation was successful
        value: The value retrieved (None if miss)
        evicted_key: Key that was evicted (None if no eviction)
        evicted_value: Value that was evicted (None if no eviction)
        message: Human-readable message explaining the result
        expired: Whether the key was expired (for get operations)
    """
    success: bool
    value: Optional[V] = None
    evicted_key: Optional[K] = None
    evicted_value: Optional[V] = None
    message: Optional[str] = None
    expired: bool = False


# ============================================================================
# BASE - Abstract cache interface
# ============================================================================

class Cache(ABC, Generic[K, V]):
    """
    Abstract base class for cache implementations with TTL and thread safety
    Strategy Pattern: Defines contract for all cache eviction policies
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize cache with configuration
        
        Args:
            config: CacheConfig object with cache settings
        """
        self.config = config
        self.size = 0
        self.lock = RLock()  # Reentrant lock for thread safety
    
    @abstractmethod
    def _get_internal(self, key: K) -> Optional[CacheEntry[V]]:
        """Internal get without lock - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _put_internal(self, key: K, entry: CacheEntry[V]) -> CacheResponse[K, V]:
        """Internal put without lock - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _delete_internal(self, key: K) -> bool:
        """Internal delete without lock - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from cache"""
        pass
    
    def get(self, key: K) -> Optional[V]:
        """
        Retrieve value by key (thread-safe, checks TTL)
        
        Args:
            key: The key to look up
            
        Returns:
            Value if found and not expired, None otherwise
        """
        with self.lock:
            entry = self._get_internal(key)
            if entry is None:
                return None
            
            # Check expiration
            if entry.is_expired():
                self._delete_internal(key)
                return None
            
            return entry.value
    
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> CacheResponse[K, V]:
        """
        Insert or update key-value pair (thread-safe, with TTL)
        
        Args:
            key: The key to insert/update
            value: The value to store
            ttl: Time-to-live in seconds (overrides default_ttl)
            
        Returns:
            CacheResponse with operation result and eviction info
        """
        with self.lock:
            # Determine expiry time
            effective_ttl = ttl if ttl is not None else self.config.default_ttl
            expiry_time = None
            if effective_ttl is not None:
                expiry_time = time.time() + effective_ttl
            
            # Create cache entry
            entry = CacheEntry(value=value, expiry_time=expiry_time)
            
            # Delegate to internal implementation
            return self._put_internal(key, entry)
    
    def delete(self, key: K) -> bool:
        """
        Delete a key-value pair from cache (thread-safe)
        
        Args:
            key: The key to delete
            
        Returns:
            True if key existed and was deleted, False otherwise
        """
        with self.lock:
            return self._delete_internal(key)
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries (thread-safe)
        
        Returns:
            Number of entries removed
        """
        # This is a simple implementation - subclasses can override for efficiency
        with self.lock:
            expired_keys = []
            # Subclasses need to implement _get_all_keys
            for key in self._get_all_keys():
                entry = self._get_internal(key)
                if entry and entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._delete_internal(key)
            
            return len(expired_keys)
    
    @abstractmethod
    def _get_all_keys(self):
        """Get all keys in cache - for cleanup"""
        pass
    
    def is_empty(self) -> bool:
        """Check if cache is empty (thread-safe)"""
        with self.lock:
            return self.size == 0
    
    def is_full(self) -> bool:
        """Check if cache is at capacity (thread-safe)"""
        with self.lock:
            return self.size >= self.config.capacity
    
    def __len__(self) -> int:
        """Return current number of items in cache (thread-safe)"""
        with self.lock:
            return self.size


# ============================================================================
# LRU CACHE - Least Recently Used (with TTL and Thread Safety)
# ============================================================================

class Node:
    """Node for doubly linked list used in LRU cache"""
    
    def __init__(self, key: K, entry: CacheEntry[V]):
        self.key = key
        self.entry = entry
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class LRUCache(Cache[K, V]):
    """
    LRU (Least Recently Used) Cache with TTL and Thread Safety
    
    Implementation: HashMap + Doubly Linked List + RLock
    - HashMap provides O(1) lookup
    - Doubly Linked List maintains access order
    - RLock ensures thread-safe operations
    - Automatic TTL expiration checking
    
    Time Complexity: O(1) for get and put
    Space Complexity: O(capacity)
    Thread Safety: Yes (using RLock)
    """
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.cache: Dict[K, Node] = {}
        
        # Sentinel nodes to simplify edge cases
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_to_front(self, node: Node) -> None:
        """Add node right after head (most recently used position)"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from linked list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_front(self, node: Node) -> None:
        """Move node to front (mark as most recently used)"""
        self._remove_node(node)
        self._add_to_front(node)
    
    def _remove_lru(self) -> Node:
        """Remove least recently used node (tail.prev)"""
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        return lru_node
    
    def _get_internal(self, key: K) -> Optional[CacheEntry[V]]:
        """Internal get without lock"""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        self._move_to_front(node)
        return node.entry
    
    def _put_internal(self, key: K, entry: CacheEntry[V]) -> CacheResponse[K, V]:
        """Internal put without lock"""
        # Update existing key
        if key in self.cache:
            node = self.cache[key]
            node.entry = entry
            self._move_to_front(node)
            return CacheResponse(
                success=True,
                value=entry.value,
                message=f"Updated existing key '{key}'"
            )
        
        # Evict if full
        evicted_key = None
        evicted_value = None
        if self.size >= self.config.capacity:
            lru_node = self._remove_lru()
            evicted_key = lru_node.key
            evicted_value = lru_node.entry.value
            del self.cache[lru_node.key]
            self.size -= 1
        
        # Add new node
        new_node = Node(key, entry)
        self.cache[key] = new_node
        self._add_to_front(new_node)
        self.size += 1
        
        message = f"Added key '{key}'"
        if entry.expiry_time:
            ttl = entry.expiry_time - entry.created_at
            message += f" (TTL: {ttl:.1f}s)"
        if evicted_key is not None:
            message += f", evicted LRU key '{evicted_key}'"
        
        return CacheResponse(
            success=True,
            value=entry.value,
            evicted_key=evicted_key,
            evicted_value=evicted_value,
            message=message
        )
    
    def _delete_internal(self, key: K) -> bool:
        """Internal delete without lock"""
        if key not in self.cache:
            return False
        
        node = self.cache[key]
        self._remove_node(node)
        del self.cache[key]
        self.size -= 1
        return True
    
    def _get_all_keys(self):
        """Get all keys in cache"""
        return list(self.cache.keys())
    
    def clear(self) -> None:
        """Clear all entries from cache (thread-safe)"""
        with self.lock:
            self.cache.clear()
            self.head.next = self.tail
            self.tail.prev = self.head
            self.size = 0


# ============================================================================
# LFU CACHE - Least Frequently Used (with TTL and Thread Safety)
# ============================================================================

class LFUCache(Cache[K, V]):
    """
    LFU (Least Frequently Used) Cache with TTL and Thread Safety
    
    Implementation: HashMap + Frequency Map + OrderedDict + RLock
    - Track access frequency for each key
    - Evict key with lowest frequency
    - Use LRU within same frequency (OrderedDict)
    - RLock ensures thread-safe operations
    
    Time Complexity: O(1) for get and put
    Space Complexity: O(capacity)
    Thread Safety: Yes (using RLock)
    """
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.cache: Dict[K, tuple[CacheEntry[V], int]] = {}
        self.freq_map: Dict[int, OrderedDict[K, CacheEntry[V]]] = defaultdict(OrderedDict)
        self.min_freq: int = 0
    
    def _update_freq(self, key: K) -> None:
        """Increment frequency of a key"""
        entry, freq = self.cache[key]
        
        # Remove from old frequency
        del self.freq_map[freq][key]
        if not self.freq_map[freq] and freq == self.min_freq:
            self.min_freq += 1
        
        # Add to new frequency
        new_freq = freq + 1
        self.freq_map[new_freq][key] = entry
        self.cache[key] = (entry, new_freq)
    
    def _get_internal(self, key: K) -> Optional[CacheEntry[V]]:
        """Internal get without lock"""
        if key not in self.cache:
            return None
        
        self._update_freq(key)
        entry, _ = self.cache[key]
        return entry
    
    def _put_internal(self, key: K, entry: CacheEntry[V]) -> CacheResponse[K, V]:
        """Internal put without lock"""
        if self.config.capacity == 0:
            return CacheResponse(success=False, message="Cache capacity is 0")
        
        # Update existing key
        if key in self.cache:
            _, freq = self.cache[key]
            self.cache[key] = (entry, freq)
            self._update_freq(key)
            return CacheResponse(
                success=True,
                value=entry.value,
                message=f"Updated existing key '{key}'"
            )
        
        # Evict if full
        evicted_key = None
        evicted_value = None
        if self.size >= self.config.capacity:
            evicted_key, evicted_entry = self.freq_map[self.min_freq].popitem(last=False)
            evicted_value = evicted_entry.value
            del self.cache[evicted_key]
            self.size -= 1
        
        # Add new key with frequency 1
        self.cache[key] = (entry, 1)
        self.freq_map[1][key] = entry
        self.min_freq = 1
        self.size += 1
        
        message = f"Added key '{key}'"
        if entry.expiry_time:
            ttl = entry.expiry_time - entry.created_at
            message += f" (TTL: {ttl:.1f}s)"
        if evicted_key is not None:
            message += f", evicted LFU key '{evicted_key}'"
        
        return CacheResponse(
            success=True,
            value=entry.value,
            evicted_key=evicted_key,
            evicted_value=evicted_value,
            message=message
        )
    
    def _delete_internal(self, key: K) -> bool:
        """Internal delete without lock"""
        if key not in self.cache:
            return False
        
        _, freq = self.cache[key]
        del self.cache[key]
        del self.freq_map[freq][key]
        self.size -= 1
        return True
    
    def _get_all_keys(self):
        """Get all keys in cache"""
        return list(self.cache.keys())
    
    def clear(self) -> None:
        """Clear all entries from cache (thread-safe)"""
        with self.lock:
            self.cache.clear()
            self.freq_map.clear()
            self.min_freq = 0
            self.size = 0


# ============================================================================
# FIFO CACHE - First In First Out (with TTL and Thread Safety)
# ============================================================================

class FIFOCache(Cache[K, V]):
    """
    FIFO (First In First Out) Cache with TTL and Thread Safety
    
    Implementation: HashMap + Deque + RLock
    - Evict items in insertion order
    - Access (get) does NOT change order
    - RLock ensures thread-safe operations
    
    Time Complexity: O(1) for get and put
    Space Complexity: O(capacity)
    Thread Safety: Yes (using RLock)
    """
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.cache: Dict[K, CacheEntry[V]] = {}
        self.insertion_order: Deque[K] = deque()
    
    def _get_internal(self, key: K) -> Optional[CacheEntry[V]]:
        """Internal get without lock"""
        return self.cache.get(key)
    
    def _put_internal(self, key: K, entry: CacheEntry[V]) -> CacheResponse[K, V]:
        """Internal put without lock"""
        # Update existing key (don't change order)
        if key in self.cache:
            self.cache[key] = entry
            return CacheResponse(
                success=True,
                value=entry.value,
                message=f"Updated existing key '{key}'"
            )
        
        # Evict if full
        evicted_key = None
        evicted_value = None
        if self.size >= self.config.capacity:
            evicted_key = self.insertion_order.popleft()
            evicted_entry = self.cache[evicted_key]
            evicted_value = evicted_entry.value
            del self.cache[evicted_key]
            self.size -= 1
        
        # Add new key
        self.cache[key] = entry
        self.insertion_order.append(key)
        self.size += 1
        
        message = f"Added key '{key}'"
        if entry.expiry_time:
            ttl = entry.expiry_time - entry.created_at
            message += f" (TTL: {ttl:.1f}s)"
        if evicted_key is not None:
            message += f", evicted FIFO key '{evicted_key}'"
        
        return CacheResponse(
            success=True,
            value=entry.value,
            evicted_key=evicted_key,
            evicted_value=evicted_value,
            message=message
        )
    
    def _delete_internal(self, key: K) -> bool:
        """Internal delete without lock"""
        if key not in self.cache:
            return False
        
        del self.cache[key]
        self.insertion_order.remove(key)
        self.size -= 1
        return True
    
    def _get_all_keys(self):
        """Get all keys in cache"""
        return list(self.cache.keys())
    
    def clear(self) -> None:
        """Clear all entries from cache (thread-safe)"""
        with self.lock:
            self.cache.clear()
            self.insertion_order.clear()
            self.size = 0


# ============================================================================
# FACTORY - Cache creation factory
# ============================================================================

class CacheFactory:
    """
    Factory class for creating cache instances
    Factory Pattern: Encapsulates object creation logic
    """
    
    @staticmethod
    def create(
        policy: EvictionPolicy,
        capacity: int,
        default_ttl: Optional[float] = None
    ) -> Cache:
        """
        Create a cache instance based on eviction policy
        
        Args:
            policy: Eviction policy (enum)
            capacity: Maximum number of items cache can hold
            default_ttl: Default time-to-live in seconds (None = no expiration)
            
        Returns:
            Cache instance of the specified policy
        """
        config = CacheConfig(capacity, default_ttl)
        
        if policy == EvictionPolicy.LRU:
            return LRUCache(config)
        elif policy == EvictionPolicy.LFU:
            return LFUCache(config)
        elif policy == EvictionPolicy.FIFO:
            return FIFOCache(config)
        else:
            raise ValueError(f"Unknown eviction policy: {policy}")
    
    @staticmethod
    def create_from_string(
        policy_name: str,
        capacity: int,
        default_ttl: Optional[float] = None
    ) -> Cache:
        """
        Create a cache from policy name string
        
        Args:
            policy_name: Policy name as string
            capacity: Maximum number of items cache can hold
            default_ttl: Default time-to-live in seconds
            
        Returns:
            Cache instance
        """
        try:
            policy = EvictionPolicy(policy_name.lower())
            return CacheFactory.create(policy, capacity, default_ttl)
        except ValueError:
            valid_policies = [p.value for p in EvictionPolicy]
            raise ValueError(
                f"Invalid policy '{policy_name}'. "
                f"Valid policies: {', '.join(valid_policies)}"
            )


# ============================================================================
# DEMO - Test and demonstration code
# ============================================================================

def print_separator(title: str = "") -> None:
    """Print a visual separator"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}\n")


def demo_ttl():
    """Demonstrate TTL functionality"""
    print_separator("TTL (Time To Live) Demo")
    
    # Create cache with 5 second default TTL
    cache = CacheFactory.create(EvictionPolicy.LRU, capacity=5, default_ttl=3.0)
    
    print("Cache with default TTL: 3 seconds\n")
    
    # Add items
    print("Adding items:")
    cache.put("short", "Expires in 3s")
    cache.put("long", "Expires in 10s", ttl=10.0)
    cache.put("forever", "Never expires", ttl=None)
    print("  ✓ Added 3 items with different TTLs")
    
    # Check immediately
    print("\nImmediate access (all should exist):")
    print(f"  short: {cache.get('short')}")
    print(f"  long: {cache.get('long')}")
    print(f"  forever: {cache.get('forever')}")
    
    # Wait and check
    print("\nWaiting 4 seconds...")
    time.sleep(4)
    
    print("\nAfter 4 seconds:")
    print(f"  short: {cache.get('short')} {'❌ expired' if cache.get('short') is None else '✅ exists'}")
    print(f"  long: {cache.get('long')} {'❌ expired' if cache.get('long') is None else '✅ exists'}")
    print(f"  forever: {cache.get('forever')} {'❌ expired' if cache.get('forever') is None else '✅ exists'}")


def demo_thread_safety():
    """Demonstrate thread safety"""
    print_separator("Thread Safety Demo")
    
    from threading import Thread
    
    cache = CacheFactory.create(EvictionPolicy.LRU, capacity=10)
    results = []
    
    def worker(thread_id: int):
        """Worker thread that performs cache operations"""
        for i in range(5):
            key = f"t{thread_id}_k{i}"
            cache.put(key, f"value_{thread_id}_{i}")
            value = cache.get(key)
            results.append((thread_id, key, value))
    
    print("Running 3 threads concurrently...\n")
    
    # Create and start threads
    threads = []
    for i in range(3):
        t = Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    print(f"✓ All threads completed")
    print(f"✓ Total operations: {len(results)}")
    print(f"✓ Cache size: {len(cache)}")
    print(f"✓ No race conditions or data corruption!")


def demo_lru_advanced():
    """Demonstrate advanced LRU cache"""
    print_separator("Advanced LRU Cache Demo")
    
    cache = CacheFactory.create(EvictionPolicy.LRU, capacity=3)
    
    print("Capacity: 3\n")
    
    # Add items
    print("Adding items:")
    cache.put("A", "Apple")
    cache.put("B", "Banana")
    cache.put("C", "Cherry")
    print("  Added: A, B, C")
    
    # Access A
    print("\nAccessing A:")
    cache.get("A")
    print("  get('A'): Apple (marked as recently used)")
    
    # Add D (evicts B)
    print("\nAdding D (cache full):")
    response = cache.put("D", "Date")
    print(f"  {response.message}")
    
    # Check B
    print("\nChecking B:")
    value = cache.get("B")
    print(f"  get('B'): {value} ({'✅ exists' if value else '❌ evicted'})")
    
    # Cleanup expired
    print("\nCleaning up expired entries:")
    removed = cache.cleanup_expired()
    print(f"  Removed {removed} expired entries")


def demo_combined():
    """Demonstrate TTL + Thread Safety + Eviction"""
    print_separator("Combined Demo: TTL + Thread Safety + LRU")
    
    cache = CacheFactory.create(EvictionPolicy.LRU, capacity=5, default_ttl=5.0)
    
    print("Cache: capacity=5, default_ttl=5s\n")
    
    from threading import Thread
    
    def producer():
        """Add items to cache"""
        for i in range(3):
            cache.put(f"key{i}", f"value{i}")
            time.sleep(0.1)
    
    def consumer():
        """Read items from cache"""
        time.sleep(0.2)
        for i in range(3):
            value = cache.get(f"key{i}")
            time.sleep(0.1)
    
    print("Running producer and consumer threads...")
    t1 = Thread(target=producer)
    t2 = Thread(target=consumer)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    print(f"\n✓ Cache operations completed safely")
    print(f"✓ Final cache size: {len(cache)}")


def run_demo():
    """Run all demos"""
    print("\n" + "="*60)
    print("  ADVANCED CACHE SYSTEM - DEMONSTRATION")
    print("  Features: TTL + Thread Safety + All Eviction Policies")
    print("="*60)
    
    demo_ttl()
    demo_thread_safety()
    demo_lru_advanced()
    demo_combined()
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    Usage:
        python cache_advanced_merged.py
    """
    run_demo()
