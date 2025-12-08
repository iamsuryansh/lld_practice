from enum import Enum
import time

class EvicitionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"

from typing import TypeVar
K = TypeVar('K')
V = TypeVar('V')

from dataclasses import dataclass
from typing import Optional
from typing import Generic
from abc import ABC, abstractmethod
from threading import RLock
from collections import defaultdict
from typing import OrderedDict


@dataclass
class CacheConfig:
    capacity: int
    default_ttl: Optional[float] = None

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("Capacity can not be negative or zero")
        if self.default_ttl <= 0:
            raise ValueError("Default ttl can not be negative or zero")


@dataclass
class CacheEntry(Generic[V]):
    value: V
    expiry_time: Optional[float] = None
    created_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        
    def is_expired(self) -> bool:
        if self.expiry_time is None:
            return False
        return self.expiry_time <= time.time()


@dataclass
class CacheResponse(Generic[K, V]):
    success: bool
    value: Optional[V] = None
    evicted_key: Optional[K] = None
    evicted_value: Optional[V] = None
    message : Optional[str] = None
    expired : bool = False



class Cache(ABC, Generic[K, V]):
    def __init__(self, config: CacheConfig):
        self.config = config
        self.size = 0
        self.lock = RLock()

    @abstractmethod
    def _get_internal(self, key: K) -> Optional[CacheEntry[V]]:
        pass

    @abstractmethod
    def _put_internal(self, key: K, entry: CacheEntry[V]) -> Optional[CacheResponse[K, V]]:
        pass

    @abstractmethod
    def _delete_internal(self, key: K) -> bool:
        pass

    @abstractmethod
    def clear(self):
        pass

    def get(self, key: K) -> Optional[V]:
        with self.lock:
            entry = self._get_internal(key)
            if entry is None:
                return None

            if entry.is_expired:
                self._delete_internal(key)
                return None

            return entry.value
    
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> CacheResponse[K, V]:
        with self.lock:
            effective_ttl = ttl if ttl is not None else self.config.default_ttl
            expiry_time = None

            if effective_ttl is not None:
                expiry_time = time.time() + effective_ttl
            
            entry = CacheEntry(value=value, expiry_time=expiry_time)
            return self._put_internal(key, entry)
    
    def delete(self, key: K) -> bool:
        with self.lock:
            return self._delete_internal(key)
    

#1. LRU

class Node:
    def __init__(self, key: K, entry: CacheEntry):
        self.key = key
        self.entry = entry
        self.next: Optional[Node] = None
        self.prev: Optional[Node] = None

from typing import Dict

class LRUCache(Cache[K, V]):
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.cache : Dict[K, Node] = {}
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_to_front(self, node: Node) -> None:
        first_node = self.head
        node.next = first_node
        first_node.prev = node

        self.head.next = node
        node.prev = self.head
    
    def _remove_node(self, node: Node) -> None:
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_front(self, node: Node) -> None:
        self._remove_node(node)
        self._add_to_front(node)
    
    def _remove_lru_node(self) -> Node:
        lru_node = self.tail_prev
        self._remove_node(lru_node)
        return lru_node
    

    def _get_internal(self, key: K) -> Optional[CacheEntry[V]]:
        if key not in self.cache:
            return None
    
        node = self.cache[key]
        self._move_to_front(node)
        return node.entry
    

    def _put_internal(self, key: K, entry: CacheEntry[V]) -> Optional[CacheResponse[K, V]]:
        # already exists (no change to cache size)
        if key in self.cache:
            node = self.cache[key]
            node.entry = entry
            self._move_to_front(node)
            
            return CacheResponse(
                success=True,
                value=entry.value,
                message=f"Updated exisiting key {key}"
            )

        # does not exist, cache size will change, check if full
        evicted_key = None
        evicted_value = None

        if self.size >= self.config.capacity:
            lru_node = self._remove_lru_node()
            evicted_key = lru_node.key
            evicted_value = lru_node.entry.value
            del self.cache[evicted_key]
            self.size -= 1

        # now add new node
        new_node = Node(key, entry)
        self.cache[key] = new_node
        self._add_to_front(new_node)
        self.size += 1

        message = f"Added key {key}"
        if entry.expiry_time:
            ttl = entry.expiry_time - entry.created_at
            message += f" (TTL : {ttl:.1f}s)"
        
        if evicted_key is not None:
            message += f", evicted LRU key {evicted_key}"

        return CacheResponse(
            success=True,
            value=entry.value,
            evicted_key=evicted_key,
            evicted_value=evicted_value,
            message=message
        )

    def _delete_internal(self, key: K) -> bool:
        if key not in self.cache:
            return False

        node = self.cache[key]
        del self.cache[key]
        self._remove_node(node)
        self.size += 1
        return True

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.size = 0
            self.head.next = self.tail
            self.tail.prev = self.head


class LFUCache(Cache[K, V]):
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        
        self.cache = Dict[K, tuple[CacheEntry[V], int]] = {}
        self.freq_map = Dict[int, OrderedDict[K, CacheEntry[V]]] = defaultdict(OrderedDict)
        self.min_freq = 0
    
    def _update_freq(self, key: K) -> None:
        entry, freq = self.cache[key]
        
        # remove from old frequency
        del self.freq_map[freq][key]
        if not self.freq_map[freq] and freq == self.min_freq:
            self.min_freq += 1

        # add to new freq    
        new_freq = freq + 1
        self.freq_map[new_freq][key] = entry
        self.cache[key] = (entry, new_freq)
    
    def _get_internal(self, key: K) -> Optional[CacheEntry[V]]:
        if key not in self.cache:
            return None

        self._update_freq(key)
        entry, _ = self.cache[key]
        return entry

    def _put_internal(self, key: K, entry: CacheEntry[V]) -> Optional[CacheResponse[K, V]]:
        if key in self.cache:
            _, freq = self.cache[key]
            self.cache[key] = (entry, freq)
            self._update_freq(key)
            
            return CacheResponse(
                success=True,
                value=entry.value,
                message=f"Updated exisiting key {key} with value {entry.value}"
            )
        
        # evict if full
        evicted_key = None
        evicted_value = None
        if self.size >= self.config.capacity:
            evicted_key, evicted_entry = self.freq_map[self.min_freq].popitem(last=False)
            evicted_value = evicted_entry.value
            del self.cache[evicted_key]
            self.size -= 1
        
        self.cache[key] = (entry, 1)
        self.freq_map[1][key] = entry
        self.min_freq = 1
        self.size += 1

        message = f"Added key: {key}"
        if entry.expiry_time:
            ttl = entry.expiry_time - entry.created_at
            message += f" (TTL : {ttl:.1f}s)"
        
        if evicted_key is not None:
            message += f", evicted LFU key = {key}"
        
        return CacheResponse(
            success=True,
            value=entry.value,
            evicted_key=evicted_key,
            evicted_value=evicted_value,
            message=message
        )

    def _delete_internal(self, key: K) -> bool:
        if key not in self.cache:
            return False
        
        _, freq = self.cache[key]
        del self.cache[key]
        del self.freq_map[freq][key]
        self.size -= 1
        return True

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.freq_map.clear()
            self.min_freq = 0
            self.size = 0
    

# factory
class CacheFactory:
    
    @staticmethod
    def create(
        policy: EvicitionPolicy,
        capacity: int,
        default_ttl: Optional[float] = None 
    ) -> Cache:
        
        config = CacheConfig(capacity, default_ttl)

        if policy == EvicitionPolicy.LFU:
            return LFUCache(config)
        elif policy == EvicitionPolicy.LRU:
            return LRUCache(config)
        else:
            raise ValueError(f"Unknown evicition policy : {policy}")


# Driver code

def demo_ttl():
    cache = CacheFactory.create(EvicitionPolicy.LRU, capacity=5, default_ttl=2.0)
    print("LRU Cache with default TTL: 2 seconds created")

    print("Adding items")
    cache.put("short", "expires in 2 seconds")
    cache.put("long", "Expires in 10 seconds", ttl=10.0)
    cache.put("forever", "never expires", ttl=None)

    print("Added three items")

    print("Check immediately, all items should be in cache")
    print(f"short: {cache.get("short")}")
    print(f"longer: {cache.get("longer")}")
    print(f"forever: {cache.get("forever")}")


    print("check after 3 seconds, only long and forever should exist in cache")
    time.sleep(3)

    print(f"short: {cache.get("short")}")
    print(f"longer: {cache.get("longer")}")
    print(f"forever: {cache.get("forever")}")


demo_ttl()          




    

    


