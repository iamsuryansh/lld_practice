"""
Autocomplete System - Low Level Design
======================================

Interview Focus:
- Trie data structure for prefix search
- Frequency-based ranking
- LRU caching for performance
- Fuzzy matching (edit distance)
- Real-time suggestions (<100ms latency)
- Scalability (billions of queries)

This implementation demonstrates:
1. Trie with frequency tracking
2. Top-K suggestions using heap
3. LRU cache for popular prefixes
4. Fuzzy matching with Levenshtein distance
5. Thread-safe operations
6. Memory-efficient storage

Production Considerations:
- Distributed: Sharding by prefix
- Storage: Persistent trie (serialization)
- Updates: Real-time frequency updates
- Personalization: User-specific suggestions
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import threading


# ============================================================================
# SECTION 1: Trie Data Structure
# ============================================================================

@dataclass
class TrieNode:
    """Node in trie with frequency tracking"""
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    is_end_of_word: bool = False
    word: Optional[str] = None
    frequency: int = 0  # How many times this word has been searched
    
    def __repr__(self):
        return f"TrieNode(word={self.word}, freq={self.frequency}, children={len(self.children)})"


class Trie:
    """
    Trie for autocomplete with frequency ranking
    
    Interview Focus:
    - Insert: O(M) where M = word length
    - Search: O(M)
    - Prefix search: O(M + N) where N = results
    - Space: O(ALPHABET_SIZE * M * N) where N = words
    """
    
    def __init__(self):
        self.root = TrieNode()
        self._lock = threading.RLock()
        self._word_count = 0
    
    def insert(self, word: str, frequency: int = 1) -> None:
        """Insert word with frequency"""
        with self._lock:
            node = self.root
            
            for char in word.lower():
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            
            node.is_end_of_word = True
            node.word = word
            node.frequency += frequency
            self._word_count += 1
    
    def search(self, word: str) -> bool:
        """Check if word exists"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix"""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find node for given prefix"""
        with self._lock:
            node = self.root
            
            for char in prefix.lower():
                if char not in node.children:
                    return None
                node = node.children[char]
            
            return node
    
    def get_suggestions(self, prefix: str, max_results: int = 10) -> List[Tuple[str, int]]:
        """
        Get top-K suggestions for prefix ranked by frequency
        
        Interview Focus: Explain DFS + heap for top-K
        """
        with self._lock:
            node = self._find_node(prefix)
            
            if node is None:
                return []
            
            # Collect all words with this prefix
            suggestions = []
            self._collect_words(node, suggestions)
            
            # Return top-K by frequency
            # Use heap: O(N log K) where N = all suggestions
            top_k = heapq.nlargest(max_results, suggestions, key=lambda x: x[1])
            return top_k
    
    def _collect_words(self, node: TrieNode, suggestions: List[Tuple[str, int]]) -> None:
        """DFS to collect all words from this node"""
        if node.is_end_of_word:
            suggestions.append((node.word, node.frequency))
        
        for child in node.children.values():
            self._collect_words(child, suggestions)
    
    def delete(self, word: str) -> bool:
        """Delete word from trie"""
        with self._lock:
            return self._delete_recursive(self.root, word.lower(), 0)
    
    def _delete_recursive(self, node: TrieNode, word: str, depth: int) -> bool:
        """Recursively delete word"""
        if depth == len(word):
            if not node.is_end_of_word:
                return False
            
            node.is_end_of_word = False
            node.word = None
            node.frequency = 0
            
            # Delete node if no children
            return len(node.children) == 0
        
        char = word[depth]
        if char not in node.children:
            return False
        
        should_delete_child = self._delete_recursive(node.children[char], word, depth + 1)
        
        if should_delete_child:
            del node.children[char]
            return not node.is_end_of_word and len(node.children) == 0
        
        return False


# ============================================================================
# SECTION 2: LRU Cache for Performance
# ============================================================================

@dataclass
class CacheEntry:
    """Cached suggestions for prefix"""
    prefix: str
    suggestions: List[Tuple[str, int]]
    timestamp: float = field(default_factory=time.time)


class LRUCache:
    """LRU cache for popular prefixes"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, prefix: str) -> Optional[List[Tuple[str, int]]]:
        """Get cached suggestions"""
        with self._lock:
            if prefix in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(prefix)
                self._access_order.append(prefix)
                self._hits += 1
                return self._cache[prefix].suggestions
            
            self._misses += 1
            return None
    
    def put(self, prefix: str, suggestions: List[Tuple[str, int]]) -> None:
        """Cache suggestions for prefix"""
        with self._lock:
            if prefix in self._cache:
                self._access_order.remove(prefix)
            elif len(self._cache) >= self.capacity:
                # Evict LRU
                lru_prefix = self._access_order.pop(0)
                del self._cache[lru_prefix]
            
            self._cache[prefix] = CacheEntry(prefix, suggestions)
            self._access_order.append(prefix)
    
    def invalidate(self, prefix: str) -> None:
        """Invalidate cache entry"""
        with self._lock:
            if prefix in self._cache:
                del self._cache[prefix]
                self._access_order.remove(prefix)
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "capacity": self.capacity
            }


# ============================================================================
# SECTION 3: Fuzzy Matching
# ============================================================================

class FuzzyMatcher:
    """Fuzzy matching using Levenshtein distance"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute edit distance between two strings
        
        Interview Focus: DP solution O(M*N)
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )
        
        return dp[m][n]
    
    @staticmethod
    def get_fuzzy_matches(
        query: str,
        candidates: List[str],
        max_distance: int = 2,
        max_results: int = 10
    ) -> List[Tuple[str, int]]:
        """Get fuzzy matches within edit distance threshold"""
        matches = []
        
        for candidate in candidates:
            distance = FuzzyMatcher.levenshtein_distance(query, candidate)
            if distance <= max_distance:
                matches.append((candidate, distance))
        
        # Sort by distance (ascending)
        matches.sort(key=lambda x: x[1])
        
        return matches[:max_results]


# ============================================================================
# SECTION 4: Autocomplete System
# ============================================================================

class AutocompleteSystem:
    """
    Autocomplete system with caching and fuzzy matching
    
    Interview Focus:
    - Real-time suggestions (<100ms)
    - Ranked by frequency
    - Cached for popular prefixes
    - Fuzzy matching for typos
    """
    
    def __init__(self, cache_capacity: int = 1000):
        self.trie = Trie()
        self.cache = LRUCache(cache_capacity)
        self._query_log: List[Tuple[str, float]] = []
        self._lock = threading.RLock()
    
    def add_word(self, word: str, frequency: int = 1) -> None:
        """Add word to dictionary"""
        with self._lock:
            self.trie.insert(word, frequency)
            
            # Invalidate cache for prefixes of this word
            for i in range(1, len(word) + 1):
                self.cache.invalidate(word[:i].lower())
    
    def add_words(self, words: List[str]) -> None:
        """Bulk add words"""
        for word in words:
            self.add_word(word)
    
    def get_suggestions(
        self,
        prefix: str,
        max_results: int = 10,
        use_cache: bool = True
    ) -> List[Tuple[str, int]]:
        """
        Get autocomplete suggestions
        
        Interview Focus: Explain caching strategy
        1. Check cache first (O(1))
        2. If miss, query trie (O(M + N))
        3. Cache result for next time
        """
        with self._lock:
            start_time = time.time()
            
            # Log query
            self._query_log.append((prefix, start_time))
            
            # Check cache
            if use_cache:
                cached = self.cache.get(prefix)
                if cached is not None:
                    return cached
            
            # Query trie
            suggestions = self.trie.get_suggestions(prefix, max_results)
            
            # Cache result
            if use_cache:
                self.cache.put(prefix, suggestions)
            
            return suggestions
    
    def get_fuzzy_suggestions(
        self,
        query: str,
        max_distance: int = 2,
        max_results: int = 10
    ) -> List[Tuple[str, int]]:
        """Get suggestions with fuzzy matching"""
        with self._lock:
            # Get exact prefix matches first
            exact_matches = self.get_suggestions(query, max_results)
            
            if len(exact_matches) >= max_results:
                return exact_matches
            
            # Add fuzzy matches
            all_words = []
            self._collect_all_words(self.trie.root, all_words)
            
            fuzzy_matches = FuzzyMatcher.get_fuzzy_matches(
                query,
                [w for w, _ in all_words],
                max_distance,
                max_results
            )
            
            # Combine and deduplicate
            combined = {word: freq for word, freq in exact_matches}
            for word, distance in fuzzy_matches:
                if word not in combined:
                    # Find frequency from trie
                    node = self.trie._find_node(word)
                    if node and node.is_end_of_word:
                        combined[word] = node.frequency
            
            # Sort by frequency
            result = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            return result[:max_results]
    
    def _collect_all_words(self, node: TrieNode, words: List[Tuple[str, int]]) -> None:
        """Collect all words from trie"""
        if node.is_end_of_word:
            words.append((node.word, node.frequency))
        
        for child in node.children.values():
            self._collect_all_words(child, words)
    
    def update_frequency(self, word: str, increment: int = 1) -> None:
        """Update word frequency (user selected this suggestion)"""
        with self._lock:
            self.trie.insert(word, increment)
            
            # Invalidate cache
            for i in range(1, len(word) + 1):
                self.cache.invalidate(word[:i].lower())
    
    def delete_word(self, word: str) -> bool:
        """Delete word from dictionary"""
        with self._lock:
            success = self.trie.delete(word)
            
            if success:
                # Invalidate cache
                for i in range(1, len(word) + 1):
                    self.cache.invalidate(word[:i].lower())
            
            return success
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            cache_stats = self.cache.get_stats()
            
            return {
                "total_words": self.trie._word_count,
                "total_queries": len(self._query_log),
                "cache_stats": cache_stats,
                "recent_queries": [q for q, _ in self._query_log[-10:]]
            }


# ============================================================================
# SECTION 5: Demo Functions
# ============================================================================

def demo_basic_autocomplete():
    """Demo 1: Basic autocomplete"""
    print("=" * 70)
    print("  Basic Autocomplete System")
    print("=" * 70)
    
    system = AutocompleteSystem()
    
    # Add dictionary
    print("\n🔹 Building dictionary:")
    words = [
        "apple", "application", "apply", "apricot",
        "banana", "band", "bandana", "banner",
        "cat", "catch", "category", "cathedral"
    ]
    
    for word in words:
        system.add_word(word, frequency=10)  # Base frequency
    
    print(f"  Added {len(words)} words")
    
    # Get suggestions
    print("\n🔹 Autocomplete suggestions:")
    prefixes = ["app", "ban", "cat"]
    
    for prefix in prefixes:
        suggestions = system.get_suggestions(prefix, max_results=5)
        print(f"\n  '{prefix}' →")
        for word, freq in suggestions:
            print(f"    • {word} (frequency: {freq})")


def demo_frequency_ranking():
    """Demo 2: Frequency-based ranking"""
    print("\n" + "=" * 70)
    print("  Frequency-Based Ranking")
    print("=" * 70)
    
    system = AutocompleteSystem()
    
    # Add words with different frequencies
    print("\n🔹 Adding words with frequencies:")
    words_with_freq = [
        ("python", 1000),
        ("pytorch", 800),
        ("pycharm", 600),
        ("pandas", 500),
        ("pytest", 300)
    ]
    
    for word, freq in words_with_freq:
        system.add_word(word, frequency=freq)
        print(f"  {word}: {freq} searches")
    
    # Get suggestions (should be ranked by frequency)
    print("\n🔹 Suggestions for 'py' (ranked by popularity):")
    suggestions = system.get_suggestions("py")
    
    for i, (word, freq) in enumerate(suggestions, 1):
        print(f"  {i}. {word} ({freq} searches)")
    
    # Simulate user selection (increase frequency)
    print("\n🔹 User selected 'pytest' → increasing frequency")
    system.update_frequency("pytest", increment=500)
    
    print("\n🔹 Updated suggestions for 'py':")
    suggestions = system.get_suggestions("py")
    
    for i, (word, freq) in enumerate(suggestions, 1):
        print(f"  {i}. {word} ({freq} searches)")


def demo_caching():
    """Demo 3: LRU caching"""
    print("\n" + "=" * 70)
    print("  LRU Caching for Performance")
    print("=" * 70)
    
    system = AutocompleteSystem(cache_capacity=100)
    
    # Add words
    words = ["javascript", "java", "javadoc", "jakarta"]
    for word in words:
        system.add_word(word)
    
    print("\n🔹 First query (cache miss):")
    start = time.time()
    suggestions = system.get_suggestions("jav")
    elapsed = (time.time() - start) * 1000
    print(f"  Time: {elapsed:.2f}ms")
    print(f"  Results: {[w for w, _ in suggestions]}")
    
    print("\n🔹 Second query (cache hit):")
    start = time.time()
    suggestions = system.get_suggestions("jav")
    elapsed = (time.time() - start) * 1000
    print(f"  Time: {elapsed:.2f}ms (cached)")
    print(f"  Results: {[w for w, _ in suggestions]}")
    
    # Show cache stats
    stats = system.get_statistics()
    print(f"\n🔹 Cache statistics:")
    print(f"  Hit rate: {stats['cache_stats']['hit_rate']:.1%}")
    print(f"  Hits: {stats['cache_stats']['hits']}")
    print(f"  Misses: {stats['cache_stats']['misses']}")


def demo_fuzzy_matching():
    """Demo 4: Fuzzy matching for typos"""
    print("\n" + "=" * 70)
    print("  Fuzzy Matching (Typo Correction)")
    print("=" * 70)
    
    system = AutocompleteSystem()
    
    # Add words
    words = ["restaurant", "reservation", "resources", "research"]
    for word in words:
        system.add_word(word, frequency=100)
    
    print("\n🔹 Dictionary:")
    for word in words:
        print(f"  • {word}")
    
    # Typo: "restaraunt" instead of "restaurant"
    print("\n🔹 Query with typo: 'restaraunt'")
    
    # Exact match (no results)
    exact = system.get_suggestions("restaraunt")
    print(f"  Exact match: {[w for w, _ in exact] if exact else 'No results'}")
    
    # Fuzzy match
    fuzzy = system.get_fuzzy_suggestions("restaraunt", max_distance=2)
    print(f"  Fuzzy match:")
    for word, freq in fuzzy:
        distance = FuzzyMatcher.levenshtein_distance("restaraunt", word)
        print(f"    • {word} (edit distance: {distance})")


def demo_real_world_scenario():
    """Demo 5: Real-world search engine"""
    print("\n" + "=" * 70)
    print("  Real-World Search Scenario")
    print("=" * 70)
    
    system = AutocompleteSystem()
    
    # Simulate search engine with popular queries
    print("\n🔹 Initializing with popular searches:")
    popular_searches = [
        ("weather", 10000),
        ("weather forecast", 8000),
        ("weather today", 7000),
        ("facebook", 15000),
        ("facebook login", 12000),
        ("youtube", 20000),
        ("youtube music", 9000),
        ("amazon", 18000),
        ("amazon prime", 11000)
    ]
    
    for query, freq in popular_searches:
        system.add_word(query, frequency=freq)
    
    print(f"  Added {len(popular_searches)} queries")
    
    # Simulate user typing
    print("\n🔹 User typing 'weat':")
    suggestions = system.get_suggestions("weat", max_results=3)
    for i, (query, freq) in enumerate(suggestions, 1):
        print(f"  {i}. {query} ({freq:,} searches)")
    
    print("\n🔹 User typing 'you':")
    suggestions = system.get_suggestions("you", max_results=3)
    for i, (query, freq) in enumerate(suggestions, 1):
        print(f"  {i}. {query} ({freq:,} searches)")
    
    # Show statistics
    stats = system.get_statistics()
    print(f"\n🔹 System statistics:")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.1%}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AUTOCOMPLETE SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Trie, Frequency ranking, LRU cache, Fuzzy matching")
    print("=" * 70)
    
    # Run all demos
    demo_basic_autocomplete()
    demo_frequency_ranking()
    demo_caching()
    demo_fuzzy_matching()
    demo_real_world_scenario()
    
    print("\n" + "=" * 70)
    print("  All demonstrations completed!")
    print("=" * 70)
