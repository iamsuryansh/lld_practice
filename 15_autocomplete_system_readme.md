# Autocomplete System - Low Level Design

## Problem Statement

Design an **autocomplete (typeahead) system** like Google Search, Amazon search bar, or IDE code completion. The system must:

1. **Low latency**: Return suggestions in <100ms (ideally <10ms)
2. **Relevance**: Rank suggestions by popularity/frequency
3. **Fuzzy matching**: Handle typos and spelling mistakes
4. **Scalability**: Handle billions of queries per day
5. **Real-time updates**: Incorporate trending searches quickly
6. **Personalization**: User-specific suggestions (optional)

### Real-World Context
Used by: Google Search, Amazon, YouTube, VS Code, Twitter, LinkedIn

### Key Requirements
- **Performance**: Sub-100ms latency at p99
- **Accuracy**: Top suggestions match user intent 90%+ of time
- **Scale**: 100k+ queries per second
- **Storage**: Efficient trie structure (<10GB for 10M words)
- **Updates**: Real-time frequency updates

---

## Implementation Phases

### Phase 1: Trie Data Structure (15-20 minutes)

**Core concept**: Prefix tree for O(M) search where M = query length

```python
@dataclass
class TrieNode:
    """
    Node in trie with frequency tracking
    
    Key insight: Each node represents a character, path from root to 
    node represents a prefix
    """
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    is_end_of_word: bool = False
    word: Optional[str] = None
    frequency: int = 0  # How many times this word has been searched
    
class Trie:
    """
    Trie for autocomplete with frequency ranking
    """
    
    def __init__(self):
        self.root = TrieNode()
        self._word_count = 0
    
    def insert(self, word: str, frequency: int = 1) -> None:
        """
        Insert word with frequency
        
        Time: O(M) where M = word length
        Space: O(M) for new word
        """
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
        """Check if word exists - O(M)"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix - O(M)"""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find node for given prefix"""
        node = self.root
        
        for char in prefix.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node
    
    def get_suggestions(self, prefix: str, max_results: int = 10) -> List[Tuple[str, int]]:
        """
        Get top-K suggestions for prefix ranked by frequency
        
        Time: O(M + N) where M = prefix length, N = matching words
        Space: O(N) for collecting suggestions
        """
        node = self._find_node(prefix)
        
        if node is None:
            return []
        
        # Collect all words with this prefix using DFS
        suggestions = []
        self._collect_words(node, suggestions)
        
        # Return top-K by frequency using heap
        # O(N log K) where N = all suggestions
        top_k = heapq.nlargest(max_results, suggestions, key=lambda x: x[1])
        return top_k
    
    def _collect_words(self, node: TrieNode, suggestions: List[Tuple[str, int]]) -> None:
        """DFS to collect all words from this node"""
        if node.is_end_of_word:
            suggestions.append((node.word, node.frequency))
        
        for child in node.children.values():
            self._collect_words(child, suggestions)
```

**Interview questions to expect**:

- Q: "Why use Trie instead of HashMap?"
- A: "HashMap requires exact key match. Trie supports prefix queries efficiently. For 'pyth', HashMap needs to check all keys, Trie walks down 4 nodes."

- Q: "Space complexity?"
- A: "Worst case: O(ALPHABET_SIZE × M × N) where N = words, M = avg length. With 26 letters, 1M words, ~100 bytes each → ~2.6GB. Optimized with compressed tries (patricia/radix trees) → ~500MB."

---

### Phase 2: LRU Cache for Performance (15-20 minutes)

**Core concept**: Cache popular prefixes to avoid trie traversal

```python
@dataclass
class CacheEntry:
    """Cached suggestions for prefix"""
    prefix: str
    suggestions: List[Tuple[str, int]]
    timestamp: float = field(default_factory=time.time)

class LRUCache:
    """
    LRU cache for popular prefixes
    
    Key insight: 80% of queries are for 20% of prefixes (power law)
    Caching gives 10-100x latency reduction
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU order
        self._hits = 0
        self._misses = 0
    
    def get(self, prefix: str) -> Optional[List[Tuple[str, int]]]:
        """
        Get cached suggestions
        
        Time: O(1) average for lookup, O(N) worst-case for LRU update
        Better: Use OrderedDict for O(1) reordering
        """
        if prefix in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(prefix)
            self._access_order.append(prefix)
            self._hits += 1
            return self._cache[prefix].suggestions
        
        self._misses += 1
        return None
    
    def put(self, prefix: str, suggestions: List[Tuple[str, int]]) -> None:
        """
        Cache suggestions for prefix
        
        Time: O(1) average
        """
        if prefix in self._cache:
            self._access_order.remove(prefix)
        elif len(self._cache) >= self.capacity:
            # Evict LRU
            lru_prefix = self._access_order.pop(0)
            del self._cache[lru_prefix]
        
        self._cache[prefix] = CacheEntry(prefix, suggestions)
        self._access_order.append(prefix)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }
```

**Interview insight**:
- **Without cache**: Every query hits trie → 5-10ms latency
- **With cache**: 80% of queries hit cache → <1ms latency
- **Real-world**: Google Search caches top 10M queries → 95%+ hit rate

---

### Phase 3: Fuzzy Matching with Levenshtein Distance (20-25 minutes)

**Core concept**: Handle typos using edit distance

```python
class FuzzyMatcher:
    """
    Fuzzy matching using Levenshtein (edit) distance
    
    Key insight: Most typos are 1-2 character mistakes
    """
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute minimum edit operations to transform s1 → s2
        
        Operations: insert, delete, substitute (each cost = 1)
        
        Time: O(M × N) where M = len(s1), N = len(s2)
        Space: O(M × N) for DP table
        
        Interview focus: Explain DP recurrence relation
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all chars from s1
        for j in range(n + 1):
            dp[0][j] = j  # Insert all chars to reach s2
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    # Characters match, no operation needed
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Take minimum of: delete, insert, substitute
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Delete from s1
                        dp[i][j - 1],      # Insert to s1
                        dp[i - 1][j - 1]   # Substitute
                    )
        
        return dp[m][n]
    
    @staticmethod
    def get_fuzzy_matches(
        query: str,
        candidates: List[str],
        max_distance: int = 2,
        max_results: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Get fuzzy matches within edit distance threshold
        
        Time: O(N × M × K) where N = candidates, M = avg length, K = max_distance
        """
        matches = []
        
        for candidate in candidates:
            distance = FuzzyMatcher.levenshtein_distance(query, candidate)
            if distance <= max_distance:
                matches.append((candidate, distance))
        
        # Sort by distance (ascending)
        matches.sort(key=lambda x: x[1])
        
        return matches[:max_results]
```

**Interview deep-dive**:

**Q: "How to optimize fuzzy matching for millions of candidates?"**

A: "Several approaches:

1. **BK-Trees** (Burkhard-Keller):
   - Index words by edit distance
   - Query time: O(log N) instead of O(N)
   - Trade-off: Complex implementation

2. **N-gram index**:
   - Break words into character sequences (e.g., 'python' → 'py', 'yt', 'th', 'ho', 'on')
   - Words with similar n-grams likely have low edit distance
   - Filter candidates before computing exact distance

3. **Phonetic algorithms**:
   - Soundex, Metaphone for pronunciation-based matching
   - 'knight' and 'night' have same phonetic code

4. **Limit search space**:
   - Only check words starting with same first 2 letters
   - Reduces candidates from 1M to ~1000"

---

### Phase 4: Complete Autocomplete System (25-30 minutes)

**Core concept**: Integrate trie, cache, and fuzzy matching

```python
class AutocompleteSystem:
    """
    Production autocomplete system
    
    Features:
    - Trie for prefix search
    - LRU cache for performance
    - Fuzzy matching for typos
    - Frequency-based ranking
    """
    
    def __init__(self, cache_capacity: int = 1000):
        self.trie = Trie()
        self.cache = LRUCache(cache_capacity)
        self._query_log: List[Tuple[str, float]] = []
    
    def add_word(self, word: str, frequency: int = 1) -> None:
        """
        Add word to dictionary
        
        Invalidate cache for all prefixes of this word
        """
        self.trie.insert(word, frequency)
        
        # Invalidate cache
        for i in range(1, len(word) + 1):
            self.cache.invalidate(word[:i].lower())
    
    def get_suggestions(
        self,
        prefix: str,
        max_results: int = 10,
        use_cache: bool = True
    ) -> List[Tuple[str, int]]:
        """
        Get autocomplete suggestions
        
        Flow:
        1. Check cache (O(1))
        2. If miss, query trie (O(M + N))
        3. Cache result
        4. Return suggestions
        """
        start_time = time.time()
        
        # Log query (for analytics)
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
        """
        Get suggestions with fuzzy matching
        
        Strategy:
        1. Try exact prefix match first
        2. If insufficient results, add fuzzy matches
        3. Combine and deduplicate
        """
        # Get exact matches
        exact_matches = self.get_suggestions(query, max_results)
        
        if len(exact_matches) >= max_results:
            return exact_matches
        
        # Need more results, try fuzzy matching
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
    
    def update_frequency(self, word: str, increment: int = 1) -> None:
        """
        Update word frequency (user selected this suggestion)
        
        Use case: User types 'pyth', selects 'python' → boost 'python' frequency
        """
        self.trie.insert(word, increment)
        
        # Invalidate cache for all prefixes
        for i in range(1, len(word) + 1):
            self.cache.invalidate(word[:i].lower())
```

**Interview Q&A**:

**Q: "How to handle real-time trending searches?"**

A: "Time-decay frequency:
```python
def get_time_decayed_frequency(base_freq: int, last_search_time: float) -> float:
    # Decay by 50% every 24 hours
    hours_since = (time.time() - last_search_time) / 3600
    decay_factor = 0.5 ** (hours_since / 24)
    return base_freq * decay_factor
```

This ensures recent searches rank higher than old popular searches."

---

### Phase 5: Optimization for Production (20-25 minutes)

**Optimizations**:

**1. Precompute top-K at each node**:
```python
@dataclass
class OptimizedTrieNode:
    children: Dict[str, 'OptimizedTrieNode']
    top_k_words: List[Tuple[str, int]] = field(default_factory=list)  # Cached
    
    def get_suggestions(self, max_results: int) -> List[Tuple[str, int]]:
        """O(1) instead of O(N) - return precomputed top-K"""
        return self.top_k_words[:max_results]
```

**Benefits**: Query time O(M) instead of O(M + N)

**Trade-off**: O(K × N) space, O(K × N log K) update time

**2. Prefix → Document ID mapping**:
```python
# Instead of storing full words in trie
# Store word IDs and maintain separate word database
class CompressedTrie:
    def __init__(self):
        self.root = TrieNode()
        self.word_db = {}  # word_id → (word, frequency)
    
    def insert(self, word: str, frequency: int):
        word_id = self.get_word_id(word)
        # Store word_id in trie nodes instead of full word
        # Saves ~70% memory for long words
```

**3. Distributed architecture**:
```
Client → Load Balancer → [Autocomplete Servers] → Shared Trie Cache (Redis)
                                                 ↓
                                         Word Frequency DB (MySQL)
```

**Benefits**:
- Horizontal scaling (add more servers)
- Shared cache across servers (higher hit rate)
- Persistent storage for frequency updates

---

## Critical Knowledge Points

### 1. Trie vs HashMap

| Operation | Trie | HashMap |
|-----------|------|---------|
| **Exact search** | O(M) | O(M) - hash computation |
| **Prefix search** | O(M + N) | O(K × M) where K = total keys |
| **Space** | O(ALPHABET × M × N) | O(M × N) |
| **Autocomplete** | Native support | Requires filtering all keys |

**Verdict**: Trie wins for prefix queries despite higher space usage.

### 2. Cache Effectiveness

**Without personalization**:
- Top 100 queries → 20% of traffic
- Top 10,000 queries → 80% of traffic
- Cache 10,000 prefixes → ~95% hit rate

**With personalization**:
- Cache must include user_id in key
- Hit rate drops to ~60-70%
- Need larger cache (100K-1M entries)

### 3. Latency Breakdown

**Query: "python" (without cache)**:
```
1. Parse request: 0.1ms
2. Find prefix node: 0.5ms (6 char traversal)
3. Collect suggestions: 2ms (DFS 1000 words)
4. Sort by frequency: 0.3ms (heap of 1000 → top 10)
5. Format response: 0.1ms
---
Total: ~3ms
```

**Query: "python" (with cache)**:
```
1. Parse request: 0.1ms
2. Cache lookup: 0.2ms
3. Format response: 0.1ms
---
Total: ~0.4ms (7.5x faster!)
```

---

## Interview Q&A

### Q1: "How does Google handle billions of queries per day?"

**Answer**: Multi-tier architecture:

**Tier 1: Edge servers (CDN)**
- Cache top 10M queries per region
- 95% hit rate → 95% of queries never reach backend
- Latency: <10ms

**Tier 2: Application servers**
- Stateless autocomplete servers
- Query distributed trie (Redis cluster)
- Latency: ~50ms for cache miss

**Tier 3: Data pipeline**
- Batch process query logs (Hadoop/Spark)
- Update frequencies every hour
- Push updates to trie cache

**Tier 4: ML models**
- Personalization: User history, location, time
- Spell correction: Deep learning models
- Ranking: Click-through rate prediction

---

### Q2: "How to handle personalization?"

**Answer**: Combine global and personal suggestions:

```python
def get_personalized_suggestions(user_id: str, prefix: str) -> List[str]:
    # Get global popular suggestions
    global_suggestions = trie.get_suggestions(prefix, max_results=20)
    
    # Get user's search history
    user_history = get_user_history(user_id, prefix)
    
    # Combine with weights
    scores = {}
    for word, freq in global_suggestions:
        scores[word] = freq * 0.7  # 70% weight to global
    
    for word, freq in user_history:
        scores[word] = scores.get(word, 0) + freq * 0.3  # 30% weight to personal
    
    # Sort and return top 10
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
```

**Trade-offs**:
- **Privacy**: User history stored, needs encryption
- **Cache**: Hit rate drops (user_id in cache key)
- **Complexity**: More infrastructure (user history DB)

---

### Q3: "How to handle multi-language support?"

**Answer**: Separate tries per language:

```python
class MultiLanguageAutocomplete:
    def __init__(self):
        self.tries = {
            'en': Trie(),  # English
            'es': Trie(),  # Spanish
            'zh': Trie(),  # Chinese
            # ...
        }
    
    def get_suggestions(self, prefix: str, language: str) -> List[str]:
        if language not in self.tries:
            language = 'en'  # Default to English
        
        return self.tries[language].get_suggestions(prefix)
```

**Challenges**:
- **Chinese/Japanese**: Need to support pinyin (romanization)
- **Accents**: 'café' vs 'cafe' → normalize or support both
- **Memory**: 50+ languages → 50x memory usage

---

### Q4: "How to prevent offensive suggestions?"

**Answer**: Blacklist + ML filtering:

```python
class SafeAutocomplete:
    def __init__(self):
        self.autocomplete = AutocompleteSystem()
        self.blacklist = set(load_offensive_words())
        self.ml_filter = load_ml_model()  # Toxicity classifier
    
    def get_suggestions(self, prefix: str) -> List[str]:
        suggestions = self.autocomplete.get_suggestions(prefix)
        
        # Filter out blacklisted words
        filtered = [(w, f) for w, f in suggestions if w not in self.blacklist]
        
        # Apply ML filter
        safe = []
        for word, freq in filtered:
            toxicity_score = self.ml_filter.predict(word)
            if toxicity_score < 0.5:  # Threshold
                safe.append((word, freq))
        
        return safe
```

---

### Q5: "Compare autocomplete algorithms"

**Answer**:

| Approach | Latency | Memory | Accuracy | Use Case |
|----------|---------|--------|----------|----------|
| **Trie** | ~5ms | High (GB) | High | Google Search |
| **N-gram** | ~10ms | Medium | Medium | SMS keyboards |
| **ML (LSTM/GPT)** | ~100ms | Very high | Very high | Code completion (GitHub Copilot) |
| **Hybrid** | ~10ms | High | Very high | Production (Trie + ML rerank) |

**Trend**: Modern systems use Trie for retrieval + Transformer for ranking.

---

### Q6: "How to test autocomplete quality?"

**Answer**: Multiple metrics:

**1. Accuracy metrics**:
```python
def mean_reciprocal_rank(queries: List[Tuple[str, str]]) -> float:
    """
    MRR = average of 1/rank where rank = position of correct suggestion
    
    Example:
    Query 'pyth', correct = 'python'
    Suggestions: ['python', 'pythagoras', 'pythonic']
    Rank = 1 → RR = 1/1 = 1.0
    
    High MRR = correct suggestion appears early in list
    """
    reciprocal_ranks = []
    for prefix, expected_word in queries:
        suggestions = autocomplete.get_suggestions(prefix)
        for rank, (word, _) in enumerate(suggestions, start=1):
            if word == expected_word:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)  # Not found
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

**2. Performance metrics**:
- Latency p50, p99, p999
- Cache hit rate
- QPS (queries per second) capacity

**3. Business metrics**:
- Click-through rate (% of suggestions clicked)
- Typing reduction (avg chars saved per query)
- User satisfaction surveys

---

### Q7: "How to handle rapidly trending searches (e.g., breaking news)?"

**Answer**: Real-time pipeline:

```
Query logs → Stream processor (Kafka) → Trend detector → Hot cache
                                               ↓
                                        Update trie frequencies
```

**Trend detection**:
```python
def detect_trends(query_stream: Iterator[str], window_minutes: int = 10) -> Set[str]:
    """
    Detect queries with sudden frequency spike
    """
    recent_counts = defaultdict(int)
    historical_counts = defaultdict(int)
    
    for query in query_stream:
        recent_counts[query] += 1
        
        # Check for spike
        if recent_counts[query] > historical_counts[query] * 10:  # 10x increase
            mark_as_trending(query)
        
        # Update historical
        if time_to_rotate_window():
            historical_counts = recent_counts
            recent_counts = defaultdict(int)
```

**Hot cache**:
- Separate cache for trending searches
- Higher TTL (Time-To-Live)
- Replicated across all edge servers

---

## Testing Strategy

### Unit Tests
```python
def test_trie_insert_and_search():
    trie = Trie()
    trie.insert("python", frequency=100)
    assert trie.search("python")
    assert not trie.search("java")

def test_trie_prefix_suggestions():
    trie = Trie()
    trie.insert("python", 1000)
    trie.insert("pytorch", 800)
    trie.insert("pandas", 600)
    
    suggestions = trie.get_suggestions("py", max_results=2)
    assert suggestions[0][0] == "python"  # Highest frequency
    assert suggestions[1][0] == "pytorch"

def test_levenshtein_distance():
    assert FuzzyMatcher.levenshtein_distance("kitten", "sitting") == 3
    assert FuzzyMatcher.levenshtein_distance("hello", "hello") == 0

def test_lru_cache_eviction():
    cache = LRUCache(capacity=2)
    cache.put("a", [("apple", 10)])
    cache.put("b", [("banana", 5)])
    cache.put("c", [("cherry", 3)])  # Evicts "a"
    
    assert cache.get("a") is None
    assert cache.get("b") is not None
```

### Integration Tests
```python
def test_autocomplete_with_cache():
    system = AutocompleteSystem(cache_capacity=100)
    system.add_word("python", frequency=1000)
    
    # First query (cache miss)
    suggestions1 = system.get_suggestions("py")
    
    # Second query (cache hit)
    suggestions2 = system.get_suggestions("py")
    
    assert suggestions1 == suggestions2
    assert system.cache.get_stats()["hit_rate"] == 0.5  # 1 hit, 1 miss

def test_fuzzy_matching():
    system = AutocompleteSystem()
    system.add_word("restaurant", frequency=100)
    
    # Typo: "restaraunt"
    suggestions = system.get_fuzzy_suggestions("restaraunt", max_distance=2)
    
    assert ("restaurant", 100) in suggestions
```

### Performance Tests
```python
def test_latency_under_load():
    system = AutocompleteSystem()
    
    # Load 1M words
    for i in range(1_000_000):
        system.add_word(f"word{i}", frequency=random.randint(1, 1000))
    
    # Measure query latency
    queries = ["word" + str(random.randint(0, 1000)) for _ in range(10000)]
    
    start = time.time()
    for query in queries:
        system.get_suggestions(query[:4], max_results=10)
    elapsed = time.time() - start
    
    avg_latency = elapsed / len(queries) * 1000  # Convert to ms
    assert avg_latency < 10  # <10ms per query
```

---

## Production Considerations

### 1. Data Ingestion
```python
# Batch process query logs to update frequencies
def process_query_logs(log_file: str):
    query_counts = defaultdict(int)
    
    with open(log_file) as f:
        for line in f:
            query = parse_query(line)
            query_counts[query] += 1
    
    # Update trie
    for query, count in query_counts.items():
        autocomplete.update_frequency(query, increment=count)
```

### 2. Monitoring Metrics
```python
metrics = {
    "qps": 50000,  # Queries per second
    "latency_p99": 15,  # milliseconds
    "cache_hit_rate": 0.92,  # 92%
    "trie_size_mb": 2048,  # 2GB
    "top_query_miss_rate": 0.01  # 1% of top queries not in trie
}
```

### 3. Scalability
- **Vertical**: 64GB RAM servers for in-memory trie
- **Horizontal**: 10-100 stateless servers + shared Redis cluster
- **Geo-distributed**: Regional tries for low latency

### 4. Data Consistency
- **Writes**: Async updates to trie (eventual consistency OK)
- **Reads**: Always consistent (trie is immutable during queries)
- **Deployment**: Blue-green deployment for trie updates

---

## Summary

### Do's ✅
- Use Trie for prefix-based search
- Cache popular prefixes with LRU
- Implement fuzzy matching for typos
- Rank by frequency (time-decayed)
- Precompute top-K at trie nodes for O(1) queries
- Monitor cache hit rate and optimize

### Don'ts ❌
- Don't use HashMap for prefix search (O(N) filtering)
- Don't skip caching (10x latency reduction)
- Don't ignore typos (10-20% of queries have errors)
- Don't use plain timestamps for ranking (need time decay)
- Don't store full words at every trie node (memory bloat)
- Don't forget to blacklist offensive terms

### Key Takeaways
1. **Trie**: O(M + N) prefix search vs O(K × M) for HashMap
2. **Cache**: 80/20 rule → 20% of prefixes = 80% of traffic
3. **Fuzzy**: Levenshtein distance O(M × N) with DP
4. **Ranking**: Frequency + time decay + personalization
5. **Scale**: Distributed with Redis cluster + edge caching

This system demonstrates production-grade autocomplete used by Google, Amazon, and modern IDEs.
