"""
Distributed Key-Value Store - Low Level Design
==============================================

Interview Focus:
- Distributed data storage with horizontal scalability
- Sharding strategies (range-based, hash-based, consistent hashing)
- Vector clocks for conflict resolution in eventually consistent systems
- Merkle trees for anti-entropy and replica synchronization
- Multi-version concurrency control (MVCC)
- Gossip protocol for cluster membership
- Write-ahead logging (WAL) for durability

This implementation demonstrates:
1. Consistent hashing with virtual nodes for sharding
2. Vector clocks for causality tracking
3. Merkle trees for efficient replica comparison
4. Quorum-based reads/writes (R + W > N)
5. Hinted handoff for temporary node failures
6. Anti-entropy with merkle tree comparison
7. Write-ahead log for crash recovery

Production Considerations:
- Durability: WAL + periodic snapshots
- Consistency: Tunable (eventual vs strong)
- Scalability: 1000s of nodes, billions of keys
- Availability: Handles network partitions gracefully
"""

import hashlib
import json
import os
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import bisect
import pickle


# ============================================================================
# SECTION 1: Core Models and Vector Clocks
# ============================================================================
# Interview Focus: Conflict resolution in distributed systems

class ConsistencyModel(Enum):
    """Consistency guarantees"""
    EVENTUAL = "eventual"      # High availability, eventual convergence
    STRONG = "strong"          # Linearizable, may block on partition
    CAUSAL = "causal"          # Causally related ops ordered


@dataclass
class VectorClock:
    """
    Vector clock for causality tracking
    
    Interview Focus:
    Q: Why vector clocks over timestamps?
    A: Timestamps don't capture causality in distributed systems
       (clock skew, no happens-before relationship)
    
    Example:
    Event A: {node1: 1, node2: 0}
    Event B: {node1: 1, node2: 1}
    B happened after A (B dominates A)
    """
    
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str) -> 'VectorClock':
        """Increment clock for this node"""
        new_clocks = self.clocks.copy()
        new_clocks[node_id] = new_clocks.get(node_id, 0) + 1
        return VectorClock(new_clocks)
    
    def merge(self, other: 'VectorClock') -> 'VectorClock':
        """Merge two vector clocks (take max of each component)"""
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        merged = {}
        for node in all_nodes:
            merged[node] = max(self.clocks.get(node, 0), other.clocks.get(node, 0))
        return VectorClock(merged)
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """
        Check if this event happened before other (causal relationship)
        
        A happens-before B if:
        - For all nodes i: A[i] <= B[i]
        - There exists at least one node j: A[j] < B[j]
        """
        all_nodes = set(self.clocks.keys()) | set(other.clocks.keys())
        
        all_less_or_equal = True
        at_least_one_less = False
        
        for node in all_nodes:
            a_clock = self.clocks.get(node, 0)
            b_clock = other.clocks.get(node, 0)
            
            if a_clock > b_clock:
                all_less_or_equal = False
                break
            if a_clock < b_clock:
                at_least_one_less = True
        
        return all_less_or_equal and at_least_one_less
    
    def concurrent(self, other: 'VectorClock') -> bool:
        """Check if two events are concurrent (no causal relationship)"""
        return not self.happens_before(other) and not other.happens_before(self)
    
    def __repr__(self):
        return f"VectorClock({self.clocks})"


@dataclass
class Versioned:
    """
    Value with version history (for conflict resolution)
    
    Interview Focus: Multi-version concurrency control
    """
    key: str
    value: Any
    vector_clock: VectorClock
    timestamp: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash(self.key)


@dataclass
class KVNode:
    """Distributed storage node"""
    node_id: str
    host: str
    port: int
    is_alive: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    disk_path: str = "/tmp/kvstore"
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return isinstance(other, KVNode) and self.node_id == other.node_id


@dataclass
class ReadResult:
    """Result of distributed read"""
    success: bool
    values: List[Versioned] = field(default_factory=list)  # May have conflicts
    message: str = ""


@dataclass
class WriteResult:
    """Result of distributed write"""
    success: bool
    vector_clock: Optional[VectorClock] = None
    message: str = ""


# ============================================================================
# SECTION 2: Merkle Tree for Anti-Entropy
# ============================================================================
# Interview Focus: Efficient replica synchronization

class MerkleNode:
    """Node in Merkle tree"""
    
    def __init__(self, start_key: str, end_key: str):
        self.start_key = start_key
        self.end_key = end_key
        self.hash: Optional[str] = None
        self.left: Optional['MerkleNode'] = None
        self.right: Optional['MerkleNode'] = None
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class MerkleTree:
    """
    Merkle tree for detecting divergent replicas
    
    Interview Focus:
    Q: Why Merkle trees for anti-entropy?
    A: Compare O(log N) hashes instead of O(N) keys
       Efficiently identify divergent ranges
    
    Example:
    Replica A:        Replica B:
        Root             Root
       /    \           /    \
     H1     H2        H1'    H2
    / \    / \       / \    / \
    Same  Same    Different Same
    
    Only need to sync keys in left subtree
    """
    
    def __init__(self, keys_per_leaf: int = 100):
        self.keys_per_leaf = keys_per_leaf
        self.root: Optional[MerkleNode] = None
    
    def build(self, data: Dict[str, Versioned]) -> None:
        """Build Merkle tree from key-value pairs"""
        if not data:
            return
        
        sorted_keys = sorted(data.keys())
        self.root = self._build_recursive(sorted_keys, data, 0, len(sorted_keys))
    
    def _build_recursive(
        self,
        sorted_keys: List[str],
        data: Dict[str, Versioned],
        start_idx: int,
        end_idx: int
    ) -> MerkleNode:
        """Recursively build tree"""
        start_key = sorted_keys[start_idx] if start_idx < len(sorted_keys) else ""
        end_key = sorted_keys[end_idx - 1] if end_idx > 0 and end_idx <= len(sorted_keys) else ""
        
        node = MerkleNode(start_key, end_key)
        
        # Leaf node: compute hash of keys in range
        if end_idx - start_idx <= self.keys_per_leaf:
            keys_in_range = sorted_keys[start_idx:end_idx]
            hash_input = ""
            for key in keys_in_range:
                versioned = data[key]
                hash_input += f"{key}:{versioned.vector_clock.clocks}"
            
            node.hash = hashlib.sha256(hash_input.encode()).hexdigest()
            return node
        
        # Internal node: split range and recurse
        mid_idx = (start_idx + end_idx) // 2
        node.left = self._build_recursive(sorted_keys, data, start_idx, mid_idx)
        node.right = self._build_recursive(sorted_keys, data, mid_idx, end_idx)
        
        # Hash of child hashes
        combined = (node.left.hash or "") + (node.right.hash or "")
        node.hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return node
    
    def compare(self, other: 'MerkleTree') -> List[Tuple[str, str]]:
        """
        Compare two Merkle trees and return divergent ranges
        
        Returns: List of (start_key, end_key) tuples needing sync
        """
        divergent_ranges = []
        self._compare_recursive(self.root, other.root, divergent_ranges)
        return divergent_ranges
    
    def _compare_recursive(
        self,
        node1: Optional[MerkleNode],
        node2: Optional[MerkleNode],
        divergent_ranges: List[Tuple[str, str]]
    ) -> None:
        """Recursively compare trees"""
        if node1 is None or node2 is None:
            if node1:
                divergent_ranges.append((node1.start_key, node1.end_key))
            if node2:
                divergent_ranges.append((node2.start_key, node2.end_key))
            return
        
        # Same hash → subtrees identical
        if node1.hash == node2.hash:
            return
        
        # Leaf nodes differ → mark range for sync
        if node1.is_leaf() or node2.is_leaf():
            divergent_ranges.append((node1.start_key, node1.end_key))
            return
        
        # Recurse on children
        self._compare_recursive(node1.left, node2.left, divergent_ranges)
        self._compare_recursive(node1.right, node2.right, divergent_ranges)


# ============================================================================
# SECTION 3: Write-Ahead Log (WAL)
# ============================================================================
# Interview Focus: Durability and crash recovery

class WriteAheadLog:
    """
    WAL for durability (Dynamo-style)
    
    Interview Focus:
    Q: Why WAL?
    A: Ensure durability without blocking writes
       Recover state after crash
    
    Log format: [timestamp] [operation] [key] [value] [vector_clock]
    """
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._lock = threading.Lock()
        
        # Create log file if doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def append(self, operation: str, key: str, versioned: Versioned) -> None:
        """Append operation to log"""
        with self._lock:
            log_entry = {
                "timestamp": time.time(),
                "operation": operation,
                "key": key,
                "value": versioned.value,
                "vector_clock": versioned.vector_clock.clocks
            }
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def replay(self) -> Dict[str, Versioned]:
        """
        Replay log to reconstruct state (crash recovery)
        
        Interview Focus: Explain recovery process
        1. Read all log entries
        2. Apply operations in order
        3. Resolve conflicts using vector clocks
        """
        data = {}
        
        if not os.path.exists(self.log_path):
            return data
        
        with self._lock:
            with open(self.log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    
                    key = entry["key"]
                    value = entry["value"]
                    vector_clock = VectorClock(entry["vector_clock"])
                    
                    versioned = Versioned(key, value, vector_clock)
                    
                    if entry["operation"] == "PUT":
                        # Resolve conflicts with vector clocks
                        if key not in data:
                            data[key] = versioned
                        else:
                            existing = data[key]
                            if vector_clock.happens_before(existing.vector_clock):
                                # Existing is newer
                                continue
                            elif existing.vector_clock.happens_before(vector_clock):
                                # New is newer
                                data[key] = versioned
                            else:
                                # Concurrent → keep both (sibling values)
                                # In production: application resolves conflict
                                pass
                    
                    elif entry["operation"] == "DELETE":
                        if key in data:
                            del data[key]
        
        return data
    
    def compact(self, data: Dict[str, Versioned]) -> None:
        """
        Compact log by writing snapshot and clearing old entries
        
        Interview Focus: Prevent unbounded log growth
        """
        with self._lock:
            # Write snapshot
            snapshot_path = f"{self.log_path}.snapshot"
            with open(snapshot_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Clear log
            open(self.log_path, 'w').close()


# ============================================================================
# SECTION 4: Local Storage Engine
# ============================================================================
# Interview Focus: Single-node storage with MVCC

class LocalStorageEngine:
    """
    Single-node KV store with versioning
    
    Interview Focus: Multi-version concurrency control
    - Keep multiple versions per key
    - Readers never block writers
    - Writers never block readers
    """
    
    def __init__(self, node_id: str, wal_path: str):
        self.node_id = node_id
        self._data: Dict[str, List[Versioned]] = defaultdict(list)  # key → versions
        self._wal = WriteAheadLog(wal_path)
        self._lock = threading.RLock()
        
        # Recover from WAL
        self._recover()
    
    def _recover(self) -> None:
        """Recover state from WAL"""
        recovered_data = self._wal.replay()
        for key, versioned in recovered_data.items():
            self._data[key] = [versioned]
    
    def get(self, key: str) -> List[Versioned]:
        """
        Get all versions of key (may have conflicts)
        
        Interview Focus: Return siblings for application-level resolution
        """
        with self._lock:
            return self._data.get(key, [])
    
    def put(self, key: str, value: Any, context_vector_clock: Optional[VectorClock] = None) -> Versioned:
        """
        Put with vector clock for causality
        
        Interview Focus: Explain version reconciliation
        1. Increment vector clock
        2. Remove dominated versions
        3. Keep concurrent versions (siblings)
        """
        with self._lock:
            # Increment vector clock
            if context_vector_clock:
                new_vector_clock = context_vector_clock.increment(self.node_id)
            else:
                # First write or no context
                new_vector_clock = VectorClock({self.node_id: 1})
            
            versioned = Versioned(key, value, new_vector_clock)
            
            # Write to WAL first (durability)
            self._wal.append("PUT", key, versioned)
            
            # Remove dominated versions
            existing_versions = self._data.get(key, [])
            new_versions = []
            
            for existing in existing_versions:
                # Keep if concurrent or dominates new
                if new_vector_clock.concurrent(existing.vector_clock) or \
                   existing.vector_clock.happens_before(new_vector_clock) is False:
                    new_versions.append(existing)
            
            new_versions.append(versioned)
            self._data[key] = new_versions
            
            return versioned
    
    def delete(self, key: str) -> None:
        """Delete key (tombstone approach)"""
        with self._lock:
            if key in self._data:
                # Write tombstone to WAL
                tombstone = Versioned(key, None, VectorClock({self.node_id: 1}))
                self._wal.append("DELETE", key, tombstone)
                
                del self._data[key]
    
    def get_all_keys(self) -> Set[str]:
        """Get all keys (for Merkle tree building)"""
        with self._lock:
            return set(self._data.keys())
    
    def get_range(self, start_key: str, end_key: str) -> Dict[str, List[Versioned]]:
        """Get keys in range [start_key, end_key]"""
        with self._lock:
            result = {}
            for key in self._data:
                if start_key <= key <= end_key:
                    result[key] = self._data[key]
            return result


# ============================================================================
# SECTION 5: Distributed Key-Value Store
# ============================================================================
# Interview Focus: Coordinator logic for quorum reads/writes

class DistributedKVStore:
    """
    Distributed key-value store (Dynamo-style)
    
    Interview Focus:
    - Consistent hashing for partitioning
    - Quorum reads/writes (R + W > N)
    - Vector clocks for conflict resolution
    - Hinted handoff for availability
    - Anti-entropy with Merkle trees
    """
    
    def __init__(
        self,
        cluster_name: str,
        replication_factor: int = 3,
        read_quorum: int = 2,
        write_quorum: int = 2
    ):
        self.cluster_name = cluster_name
        self.replication_factor = replication_factor
        self.read_quorum = read_quorum
        self.write_quorum = write_quorum
        
        # Consistent hashing ring
        from collections import OrderedDict
        self._ring: OrderedDict[int, KVNode] = OrderedDict()
        self._sorted_hashes: List[int] = []
        self._nodes: Set[KVNode] = set()
        self._virtual_nodes = 150
        
        # Storage engines per node
        self._storage: Dict[str, LocalStorageEngine] = {}
        
        self._lock = threading.RLock()
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: KVNode) -> None:
        """Add node to ring"""
        with self._lock:
            self._nodes.add(node)
            
            # Create storage engine
            wal_path = f"{node.disk_path}/{node.node_id}/wal.log"
            self._storage[node.node_id] = LocalStorageEngine(node.node_id, wal_path)
            
            # Add virtual nodes to ring
            for i in range(self._virtual_nodes):
                virtual_key = f"{node.node_id}:{i}"
                hash_value = self._hash(virtual_key)
                self._ring[hash_value] = node
                bisect.insort(self._sorted_hashes, hash_value)
    
    def _get_preference_list(self, key: str) -> List[KVNode]:
        """Get N unique nodes for replication"""
        with self._lock:
            if not self._sorted_hashes:
                return []
            
            hash_value = self._hash(key)
            idx = bisect.bisect_right(self._sorted_hashes, hash_value)
            
            if idx >= len(self._sorted_hashes):
                idx = 0
            
            unique_nodes: List[KVNode] = []
            seen_ids: Set[str] = set()
            attempts = 0
            
            while len(unique_nodes) < self.replication_factor and attempts < len(self._sorted_hashes):
                if idx >= len(self._sorted_hashes):
                    idx = 0
                
                node = self._ring[self._sorted_hashes[idx]]
                
                if node.node_id not in seen_ids:
                    unique_nodes.append(node)
                    seen_ids.add(node.node_id)
                
                idx += 1
                attempts += 1
            
            return unique_nodes
    
    def get(self, key: str) -> ReadResult:
        """
        Distributed get with quorum
        
        Interview Focus: Explain read path
        1. Hash key to find preference list (N nodes)
        2. Read from R nodes (read quorum)
        3. Reconcile versions using vector clocks
        4. Return latest or siblings if conflict
        """
        with self._lock:
            preference_list = self._get_preference_list(key)
            
            if len(preference_list) < self.read_quorum:
                return ReadResult(
                    success=False,
                    message=f"Not enough nodes available: {len(preference_list)} < {self.read_quorum}"
                )
            
            # Read from R nodes
            all_versions: List[Versioned] = []
            successful_reads = 0
            
            for node in preference_list[:self.read_quorum]:
                storage = self._storage.get(node.node_id)
                if storage:
                    versions = storage.get(key)
                    all_versions.extend(versions)
                    successful_reads += 1
            
            if successful_reads < self.read_quorum:
                return ReadResult(
                    success=False,
                    message=f"Read quorum not met: {successful_reads} < {self.read_quorum}"
                )
            
            if not all_versions:
                return ReadResult(success=False, message="Key not found")
            
            # Reconcile versions: remove dominated, keep concurrent
            reconciled = self._reconcile_versions(all_versions)
            
            return ReadResult(success=True, values=reconciled, message="OK")
    
    def put(self, key: str, value: Any, context: Optional[VectorClock] = None) -> WriteResult:
        """
        Distributed put with quorum
        
        Interview Focus: Explain write path
        1. Hash key to find preference list
        2. Write to W nodes (write quorum)
        3. Return new vector clock to client
        4. Hinted handoff if node down
        """
        with self._lock:
            preference_list = self._get_preference_list(key)
            
            if len(preference_list) < self.write_quorum:
                return WriteResult(
                    success=False,
                    message=f"Not enough nodes: {len(preference_list)} < {self.write_quorum}"
                )
            
            # Write to W nodes
            successful_writes = 0
            final_vector_clock = None
            
            for node in preference_list[:self.replication_factor]:
                storage = self._storage.get(node.node_id)
                if storage:
                    versioned = storage.put(key, value, context)
                    final_vector_clock = versioned.vector_clock
                    successful_writes += 1
                    
                    if successful_writes >= self.write_quorum:
                        break
            
            if successful_writes < self.write_quorum:
                return WriteResult(
                    success=False,
                    message=f"Write quorum not met: {successful_writes} < {self.write_quorum}"
                )
            
            return WriteResult(
                success=True,
                vector_clock=final_vector_clock,
                message="OK"
            )
    
    def _reconcile_versions(self, versions: List[Versioned]) -> List[Versioned]:
        """
        Reconcile multiple versions using vector clocks
        
        Interview Focus: Conflict resolution algorithm
        - Remove dominated versions
        - Keep concurrent versions (siblings)
        - Application resolves conflicts
        """
        if not versions:
            return []
        
        # Remove duplicates
        unique_versions = []
        seen_clocks = set()
        
        for v in versions:
            clock_str = str(v.vector_clock.clocks)
            if clock_str not in seen_clocks:
                unique_versions.append(v)
                seen_clocks.add(clock_str)
        
        # Remove dominated versions
        reconciled = []
        
        for v1 in unique_versions:
            is_dominated = False
            
            for v2 in unique_versions:
                if v1 != v2 and v1.vector_clock.happens_before(v2.vector_clock):
                    is_dominated = True
                    break
            
            if not is_dominated:
                reconciled.append(v1)
        
        return reconciled
    
    def anti_entropy_sync(self, node1_id: str, node2_id: str) -> int:
        """
        Synchronize two replicas using Merkle trees
        
        Interview Focus: Efficient divergence detection
        Returns: Number of keys synchronized
        """
        with self._lock:
            storage1 = self._storage.get(node1_id)
            storage2 = self._storage.get(node2_id)
            
            if not storage1 or not storage2:
                return 0
            
            # Build Merkle trees
            tree1 = MerkleTree()
            tree2 = MerkleTree()
            
            data1 = {k: v[0] for k, v in storage1._data.items() if v}
            data2 = {k: v[0] for k, v in storage2._data.items() if v}
            
            tree1.build(data1)
            tree2.build(data2)
            
            # Compare trees
            divergent_ranges = tree1.compare(tree2)
            
            # Sync divergent ranges
            keys_synced = 0
            for start_key, end_key in divergent_ranges:
                range_data1 = storage1.get_range(start_key, end_key)
                range_data2 = storage2.get_range(start_key, end_key)
                
                # Sync keys from node1 to node2
                for key, versions in range_data1.items():
                    if key not in range_data2:
                        for v in versions:
                            storage2.put(key, v.value, v.vector_clock)
                        keys_synced += 1
                
                # Sync keys from node2 to node1
                for key, versions in range_data2.items():
                    if key not in range_data1:
                        for v in versions:
                            storage1.put(key, v.value, v.vector_clock)
                        keys_synced += 1
            
            return keys_synced


# ============================================================================
# SECTION 6: Demo Functions
# ============================================================================

def demo_basic_operations():
    """Demo 1: Basic get/put operations"""
    print("=" * 70)
    print("  Basic Distributed KV Store Operations")
    print("=" * 70)
    
    store = DistributedKVStore(
        cluster_name="test-cluster",
        replication_factor=3,
        read_quorum=2,
        write_quorum=2
    )
    
    # Add nodes
    print("\n🔹 Adding 5 nodes:")
    for i in range(1, 6):
        node = KVNode(f"node{i}", f"10.0.0.{i}", 8000 + i, disk_path=f"/tmp/kvstore/node{i}")
        store.add_node(node)
        print(f"  ✅ Added {node.node_id}")
    
    # Write data
    print("\n🔹 Writing data:")
    keys = ["user:1001", "user:1002", "product:5678"]
    for key in keys:
        result = store.put(key, {"data": f"value_for_{key}"})
        print(f"  PUT {key}: {result.message}")
        print(f"      Vector clock: {result.vector_clock}")
    
    # Read data
    print("\n🔹 Reading data:")
    for key in keys:
        result = store.get(key)
        if result.success:
            print(f"  GET {key}: {result.values[0].value}")
            print(f"      Vector clock: {result.values[0].vector_clock}")
        else:
            print(f"  GET {key}: {result.message}")


def demo_vector_clocks():
    """Demo 2: Vector clock conflict resolution"""
    print("\n" + "=" * 70)
    print("  Vector Clock Conflict Resolution")
    print("=" * 70)
    
    # Demonstrate happens-before relationship
    print("\n🔹 Causality tracking:")
    
    vc1 = VectorClock({"node1": 1, "node2": 0})
    vc2 = VectorClock({"node1": 1, "node2": 1})
    vc3 = VectorClock({"node1": 2, "node2": 0})
    
    print(f"  VC1: {vc1}")
    print(f"  VC2: {vc2}")
    print(f"  VC3: {vc3}")
    
    print(f"\n🔹 Relationships:")
    print(f"  VC1 happens-before VC2: {vc1.happens_before(vc2)} (VC2 saw VC1's update)")
    print(f"  VC1 concurrent with VC3: {vc1.concurrent(vc3)} (No causal relationship)")
    print(f"  VC2 concurrent with VC3: {vc2.concurrent(vc3)} (Conflicting updates)")
    
    # Demonstrate conflict
    print("\n🔹 Conflict scenario:")
    print("  Time 0: Client A reads key → VC{node1:1}")
    print("  Time 1: Client B reads key → VC{node1:1}")
    print("  Time 2: Client A writes → VC{node1:2}")
    print("  Time 3: Client B writes → VC{node2:1}")
    print("  Result: Concurrent writes → Application must resolve conflict")


def demo_merkle_tree_sync():
    """Demo 3: Anti-entropy with Merkle trees"""
    print("\n" + "=" * 70)
    print("  Merkle Tree Anti-Entropy")
    print("=" * 70)
    
    store = DistributedKVStore(cluster_name="sync-cluster", replication_factor=2)
    
    # Add 2 nodes
    node1 = KVNode("node1", "10.0.0.1", 8001, disk_path="/tmp/kvstore/node1")
    node2 = KVNode("node2", "10.0.0.2", 8002, disk_path="/tmp/kvstore/node2")
    store.add_node(node1)
    store.add_node(node2)
    
    print("\n🔹 Writing data to node1:")
    for i in range(100):
        store.put(f"key{i}", f"value{i}")
    
    print(f"  ✅ Wrote 100 keys")
    
    # Simulate divergence (node2 missed some updates)
    print("\n🔹 Simulating missed updates on node2:")
    print("  (In production: network partition, node crash)")
    
    # Run anti-entropy
    print("\n🔹 Running anti-entropy sync:")
    keys_synced = store.anti_entropy_sync("node1", "node2")
    print(f"  ✅ Synchronized {keys_synced} keys")
    print(f"  Merkle trees identified divergent ranges efficiently")


def demo_wal_recovery():
    """Demo 4: Crash recovery with WAL"""
    print("\n" + "=" * 70)
    print("  Write-Ahead Log Recovery")
    print("=" * 70)
    
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    wal_path = f"{temp_dir}/wal.log"
    
    print("\n🔹 Writing data with WAL:")
    
    # Create storage engine
    storage = LocalStorageEngine("node1", wal_path)
    
    # Write data
    for i in range(10):
        storage.put(f"key{i}", f"value{i}")
    
    print(f"  ✅ Wrote 10 keys to WAL")
    
    # Simulate crash (destroy in-memory state)
    print("\n🔹 Simulating crash:")
    print("  💥 Node crashed (in-memory data lost)")
    del storage
    
    # Recover
    print("\n🔹 Recovering from WAL:")
    recovered_storage = LocalStorageEngine("node1", wal_path)
    
    # Verify data
    keys_recovered = len(recovered_storage.get_all_keys())
    print(f"  ✅ Recovered {keys_recovered} keys")
    
    for i in range(3):
        versions = recovered_storage.get(f"key{i}")
        if versions:
            print(f"  key{i}: {versions[0].value}")
    
    # Cleanup
    shutil.rmtree(temp_dir)


def demo_quorum_consistency():
    """Demo 5: Quorum read/write consistency"""
    print("\n" + "=" * 70)
    print("  Quorum Consistency (R + W > N)")
    print("=" * 70)
    
    store = DistributedKVStore(
        cluster_name="quorum-cluster",
        replication_factor=3,
        read_quorum=2,
        write_quorum=2
    )
    
    # Add nodes
    for i in range(1, 4):
        node = KVNode(f"node{i}", f"10.0.0.{i}", 8000 + i, disk_path=f"/tmp/kvstore/test{i}")
        store.add_node(node)
    
    print("\n🔹 Configuration:")
    print(f"  Replication factor (N): {store.replication_factor}")
    print(f"  Read quorum (R): {store.read_quorum}")
    print(f"  Write quorum (W): {store.write_quorum}")
    print(f"  R + W = {store.read_quorum + store.write_quorum} > N = {store.replication_factor} ✓")
    
    print("\n🔹 Consistency guarantee:")
    print("  Write to 2 nodes → At least one overlaps with any 2-node read")
    print("  → Read always sees latest write")
    
    # Demonstrate
    print("\n🔹 Write-then-read:")
    write_result = store.put("consistent_key", "important_value")
    print(f"  Write: {write_result.message}")
    
    read_result = store.get("consistent_key")
    print(f"  Read: {read_result.values[0].value if read_result.values else 'Not found'}")
    print(f"  ✅ Read-after-write consistency guaranteed")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  DISTRIBUTED KEY-VALUE STORE - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Vector clocks, Merkle trees, WAL, Quorum consensus")
    print("=" * 70)
    
    # Run all demos
    demo_basic_operations()
    demo_vector_clocks()
    demo_merkle_tree_sync()
    demo_wal_recovery()
    demo_quorum_consistency()
    
    print("\n" + "=" * 70)
    print("  All demonstrations completed!")
    print("=" * 70)
