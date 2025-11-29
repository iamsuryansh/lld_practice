# File Storage System (Dropbox-like) - Low Level Design

## Problem Statement

Design a **cloud file storage and synchronization system** like Dropbox, Google Drive, or OneDrive. The system must:

1. **Efficient storage**: Deduplication to save space (40-60% savings)
2. **Fast sync**: Only transfer changed data (delta sync)
3. **Version control**: Keep file history for time-travel
4. **Conflict resolution**: Handle concurrent edits from multiple clients
5. **Scalability**: Support billions of files, petabytes of data
6. **Reliability**: Zero data loss, 99.99% availability

### Real-World Context
Used by: Dropbox, Google Drive, OneDrive, iCloud, Box

### Key Requirements
- **Deduplication**: <50% storage redundancy
- **Sync efficiency**: 90%+ bandwidth savings for small changes
- **Versioning**: Keep 30-90 days of history
- **Conflict resolution**: Automatic when possible
- **Performance**: <5 seconds sync time for typical files

---

## Implementation Phases

### Phase 1: Content-Addressable Storage with Chunking (20-25 minutes)

**Core concept**: SHA-256 hash as chunk ID enables automatic deduplication

```python
class ChunkSize(Enum):
    """Standard chunk sizes for different file types"""
    SMALL = 4 * 1024        # 4 KB for small files
    MEDIUM = 64 * 1024      # 64 KB for documents
    LARGE = 1 * 1024 * 1024 # 1 MB for large files

@dataclass
class Chunk:
    """
    Content-addressable chunk
    
    Key insight: Hash = ID means identical content → same chunk
    This enables automatic deduplication across all files and users
    """
    chunk_id: str           # SHA-256 hash (64 hex chars)
    data: bytes
    size: int
    ref_count: int = 0      # Number of files referencing this chunk
    
    @staticmethod
    def create(data: bytes) -> 'Chunk':
        """Create chunk with content-based ID"""
        chunk_id = hashlib.sha256(data).hexdigest()
        return Chunk(chunk_id, data, len(data))

class ChunkStore:
    """
    Content-addressable storage for chunks
    
    Interview focus:
    - Deduplication: Same content → same chunk ID → stored once
    - Reference counting: Delete only when ref_count = 0
    - Space savings: O(unique_chunks) vs O(total_chunks)
    """
    
    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}
        self._total_bytes = 0
        self._dedupe_saved_bytes = 0
    
    def put(self, data: bytes) -> str:
        """
        Store chunk and return ID
        
        Time: O(N) for SHA-256 where N = chunk size
        Space: O(1) if deduplicated, O(N) if new chunk
        """
        chunk = Chunk.create(data)
        
        if chunk.chunk_id in self._chunks:
            # Deduplication! Don't store again
            self._chunks[chunk.chunk_id].ref_count += 1
            self._dedupe_saved_bytes += chunk.size
        else:
            # New unique chunk
            chunk.ref_count = 1
            self._chunks[chunk.chunk_id] = chunk
            self._total_bytes += chunk.size
        
        return chunk.chunk_id
    
    def get(self, chunk_id: str) -> Optional[bytes]:
        """Retrieve chunk data - O(1)"""
        chunk = self._chunks.get(chunk_id)
        return chunk.data if chunk else None
    
    def decrement_ref(self, chunk_id: str) -> None:
        """
        Decrement reference count and garbage collect if 0
        
        Interview question: "Why reference counting?"
        Answer: "Can't delete chunk when deleting one file - other files might use it"
        """
        if chunk_id in self._chunks:
            self._chunks[chunk_id].ref_count -= 1
            
            if self._chunks[chunk_id].ref_count == 0:
                # No files reference this chunk, safe to delete
                size = self._chunks[chunk_id].size
                del self._chunks[chunk_id]
                self._total_bytes -= size
```

**Interview deep-dive**:

**Q: "Why chunk files instead of storing whole files?"**

A: "Multiple benefits:
1. **Deduplication**: If 10 users have same 1GB movie, store once (1GB) not 10 times (10GB)
2. **Delta sync**: Change 1 page of 1000-page PDF → transfer 1 chunk (64KB) not full file (10MB)
3. **Parallel upload**: Upload 100 chunks in parallel → 100x faster
4. **Bandwidth**: Resume interrupted uploads from last chunk, not from start"

**Q: "What's the optimal chunk size?"**

A: "Trade-offs:

| Chunk Size | Pros | Cons |
|------------|------|------|
| **Small (4KB)** | Better deduplication, fine-grained delta sync | More metadata overhead, more chunks to manage |
| **Medium (64KB)** | Balanced | Moderate metadata, moderate dedup |
| **Large (1MB)** | Less metadata, fewer chunks | Poor deduplication, coarse delta sync |

**Production**: Dropbox uses ~4MB (variable size), Google Drive ~8MB, rsync ~700 bytes.

**Variable-size (Rabin fingerprinting)** is better for dedup but more complex."

---

### Phase 2: Delta Sync (Rsync Algorithm) (25-30 minutes)

**Core concept**: Only transfer changed chunks, not entire file

```python
@dataclass
class ChunkSignature:
    """
    Signature for chunk (used in delta sync)
    
    Interview focus: Explain rolling hash vs strong hash
    """
    chunk_id: str           # Strong hash (SHA-256)
    rolling_hash: int       # Fast hash for quick comparison
    offset: int             # Byte offset in file

class DeltaSync:
    """
    Rsync-style delta sync algorithm
    
    Key insight: 
    1. Client has old file, server has new file
    2. Server sends signatures (hashes) of new file
    3. Client compares with old file chunks
    4. Client requests only CHANGED chunks
    5. Bandwidth: O(changed) instead of O(file_size)
    """
    
    @staticmethod
    def compute_signature(
        chunk_map: FileChunkMap,
        chunk_store: ChunkStore
    ) -> List[ChunkSignature]:
        """
        Compute signature for existing file
        
        Time: O(chunks)
        Space: O(chunks)
        """
        signatures = []
        offset = 0
        
        for chunk_id in chunk_map.chunk_ids:
            data = chunk_store.get(chunk_id)
            if data:
                rolling_hash = hash(data) % (2**32)  # Fast hash
                signatures.append(ChunkSignature(
                    chunk_id=chunk_id,
                    rolling_hash=rolling_hash,
                    offset=offset
                ))
                offset += len(data)
        
        return signatures
    
    @staticmethod
    def compute_delta(
        new_chunks: List[bytes],
        old_signatures: List[ChunkSignature]
    ) -> Tuple[List[str], List[bytes]]:
        """
        Compute delta between new file and old signatures
        
        Returns:
        - reused_chunk_ids: Chunks that match old file (don't transfer)
        - new_chunk_data: New/modified chunks (transfer these)
        
        Time: O(new_chunks) with hash lookup
        Bandwidth savings: If 10MB file, 1MB changed → transfer 1MB not 10MB
        """
        # Build hash map for O(1) lookup
        old_sig_map = {sig.chunk_id: sig for sig in old_signatures}
        
        reused_chunk_ids = []
        new_chunk_data = []
        
        for chunk in new_chunks:
            chunk_id = hashlib.sha256(chunk).hexdigest()
            
            if chunk_id in old_sig_map:
                # Chunk exists in old file - REUSE
                reused_chunk_ids.append(chunk_id)
            else:
                # New or modified chunk - TRANSFER
                new_chunk_data.append(chunk)
        
        return reused_chunk_ids, new_chunk_data
```

**Interview insight**:

**Q: "Explain rsync algorithm step-by-step"**

A: "Real-world example:
```
Scenario: Edit 1 line in 1000-line source file (100KB)

Step 1: Client has old file (version 1)
Step 2: Server has new file (version 2)

Without delta sync:
- Transfer entire 100KB → slow, wasteful

With delta sync:
1. Server: Compute signatures of new file
   - Chunk 1 (64KB): SHA = abc123...
   - Chunk 2 (36KB): SHA = def456... (CHANGED)
   
2. Server → Client: Send signatures (64 bytes each)
   - Transfer: 128 bytes (2 signatures)
   
3. Client: Compare signatures with old file chunks
   - Chunk 1: Match! (SHA = abc123...)
   - Chunk 2: Different (SHA = ghi789... vs def456...)
   
4. Client → Server: Request chunk 2 only
   
5. Server → Client: Send chunk 2 (36KB)
   
6. Client: Reconstruct new file
   - Reuse chunk 1 from local (64KB)
   - Download chunk 2 (36KB)
   
Total transfer: 128 bytes + 36KB = ~36KB vs 100KB
Savings: 64% bandwidth
```

**Production**: Dropbox achieves ~90% bandwidth savings on average."

---

### Phase 3: Version Control with Snapshots (20-25 minutes)

**Core concept**: Immutable snapshots with parent pointers (Git-style)

```python
@dataclass
class FileVersion:
    """
    Immutable file version snapshot
    
    Key insight: Never modify versions, always create new ones
    This enables perfect history and easy rollback
    """
    version_id: str
    file_id: str
    chunk_map: FileChunkMap      # Which chunks make up this version
    size: int
    modified_time: float
    checksum: str                 # SHA-256 of entire file
    parent_version: Optional[str] = None  # Git-style parent pointer
    
    @staticmethod
    def create(
        file_id: str,
        chunk_map: FileChunkMap,
        parent_version: Optional[str] = None
    ) -> 'FileVersion':
        """
        Create new version
        
        Interview focus: Explain immutability benefits
        """
        version_id = hashlib.sha256(
            f"{file_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Compute file checksum from chunk IDs
        checksum = hashlib.sha256()
        for chunk_id in chunk_map.chunk_ids:
            checksum.update(chunk_id.encode())
        
        return FileVersion(
            version_id=version_id,
            file_id=file_id,
            chunk_map=chunk_map,
            size=chunk_map.total_size,
            modified_time=time.time(),
            checksum=checksum.hexdigest(),
            parent_version=parent_version
        )

@dataclass
class FileMetadata:
    """File metadata with version history"""
    file_id: str
    path: str
    name: str
    owner: str
    created_time: float
    versions: List[FileVersion] = field(default_factory=list)
    current_version: Optional[str] = None
    
    def add_version(self, version: FileVersion) -> None:
        """Add new version to history"""
        self.versions.append(version)
        self.current_version = version.version_id
    
    def get_version(self, version_id: str) -> Optional[FileVersion]:
        """Get specific version (time-travel!)"""
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None
```

**Interview Q&A**:

**Q: "How does version control save space?"**

A: "Example:
```
v1: [chunk_A, chunk_B, chunk_C] → 300KB
v2: [chunk_A, chunk_D, chunk_C] → 300KB (only chunk_B → chunk_D changed)

Without deduplication:
- Store v1: 300KB
- Store v2: 300KB
- Total: 600KB

With deduplication:
- chunk_A: 100KB (ref_count=2, shared by v1 and v2)
- chunk_B: 100KB (ref_count=1, only v1)
- chunk_C: 100KB (ref_count=2, shared by v1 and v2)
- chunk_D: 100KB (ref_count=1, only v2)
- Total: 400KB

Savings: 33% space for 2 versions!
```

With more versions of mostly unchanged files → 80-90% savings."

---

### Phase 4: Conflict Resolution (25-30 minutes)

**Core concept**: Detect conflicts, resolve automatically when possible

```python
class ConflictType(Enum):
    """Types of sync conflicts"""
    MODIFY_MODIFY = "both_modified"   # Most common
    MODIFY_DELETE = "modified_deleted"
    DELETE_DELETE = "both_deleted"

@dataclass
class SyncConflict:
    """Sync conflict between versions"""
    file_id: str
    conflict_type: ConflictType
    local_version: Optional[FileVersion]
    remote_version: Optional[FileVersion]
    common_ancestor: Optional[FileVersion]  # For 3-way merge

class ConflictResolver:
    """
    Conflict resolution strategies
    
    Interview focus: Explain when each strategy is appropriate
    """
    
    @staticmethod
    def resolve_by_timestamp(conflict: SyncConflict) -> Optional[FileVersion]:
        """
        Last-write-wins (LWW)
        
        Pros: Simple, always resolves automatically
        Cons: May lose data (discards earlier edit)
        
        Use case: Non-critical files (logs, caches)
        """
        if not conflict.local_version:
            return conflict.remote_version
        if not conflict.remote_version:
            return conflict.local_version
        
        # Compare timestamps
        if conflict.local_version.modified_time > conflict.remote_version.modified_time:
            return conflict.local_version
        else:
            return conflict.remote_version
    
    @staticmethod
    def three_way_merge(
        conflict: SyncConflict,
        chunk_store: ChunkStore
    ) -> Optional[FileVersion]:
        """
        Three-way merge (like Git)
        
        Algorithm:
        1. Find common ancestor (last synced version)
        2. Compute changes: ancestor → local, ancestor → remote
        3. Merge non-overlapping changes
        4. Flag overlapping changes as conflict
        
        Pros: Preserves both edits when possible
        Cons: Complex, may still have conflicts
        
        Use case: Code files, documents (critical data)
        """
        if not conflict.common_ancestor:
            # No common ancestor, fallback to LWW
            return ConflictResolver.resolve_by_timestamp(conflict)
        
        # Get chunks for each version
        ancestor_chunks = conflict.common_ancestor.chunk_map.chunk_ids
        local_chunks = conflict.local_version.chunk_map.chunk_ids if conflict.local_version else []
        remote_chunks = conflict.remote_version.chunk_map.chunk_ids if conflict.remote_version else []
        
        # Identify changes
        local_changes = set(local_chunks) - set(ancestor_chunks)
        remote_changes = set(remote_chunks) - set(ancestor_chunks)
        
        # Check for overlapping changes
        if local_changes & remote_changes:
            # Conflict! Both modified same chunk
            # In production, create conflict file: "file.txt" and "file (conflicted copy).txt"
            return None  # Manual resolution needed
        
        # No overlap, merge changes
        merged_chunks = list(set(local_chunks) | set(remote_chunks))
        
        # Create merged version
        # (Simplified - production needs proper chunk ordering)
        return conflict.local_version  # Placeholder
```

**Interview deep-dive**:

**Q: "How does Dropbox handle conflicts?"**

A: "Multi-strategy approach:

**Strategy 1: Optimistic locking** (prevent conflicts)
```
Client A opens file → lock file on server
Client B tries to open → show 'read-only, locked by Client A'
Client A saves → release lock
Client B can now edit
```

**Strategy 2: Conflict detection** (detect conflicts)
```
Both clients edit offline
Both upload
Server detects: version_A and version_B have same parent
→ CONFLICT detected
```

**Strategy 3: Automatic resolution** (simple conflicts)
```
If changes in different chunks:
  → Three-way merge, combine both edits
If changes in same chunk:
  → Create conflict files
```

**Strategy 4: Manual resolution** (complex conflicts)
```
Create files:
- file.txt (remote version, usually latest)
- file (conflicted copy 2024-11-29).txt (local version)

User manually merges
```"

---

### Phase 5: Complete File Storage System (25-30 minutes)

**Core concept**: Integrate all components

```python
class FileStorageSystem:
    """
    Dropbox-like file storage system
    
    Features:
    - Content-addressable chunks (deduplication)
    - Delta sync (bandwidth savings)
    - Version history (time-travel)
    - Conflict resolution (automatic + manual)
    """
    
    def __init__(self):
        self.chunk_store = ChunkStore()
        self._files: Dict[str, FileMetadata] = {}
    
    def upload_file(
        self,
        path: str,
        data: bytes,
        owner: str,
        parent_version: Optional[str] = None
    ) -> str:
        """
        Upload file (new or update)
        
        Steps:
        1. Generate file_id from path + owner
        2. Chunk file
        3. Store chunks (deduplicated automatically)
        4. Create file version
        5. Update metadata
        """
        # Generate deterministic file ID
        file_id = hashlib.sha256(f"{owner}:{path}".encode()).hexdigest()[:16]
        
        # Chunk file
        chunks = FileChunker.chunk_file(data)
        
        # Store chunks and build chunk map
        chunk_ids = []
        chunk_offsets = [0]
        offset = 0
        
        for chunk in chunks:
            chunk_id = self.chunk_store.put(chunk)
            chunk_ids.append(chunk_id)
            offset += len(chunk)
            chunk_offsets.append(offset)
        
        chunk_offsets.pop()  # Remove last offset
        
        # Create chunk map
        chunk_map = FileChunkMap(
            file_id=file_id,
            chunk_ids=chunk_ids,
            chunk_offsets=chunk_offsets,
            total_size=len(data)
        )
        
        # Create version
        version = FileVersion.create(file_id, chunk_map, parent_version)
        
        # Update metadata
        if file_id in self._files:
            self._files[file_id].add_version(version)
        else:
            metadata = FileMetadata(
                file_id=file_id,
                path=path,
                name=os.path.basename(path),
                owner=owner,
                created_time=time.time()
            )
            metadata.add_version(version)
            self._files[file_id] = metadata
        
        return version.version_id
    
    def sync_file(
        self,
        path: str,
        data: bytes,
        owner: str,
        local_version: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Sync file with conflict detection
        
        Returns: (version_id, had_conflict)
        
        Flow:
        1. Check if file exists on server
        2. If not, upload as new
        3. If exists, check for conflict (local_version != server version)
        4. If conflict, resolve
        5. If no conflict, delta sync
        """
        file_id = hashlib.sha256(f"{owner}:{path}".encode()).hexdigest()[:16]
        
        # New file
        if file_id not in self._files:
            version_id = self.upload_file(path, data, owner)
            return version_id, False
        
        metadata = self._files[file_id]
        current_version = metadata.get_current_version()
        
        # Check for conflict
        if local_version and current_version and local_version != current_version.version_id:
            # CONFLICT!
            conflict = SyncConflict(
                file_id=file_id,
                conflict_type=ConflictType.MODIFY_MODIFY,
                local_version=metadata.get_version(local_version),
                remote_version=current_version,
                common_ancestor=None  # Simplified
            )
            
            # Resolve conflict (use timestamp)
            winning_version = ConflictResolver.resolve_by_timestamp(conflict)
            
            if winning_version and winning_version.version_id == local_version:
                # Local wins, upload
                version_id = self.upload_file(path, data, owner, current_version.version_id)
                return version_id, True
            else:
                # Remote wins, return remote version
                return current_version.version_id, True
        
        # No conflict, delta sync
        if current_version:
            # Compute delta
            old_signatures = DeltaSync.compute_signature(
                current_version.chunk_map,
                self.chunk_store
            )
            
            new_chunks = FileChunker.chunk_file(data)
            reused_ids, new_chunk_data = DeltaSync.compute_delta(new_chunks, old_signatures)
            
            # Upload only new chunks (bandwidth savings!)
            version_id = self.upload_file(path, data, owner, current_version.version_id)
            return version_id, False
        else:
            # First version
            version_id = self.upload_file(path, data, owner)
            return version_id, False
```

**Interview insight**: This architecture enables:
- **40-60% space savings** from deduplication
- **90% bandwidth savings** from delta sync
- **Unlimited versions** with minimal storage cost
- **Fast conflict detection** with version IDs

---

## Critical Knowledge Points

### 1. Chunking Strategy Comparison

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Fixed-size** | Simple, fast | Poor dedup across boundaries | Dropbox (4MB chunks) |
| **Variable-size (Rabin)** | Better dedup | Complex, slower | Git, rsync |
| **Content-defined** | Best dedup | Most complex | Backup systems |

**Rabin fingerprinting** (rolling hash):
```python
def rabin_chunk(data: bytes) -> List[bytes]:
    """
    Variable-size chunking with rolling hash
    
    Key insight: Split at content boundaries, not fixed offsets
    Example: Insert 1 char at start of file
    - Fixed: ALL chunks change (bad!)
    - Rabin: Only first chunk changes (good!)
    """
    chunks = []
    window = 48  # Bytes
    mask = (1 << 13) - 1  # 13-bit mask → ~8KB avg chunk size
    
    chunk_start = 0
    for i in range(window, len(data)):
        # Compute rolling hash of window
        hash_val = hash(data[i-window:i])
        
        # Split if hash matches pattern (e.g., last 13 bits = 0)
        if (hash_val & mask) == 0:
            chunks.append(data[chunk_start:i])
            chunk_start = i
    
    chunks.append(data[chunk_start:])
    return chunks
```

### 2. Deduplication Math

**Example: 100 users, 1000 files each**

**Without deduplication**:
- Total files: 100 × 1000 = 100,000 files
- Avg file size: 1 MB
- Total storage: 100,000 × 1 MB = 100 GB

**With deduplication** (40% unique):
- Unique files: 40,000 (many files duplicated across users)
- Total storage: 40 GB
- **Savings: 60%**

**Real-world** (Dropbox engineering blog):
- Personal users: ~40% dedup ratio
- Enterprise users: ~60% dedup ratio (more shared files)
- Across all users: ~70% dedup ratio

### 3. Delta Sync Efficiency

**Scenario: Edit 1 line in source file**

| File Size | Changed | Without Delta | With Delta | Savings |
|-----------|---------|---------------|------------|---------|
| 100 KB | 100 bytes | 100 KB | 64 KB (1 chunk) | 36% |
| 1 MB | 1 KB | 1 MB | 64 KB (1 chunk) | 94% |
| 10 MB | 10 KB | 10 MB | 128 KB (2 chunks) | 99% |

**Key insight**: Larger files → bigger savings (fixed chunk overhead)

---

## Interview Q&A

### Q1: "How does Dropbox achieve such high deduplication rates?"

**Answer**: Multi-level deduplication:

**Level 1: File-level** (cheap, fast)
```python
def file_level_dedup(file_hash: str) -> bool:
    """If exact file exists, just create reference"""
    if file_hash in global_file_index:
        create_reference(file_hash)
        return True  # Deduplicated!
    return False  # Need to upload
```

**Level 2: Chunk-level** (moderate cost, good dedup)
```python
def chunk_level_dedup(chunks: List[bytes]) -> List[str]:
    """For each chunk, check if already stored"""
    chunk_ids = []
    for chunk in chunks:
        chunk_id = sha256(chunk)
        if chunk_id not in chunk_store:
            upload_chunk(chunk)  # Only upload new chunks
        chunk_ids.append(chunk_id)
    return chunk_ids
```

**Level 3: Cross-user** (highest dedup)
```
If 1000 users upload "Ubuntu.iso":
- User 1: Upload 2 GB
- Users 2-1000: Upload 0 bytes (reference User 1's chunks)
- Total storage: 2 GB instead of 2000 GB
- Savings: 99.9%
```

---

### Q2: "How to handle large files (>10GB)?"**

**Answer**: Chunked upload with resumption:

```python
def upload_large_file(file_path: str) -> str:
    """
    Upload large file in chunks with progress tracking
    """
    file_size = os.path.getsize(file_path)
    chunk_size = 4 * 1024 * 1024  # 4 MB chunks
    
    # Initialize upload session
    session_id = create_upload_session(file_path)
    
    uploaded_chunks = []
    offset = 0
    
    with open(file_path, 'rb') as f:
        while offset < file_size:
            # Read chunk
            chunk = f.read(chunk_size)
            
            # Upload with retry
            for retry in range(3):
                try:
                    chunk_id = upload_chunk(session_id, chunk, offset)
                    uploaded_chunks.append(chunk_id)
                    break
                except NetworkError:
                    if retry == 2:
                        # Save progress, allow resumption
                        save_upload_state(session_id, uploaded_chunks, offset)
                        raise
                    time.sleep(2 ** retry)  # Exponential backoff
            
            offset += len(chunk)
            update_progress(offset, file_size)
    
    # Finalize upload
    return commit_upload(session_id, uploaded_chunks)
```

**Benefits**:
- **Parallel**: Upload 10 chunks simultaneously
- **Resumable**: Network failure → resume from last chunk
- **Progress**: Show accurate progress bar

---

### Q3: "Explain Dropbox's architecture"**

**Answer**:

```
Client (Desktop/Mobile)
    ↓
[Sync Client] ← Local file monitor
    ↓
API Gateway (Load Balancer)
    ↓
[Metadata Servers] ← PostgreSQL (file metadata, versions, users)
    ↓
[Block Servers] ← Storage layer
    ↓
[Object Storage] ← S3/Azure (actual chunk data)
    ↓
[CDN] ← CloudFront (fast downloads)
```

**Components**:

**1. Sync Client**:
- Monitors local filesystem changes
- Computes chunk hashes
- Uploads only changed chunks
- Downloads remote changes

**2. Metadata Servers**:
- Store file metadata (path, owner, versions)
- Track chunk → file mappings
- Handle conflict detection
- Serve API requests

**3. Block Servers**:
- Store chunks in object storage
- Handle deduplication logic
- Manage reference counting
- Serve chunk downloads

**4. Object Storage (S3)**:
- Durable (99.999999999% durability)
- Scalable (petabytes)
- Geo-replicated
- Cheap (~$0.02/GB/month)

---

### Q4: "How to handle conflicts in real-time collaborative editing?"**

**Answer**: Operational Transformation (OT) or CRDTs

**Problem with simple versioning**:
```
User A: Types "Hello" at position 0
User B: Types "World" at position 0 (simultaneously)

Naive merge: "WorldHello" or "HelloWorld"?
Neither is correct user intent!
```

**Solution 1: Operational Transformation** (Google Docs)
```python
def transform(op1: Operation, op2: Operation) -> Operation:
    """
    Transform op1 against op2
    
    Example:
    op1: Insert "H" at pos 0
    op2: Insert "W" at pos 0
    
    Transform op1 after op2: Insert "H" at pos 1 (shifted!)
    """
    if op1.type == "INSERT" and op2.type == "INSERT":
        if op1.position < op2.position:
            return op1  # No change
        else:
            return Insert(op1.char, op1.position + 1)  # Shift
    # ... handle all op type combinations
```

**Solution 2: CRDTs** (Conflict-free Replicated Data Types)
```python
class CRDT:
    """
    Automatically resolves conflicts
    
    Key insight: Operations commute (order-independent)
    """
    def insert(self, char: str, position: int, user_id: str, timestamp: int):
        """Insert with globally unique ID"""
        unique_id = (user_id, timestamp, position)
        self.chars[unique_id] = char
    
    def render(self) -> str:
        """Sort by unique IDs → deterministic order"""
        sorted_chars = sorted(self.chars.items(), key=lambda x: x[0])
        return ''.join(c for _, c in sorted_chars)
```

**Trade-offs**:
- **OT**: Complex, requires central server, but efficient
- **CRDT**: Simpler, works offline, but more metadata

---

### Q5: "How to implement smart sync?"**

**Answer**: Prioritize based on context:

```python
def smart_sync(files: List[File], context: SyncContext) -> List[File]:
    """
    Prioritize syncs based on user context
    """
    priority_queue = []
    
    for file in files:
        priority = calculate_priority(file, context)
        heapq.heappush(priority_queue, (-priority, file))  # Max heap
    
    return [f for _, f in sorted(priority_queue)]

def calculate_priority(file: File, context: SyncContext) -> int:
    """
    Priority scoring
    """
    score = 0
    
    # Recently accessed → high priority
    if file.last_access_time > time.time() - 3600:  # Last hour
        score += 100
    
    # Currently open → highest priority
    if file.path in context.open_files:
        score += 1000
    
    # Small files → high priority (fast sync)
    if file.size < 1 * 1024 * 1024:  # <1 MB
        score += 50
    
    # Photos/Videos → low priority (large, can wait)
    if file.extension in ['.mp4', '.mov', '.jpg']:
        score -= 50
    
    # User explicitly requested → highest priority
    if file.path in context.user_requested:
        score += 10000
    
    return score
```

**Benefits**:
- **Responsiveness**: Sync open files first
- **Efficiency**: Batch small files, defer large files
- **User intent**: Respect explicit sync requests

---

### Q6: "Security considerations for file storage?"**

**Answer**: Multi-layer security:

**1. Encryption at rest**:
```python
def encrypt_chunk(chunk: bytes, key: bytes) -> bytes:
    """
    Encrypt chunk before storing
    
    Key management:
    - User password → KDF (PBKDF2) → Master key
    - Master key encrypts per-file keys
    - Per-file keys encrypt chunks
    """
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(chunk)
    return cipher.nonce + tag + ciphertext
```

**2. Encryption in transit**:
- TLS 1.3 for all API calls
- Certificate pinning on mobile clients

**3. Access control**:
```python
def check_permission(user_id: str, file_id: str, action: str) -> bool:
    """
    Check if user can perform action on file
    """
    file = get_file(file_id)
    
    # Owner always has access
    if file.owner == user_id:
        return True
    
    # Check shared permissions
    permissions = get_shared_permissions(file_id, user_id)
    
    if action == "read":
        return permissions.can_read
    elif action == "write":
        return permissions.can_write
    elif action == "share":
        return permissions.can_share
    
    return False
```

**4. Audit logging**:
- Log all file accesses
- Track share events
- Alert on suspicious activity (bulk downloads, unusual access patterns)

---

### Q7: "How to test file storage system?"**

**Answer**: Multiple test categories:

**1. Unit tests**:
```python
def test_chunk_deduplication():
    store = ChunkStore()
    chunk_id1 = store.put(b"hello")
    chunk_id2 = store.put(b"hello")  # Same content
    
    assert chunk_id1 == chunk_id2  # Same ID
    assert store._chunks[chunk_id1].ref_count == 2
    assert len(store._chunks) == 1  # Only stored once

def test_delta_sync():
    old_chunks = [b"A" * 64000, b"B" * 64000]
    new_chunks = [b"A" * 64000, b"C" * 64000]  # Changed second chunk
    
    old_sigs = DeltaSync.compute_signature(old_chunks)
    reused, new = DeltaSync.compute_delta(new_chunks, old_sigs)
    
    assert len(reused) == 1  # Reused first chunk
    assert len(new) == 1  # New second chunk
```

**2. Integration tests**:
```python
def test_file_sync_workflow():
    system = FileStorageSystem()
    
    # Upload v1
    v1 = system.upload_file("/doc.txt", b"version 1", "user1")
    
    # Upload v2 (simulate small change)
    v2, conflict = system.sync_file("/doc.txt", b"version 2", "user1", v1)
    
    assert not conflict  # No conflict
    assert system.download_file(file_id, v1) == b"version 1"  # v1 still accessible
    assert system.download_file(file_id, v2) == b"version 2"  # v2 is current
```

**3. Load tests**:
```python
def test_concurrent_uploads():
    system = FileStorageSystem()
    
    # Simulate 1000 users uploading simultaneously
    threads = []
    for i in range(1000):
        t = threading.Thread(
            target=system.upload_file,
            args=(f"/file{i}.txt", b"data" * 1000, f"user{i}")
        )
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Verify all uploads succeeded
    assert len(system._files) == 1000
```

**4. Chaos tests**:
```python
def test_network_partition():
    # Simulate client losing connection mid-upload
    system = FileStorageSystem()
    
    # Upload large file
    chunks = create_large_file(10 * 1024 * 1024)  # 10 MB
    
    # Simulate network failure after 50% uploaded
    with pytest.raises(NetworkError):
        system.upload_file_with_interruption(chunks, fail_at=0.5)
    
    # Resume upload
    system.resume_upload(chunks)
    
    # Verify file is complete
    downloaded = system.download_file(file_id)
    assert downloaded == chunks
```

---

## Testing Strategy

### Unit Tests
```python
def test_content_addressable_storage():
    chunk1 = Chunk.create(b"hello")
    chunk2 = Chunk.create(b"hello")
    assert chunk1.chunk_id == chunk2.chunk_id  # Same content → same ID

def test_reference_counting():
    store = ChunkStore()
    chunk_id = store.put(b"data")
    assert store._chunks[chunk_id].ref_count == 1
    
    store.put(b"data")  # Upload again
    assert store._chunks[chunk_id].ref_count == 2
    
    store.decrement_ref(chunk_id)
    assert store._chunks[chunk_id].ref_count == 1
    
    store.decrement_ref(chunk_id)
    assert chunk_id not in store._chunks  # Garbage collected

def test_version_history():
    metadata = FileMetadata(...)
    v1 = FileVersion.create(...)
    metadata.add_version(v1)
    
    v2 = FileVersion.create(..., parent_version=v1.version_id)
    metadata.add_version(v2)
    
    assert metadata.get_version(v1.version_id) == v1
    assert metadata.get_version(v2.version_id) == v2
```

---

## Production Considerations

### 1. Storage Backend
- **Metadata**: PostgreSQL with replication
- **Chunks**: S3/GCS with versioning enabled
- **Cache**: Redis for hot chunks
- **CDN**: CloudFront for global distribution

### 2. Monitoring Metrics
```python
metrics = {
    "dedup_ratio": 0.58,  # 58% of data deduplicated
    "avg_sync_time": 2.3,  # seconds
    "bandwidth_savings": 0.89,  # 89% saved vs full transfer
    "conflict_rate": 0.001,  # 0.1% of syncs have conflicts
    "storage_efficiency": 0.45  # 55% overhead (versions, metadata)
}
```

### 3. Scalability
- **Sharding**: Hash(file_id) % num_shards → distribute files
- **Replication**: 3x replication for durability
- **Caching**: LRU cache for hot files
- **CDN**: Serve chunks from edge locations

### 4. Cost Optimization
```python
# Delete old versions after 90 days
def cleanup_old_versions():
    for file in files:
        for version in file.versions:
            if version.age_days > 90:
                delete_version(version)
                # Decrement chunk ref counts
                for chunk_id in version.chunk_map.chunk_ids:
                    chunk_store.decrement_ref(chunk_id)
```

---

## Summary

### Do's ✅
- Use content-addressable storage (SHA-256 chunk IDs)
- Implement chunking for deduplication and delta sync
- Use delta sync (rsync algorithm) for bandwidth savings
- Keep immutable version snapshots for history
- Implement three-way merge for conflict resolution
- Monitor dedup ratio and sync efficiency

### Don'ts ❌
- Don't store entire files (chunk them!)
- Don't transfer full files on update (delta sync!)
- Don't delete chunks immediately (reference counting!)
- Don't use timestamps for conflict resolution only (data loss!)
- Don't skip encryption (security critical!)
- Don't forget to test network failures

### Key Takeaways
1. **Deduplication**: SHA-256 hashing enables 40-60% space savings
2. **Delta sync**: Rsync algorithm gives 90% bandwidth savings
3. **Chunking**: 64KB chunks balance dedup vs metadata overhead
4. **Versioning**: Immutable snapshots enable time-travel
5. **Conflicts**: Three-way merge when possible, LWW as fallback

This system demonstrates production-grade file storage used by Dropbox, Google Drive, and OneDrive.
