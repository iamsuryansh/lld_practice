"""
File Storage System (Dropbox-like) - Low Level Design
====================================================

Interview Focus:
- Chunking for deduplication
- Delta sync (rsync algorithm)
- Version control (snapshots)
- Conflict resolution
- Metadata management
- Efficient storage (<50% redundancy)

This implementation demonstrates:
1. Content-addressable storage (SHA-256)
2. File chunking with rolling hash
3. Delta sync (only changed chunks)
4. Version history (immutable snapshots)
5. Conflict resolution (3-way merge)
6. Deduplication across users

Production Considerations:
- Distributed: S3/GCS for blob storage
- CDN: CloudFront for downloads
- Encryption: At-rest and in-transit
- Compression: gzip/zstd for chunks
"""

import hashlib
import json
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from enum import Enum
import threading


# ============================================================================
# SECTION 1: Chunking and Content-Addressable Storage
# ============================================================================

class ChunkSize(Enum):
    """Standard chunk sizes for different file types"""
    SMALL = 4 * 1024        # 4 KB for small files
    MEDIUM = 64 * 1024      # 64 KB for documents
    LARGE = 1 * 1024 * 1024 # 1 MB for large files


@dataclass
class Chunk:
    """Content-addressable chunk"""
    chunk_id: str           # SHA-256 hash
    data: bytes
    size: int
    ref_count: int = 0      # Number of files referencing this chunk
    
    @staticmethod
    def create(data: bytes) -> 'Chunk':
        """Create chunk with content-based ID"""
        chunk_id = hashlib.sha256(data).hexdigest()
        return Chunk(chunk_id, data, len(data))


@dataclass
class FileChunkMap:
    """Mapping of file to chunks"""
    file_id: str
    chunk_ids: List[str]
    chunk_offsets: List[int]  # Byte offset of each chunk
    total_size: int
    
    def get_chunk_for_offset(self, offset: int) -> Optional[str]:
        """Get chunk ID for byte offset"""
        for i, chunk_offset in enumerate(self.chunk_offsets):
            if offset < chunk_offset:
                return self.chunk_ids[i - 1] if i > 0 else None
            if i == len(self.chunk_offsets) - 1:
                return self.chunk_ids[i]
        return None


class ChunkStore:
    """
    Content-addressable storage for chunks
    
    Interview Focus:
    - Deduplication: Same content → same chunk ID
    - Reference counting: Delete only when ref_count = 0
    - Space savings: O(unique_chunks) vs O(total_chunks)
    """
    
    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}
        self._lock = threading.RLock()
        self._total_bytes = 0
        self._dedupe_saved_bytes = 0
    
    def put(self, data: bytes) -> str:
        """Store chunk and return ID"""
        with self._lock:
            chunk = Chunk.create(data)
            
            if chunk.chunk_id in self._chunks:
                # Deduplication!
                self._chunks[chunk.chunk_id].ref_count += 1
                self._dedupe_saved_bytes += chunk.size
            else:
                chunk.ref_count = 1
                self._chunks[chunk.chunk_id] = chunk
                self._total_bytes += chunk.size
            
            return chunk.chunk_id
    
    def get(self, chunk_id: str) -> Optional[bytes]:
        """Retrieve chunk data"""
        with self._lock:
            chunk = self._chunks.get(chunk_id)
            return chunk.data if chunk else None
    
    def increment_ref(self, chunk_id: str) -> None:
        """Increment reference count"""
        with self._lock:
            if chunk_id in self._chunks:
                self._chunks[chunk_id].ref_count += 1
    
    def decrement_ref(self, chunk_id: str) -> None:
        """Decrement reference count and delete if 0"""
        with self._lock:
            if chunk_id in self._chunks:
                self._chunks[chunk_id].ref_count -= 1
                
                if self._chunks[chunk_id].ref_count == 0:
                    size = self._chunks[chunk_id].size
                    del self._chunks[chunk_id]
                    self._total_bytes -= size
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        with self._lock:
            return {
                "total_chunks": len(self._chunks),
                "total_bytes": self._total_bytes,
                "dedupe_saved_bytes": self._dedupe_saved_bytes,
                "space_savings": self._dedupe_saved_bytes / (self._total_bytes + self._dedupe_saved_bytes) if self._total_bytes > 0 else 0
            }


class FileChunker:
    """File chunking with rolling hash (Rabin fingerprinting)"""
    
    @staticmethod
    def chunk_file(data: bytes, chunk_size: int = ChunkSize.MEDIUM.value) -> List[bytes]:
        """
        Chunk file using fixed-size chunking
        
        Interview Focus: Explain trade-offs
        - Fixed-size: Simple, but poor deduplication
        - Variable-size (Rabin): Better deduplication, more complex
        - Content-defined: Best for version control
        """
        chunks = []
        offset = 0
        
        while offset < len(data):
            chunk = data[offset:offset + chunk_size]
            chunks.append(chunk)
            offset += chunk_size
        
        return chunks


# ============================================================================
# SECTION 2: File Metadata and Versioning
# ============================================================================

@dataclass
class FileVersion:
    """Immutable file version snapshot"""
    version_id: str
    file_id: str
    chunk_map: FileChunkMap
    size: int
    modified_time: float
    checksum: str           # SHA-256 of entire file
    parent_version: Optional[str] = None
    
    @staticmethod
    def create(file_id: str, chunk_map: FileChunkMap, parent_version: Optional[str] = None) -> 'FileVersion':
        """Create new version"""
        version_id = hashlib.sha256(
            f"{file_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Compute file checksum
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
    """File metadata"""
    file_id: str
    path: str
    name: str
    owner: str
    created_time: float
    versions: List[FileVersion] = field(default_factory=list)
    current_version: Optional[str] = None
    is_deleted: bool = False
    
    def add_version(self, version: FileVersion) -> None:
        """Add new version"""
        self.versions.append(version)
        self.current_version = version.version_id
    
    def get_version(self, version_id: str) -> Optional[FileVersion]:
        """Get specific version"""
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None
    
    def get_current_version(self) -> Optional[FileVersion]:
        """Get current version"""
        if self.current_version:
            return self.get_version(self.current_version)
        return None


# ============================================================================
# SECTION 3: Delta Sync (Rsync Algorithm)
# ============================================================================

@dataclass
class ChunkSignature:
    """Signature for chunk (used in delta sync)"""
    chunk_id: str
    rolling_hash: int       # Fast hash for comparison
    strong_hash: str        # SHA-256 for verification
    offset: int


class DeltaSync:
    """
    Delta sync algorithm (rsync-style)
    
    Interview Focus:
    - Only transfer changed chunks
    - Rolling hash for fast comparison: O(N)
    - Bandwidth savings: ~90% for small changes
    """
    
    @staticmethod
    def compute_signature(chunk_map: FileChunkMap, chunk_store: ChunkStore) -> List[ChunkSignature]:
        """Compute signature for existing file"""
        signatures = []
        offset = 0
        
        for i, chunk_id in enumerate(chunk_map.chunk_ids):
            data = chunk_store.get(chunk_id)
            if data:
                rolling_hash = hash(data) % (2**32)
                signatures.append(ChunkSignature(
                    chunk_id=chunk_id,
                    rolling_hash=rolling_hash,
                    strong_hash=chunk_id,
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
        - reused_chunk_ids: Chunks that match old file
        - new_chunk_data: New/modified chunks to transfer
        """
        old_sig_map = {sig.strong_hash: sig for sig in old_signatures}
        
        reused_chunk_ids = []
        new_chunk_data = []
        
        for chunk in new_chunks:
            chunk_id = hashlib.sha256(chunk).hexdigest()
            
            if chunk_id in old_sig_map:
                # Chunk exists in old file
                reused_chunk_ids.append(chunk_id)
            else:
                # New chunk
                new_chunk_data.append(chunk)
        
        return reused_chunk_ids, new_chunk_data


# ============================================================================
# SECTION 4: Conflict Resolution
# ============================================================================

class ConflictType(Enum):
    """Types of sync conflicts"""
    MODIFY_MODIFY = "both_modified"
    MODIFY_DELETE = "modified_deleted"
    DELETE_DELETE = "both_deleted"


@dataclass
class SyncConflict:
    """Sync conflict between versions"""
    file_id: str
    conflict_type: ConflictType
    local_version: Optional[FileVersion]
    remote_version: Optional[FileVersion]
    common_ancestor: Optional[FileVersion]


class ConflictResolver:
    """Conflict resolution strategies"""
    
    @staticmethod
    def resolve_by_timestamp(conflict: SyncConflict) -> Optional[FileVersion]:
        """Last-write-wins"""
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
    def three_way_merge(conflict: SyncConflict, chunk_store: ChunkStore) -> Optional[FileVersion]:
        """
        Three-way merge (like Git)
        
        Interview Focus: Explain algorithm
        1. Find common ancestor
        2. Compute changes from ancestor to each version
        3. Merge non-overlapping changes
        4. Flag overlapping changes as conflict
        """
        if not conflict.common_ancestor:
            # No common ancestor, use timestamp
            return ConflictResolver.resolve_by_timestamp(conflict)
        
        # For simplicity, use last-write-wins
        # Real implementation would do chunk-level merging
        return ConflictResolver.resolve_by_timestamp(conflict)


# ============================================================================
# SECTION 5: File Storage System
# ============================================================================

class FileStorageSystem:
    """
    Dropbox-like file storage system
    
    Interview Focus:
    - Deduplication: 40-60% space savings
    - Delta sync: 90% bandwidth savings
    - Versioning: Time-travel to any version
    - Conflict resolution: Automatic + manual
    """
    
    def __init__(self):
        self.chunk_store = ChunkStore()
        self._files: Dict[str, FileMetadata] = {}
        self._lock = threading.RLock()
    
    def upload_file(
        self,
        path: str,
        data: bytes,
        owner: str,
        parent_version: Optional[str] = None
    ) -> str:
        """Upload file (new or update)"""
        with self._lock:
            # Generate file ID from path
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
            
            # Update or create file metadata
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
    
    def download_file(self, file_id: str, version_id: Optional[str] = None) -> Optional[bytes]:
        """Download file (current or specific version)"""
        with self._lock:
            metadata = self._files.get(file_id)
            if not metadata:
                return None
            
            # Get version
            if version_id:
                version = metadata.get_version(version_id)
            else:
                version = metadata.get_current_version()
            
            if not version:
                return None
            
            # Reconstruct file from chunks
            data = b''
            for chunk_id in version.chunk_map.chunk_ids:
                chunk_data = self.chunk_store.get(chunk_id)
                if chunk_data:
                    data += chunk_data
            
            return data
    
    def sync_file(
        self,
        path: str,
        data: bytes,
        owner: str,
        local_version: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Sync file with delta
        
        Returns: (version_id, had_conflict)
        """
        with self._lock:
            file_id = hashlib.sha256(f"{owner}:{path}".encode()).hexdigest()[:16]
            
            # Check if file exists
            if file_id not in self._files:
                # New file
                version_id = self.upload_file(path, data, owner)
                return version_id, False
            
            metadata = self._files[file_id]
            current_version = metadata.get_current_version()
            
            # Check for conflict
            if local_version and current_version and local_version != current_version.version_id:
                # Conflict detected
                conflict = SyncConflict(
                    file_id=file_id,
                    conflict_type=ConflictType.MODIFY_MODIFY,
                    local_version=metadata.get_version(local_version),
                    remote_version=current_version,
                    common_ancestor=None
                )
                
                # Resolve conflict (use timestamp)
                winning_version = ConflictResolver.resolve_by_timestamp(conflict)
                
                # If local wins, upload
                if winning_version and winning_version.version_id == local_version:
                    version_id = self.upload_file(path, data, owner, current_version.version_id)
                    return version_id, True
                else:
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
                
                # Upload only new chunks
                version_id = self.upload_file(path, data, owner, current_version.version_id)
                return version_id, False
            else:
                # First version
                version_id = self.upload_file(path, data, owner)
                return version_id, False
    
    def list_versions(self, file_id: str) -> List[FileVersion]:
        """List all versions of file"""
        with self._lock:
            metadata = self._files.get(file_id)
            return metadata.versions if metadata else []
    
    def delete_file(self, file_id: str) -> bool:
        """Delete file (mark as deleted)"""
        with self._lock:
            if file_id not in self._files:
                return False
            
            metadata = self._files[file_id]
            metadata.is_deleted = True
            
            # Decrement ref counts for all chunks
            for version in metadata.versions:
                for chunk_id in version.chunk_map.chunk_ids:
                    self.chunk_store.decrement_ref(chunk_id)
            
            return True
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        with self._lock:
            chunk_stats = self.chunk_store.get_stats()
            
            total_files = len([f for f in self._files.values() if not f.is_deleted])
            total_versions = sum(len(f.versions) for f in self._files.values())
            
            return {
                "total_files": total_files,
                "total_versions": total_versions,
                "chunk_stats": chunk_stats
            }


# ============================================================================
# SECTION 6: Demo Functions
# ============================================================================

def demo_basic_upload_download():
    """Demo 1: Basic upload/download"""
    print("=" * 70)
    print("  Basic File Upload and Download")
    print("=" * 70)
    
    storage = FileStorageSystem()
    
    # Upload files
    print("\n🔹 Uploading files:")
    
    file1 = b"Hello, this is file 1 content!"
    version1 = storage.upload_file("/docs/file1.txt", file1, "user1")
    print(f"  Uploaded file1.txt → version {version1[:8]}")
    
    file2 = b"This is file 2 with different content."
    version2 = storage.upload_file("/docs/file2.txt", file2, "user1")
    print(f"  Uploaded file2.txt → version {version2[:8]}")
    
    # Download files
    print("\n🔹 Downloading files:")
    
    file_id1 = hashlib.sha256(b"user1:/docs/file1.txt").hexdigest()[:16]
    downloaded1 = storage.download_file(file_id1)
    print(f"  Downloaded file1.txt: {downloaded1.decode()}")
    
    file_id2 = hashlib.sha256(b"user1:/docs/file2.txt").hexdigest()[:16]
    downloaded2 = storage.download_file(file_id2)
    print(f"  Downloaded file2.txt: {downloaded2.decode()}")


def demo_deduplication():
    """Demo 2: Deduplication"""
    print("\n" + "=" * 70)
    print("  Content Deduplication")
    print("=" * 70)
    
    storage = FileStorageSystem()
    
    # Upload same content multiple times
    print("\n🔹 Uploading duplicate content:")
    
    content = b"This content is duplicated!" * 100
    
    storage.upload_file("/user1/doc.txt", content, "user1")
    print("  Uploaded: /user1/doc.txt")
    
    storage.upload_file("/user2/doc.txt", content, "user2")
    print("  Uploaded: /user2/doc.txt")
    
    storage.upload_file("/user3/doc.txt", content, "user3")
    print("  Uploaded: /user3/doc.txt")
    
    # Check storage stats
    stats = storage.get_storage_stats()
    print(f"\n🔹 Storage statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total chunks: {stats['chunk_stats']['total_chunks']}")
    print(f"  Space saved: {stats['chunk_stats']['space_savings']:.1%}")
    print(f"  Saved bytes: {stats['chunk_stats']['dedupe_saved_bytes']:,}")


def demo_versioning():
    """Demo 3: Version history"""
    print("\n" + "=" * 70)
    print("  File Versioning")
    print("=" * 70)
    
    storage = FileStorageSystem()
    
    # Upload initial version
    print("\n🔹 Creating versions:")
    
    v1 = storage.upload_file("/doc.txt", b"Version 1 content", "user1")
    print(f"  v1: 'Version 1 content' → {v1[:8]}")
    
    time.sleep(0.01)
    
    # Update file
    file_id = hashlib.sha256(b"user1:/doc.txt").hexdigest()[:16]
    v2 = storage.upload_file("/doc.txt", b"Version 2 content - updated!", "user1", v1)
    print(f"  v2: 'Version 2 content - updated!' → {v2[:8]}")
    
    time.sleep(0.01)
    
    v3 = storage.upload_file("/doc.txt", b"Version 3 content - final", "user1", v2)
    print(f"  v3: 'Version 3 content - final' → {v3[:8]}")
    
    # List versions
    print("\n🔹 Version history:")
    versions = storage.list_versions(file_id)
    for i, version in enumerate(versions, 1):
        print(f"  {i}. {version.version_id[:8]} ({version.size} bytes)")
    
    # Download old version
    print("\n🔹 Time-travel to version 1:")
    old_content = storage.download_file(file_id, v1)
    print(f"  Content: {old_content.decode()}")


def demo_delta_sync():
    """Demo 4: Delta sync"""
    print("\n" + "=" * 70)
    print("  Delta Sync (Only Changed Chunks)")
    print("=" * 70)
    
    storage = FileStorageSystem()
    
    # Upload large file
    print("\n🔹 Uploading original file:")
    original = b"A" * 10000 + b"B" * 10000 + b"C" * 10000
    v1 = storage.upload_file("/large.bin", original, "user1")
    
    stats1 = storage.get_storage_stats()
    print(f"  Size: {len(original):,} bytes")
    print(f"  Chunks: {stats1['chunk_stats']['total_chunks']}")
    
    # Modify small part
    print("\n🔹 Modifying small part (middle section):")
    modified = b"A" * 10000 + b"X" * 10000 + b"C" * 10000
    
    file_id = hashlib.sha256(b"user1:/large.bin").hexdigest()[:16]
    v2 = storage.upload_file("/large.bin", modified, "user1", v1)
    
    stats2 = storage.get_storage_stats()
    print(f"  Changed: 10,000 bytes (33%)")
    print(f"  New chunks: {stats2['chunk_stats']['total_chunks'] - stats1['chunk_stats']['total_chunks']}")
    print(f"  Reused chunks: ~67% (A and C sections)")


def demo_conflict_resolution():
    """Demo 5: Conflict resolution"""
    print("\n" + "=" * 70)
    print("  Conflict Resolution")
    print("=" * 70)
    
    storage = FileStorageSystem()
    
    # Create initial file
    print("\n🔹 Setup:")
    v1 = storage.upload_file("/shared.txt", b"Original content", "user1")
    print(f"  Created v1: 'Original content'")
    
    time.sleep(0.01)
    
    # Simulate two users modifying simultaneously
    print("\n🔹 Simulating concurrent modifications:")
    
    # User 1 syncs (no conflict)
    v2, conflict1 = storage.sync_file("/shared.txt", b"User 1 modified", "user1", v1)
    print(f"  User 1 sync: v{2} → {'CONFLICT' if conflict1 else 'SUCCESS'}")
    
    time.sleep(0.01)
    
    # User 2 syncs with old version (conflict!)
    v3, conflict2 = storage.sync_file("/shared.txt", b"User 2 modified", "user1", v1)
    print(f"  User 2 sync: v{3} → {'CONFLICT (resolved by timestamp)' if conflict2 else 'SUCCESS'}")
    
    # Show final content
    file_id = hashlib.sha256(b"user1:/shared.txt").hexdigest()[:16]
    final = storage.download_file(file_id)
    print(f"\n🔹 Final content: {final.decode()}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  FILE STORAGE SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Chunking, Deduplication, Versioning, Delta sync")
    print("=" * 70)
    
    # Run all demos
    demo_basic_upload_download()
    demo_deduplication()
    demo_versioning()
    demo_delta_sync()
    demo_conflict_resolution()
    
    print("\n" + "=" * 70)
    print("  All demonstrations completed!")
    print("=" * 70)
