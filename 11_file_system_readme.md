# File System - Interview Preparation Guide

**Target Audience**: Software Engineers with 2-5 years of experience  
**Focus**: Tree data structures, path resolution, permissions, hierarchical organization  
**Estimated Study Time**: 3-4 hours

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
3. [Critical Knowledge Points](#critical-knowledge-points)
4. [Expected Interview Questions & Answers](#expected-interview-questions--answers)
5. [Testing Strategy](#testing-strategy)
6. [Production Considerations](#production-considerations)

---

## Problem Statement

Design a file system that can:
- Support hierarchical directory structure (tree-based)
- Perform file operations (create, delete, rename, move, copy)
- Implement Unix-style permissions (read, write, execute)
- Resolve absolute and relative paths efficiently
- Traverse directory tree (DFS and BFS)
- Search for files by name, extension, or other criteria
- Track metadata (size, timestamps, owner, permissions)

**Core Challenge**: How do you design a file system that efficiently manages hierarchical data, handles path resolution, and enforces permissions while maintaining integrity?

---

## Step-by-Step Implementation Guide

### Phase 1: Tree Node Structure with Composite Pattern (20-25 minutes)

**What to do**:
```python
class FileSystemNode(ABC):
    def __init__(self, metadata, parent):
        self.metadata = metadata
        self.parent = parent
    
    @abstractmethod
    def get_size(self) -> int:
        pass
    
    @abstractmethod
    def is_directory(self) -> bool:
        pass

class File(FileSystemNode):
    def __init__(self, metadata, parent, content=""):
        super().__init__(metadata, parent)
        self._content = content
    
    def get_size(self):
        return len(self._content)

class Directory(FileSystemNode):
    def __init__(self, metadata, parent):
        super().__init__(metadata, parent)
        self.children: Dict[str, FileSystemNode] = {}
    
    def get_size(self):
        return sum(child.get_size() for child in self.children.values())
```

**Why Composite Pattern**:
- **Uniform interface**: Files and directories share common operations
- **Recursive composition**: Directories can contain files and other directories
- **Tree traversal**: Easy to traverse entire tree with single interface
- **Flexibility**: Easy to add new node types (symlinks, etc.)

**Key Insight**: The parent pointer enables path construction by traversing up to root, which is O(depth) time complexity.

**Common mistake**: Using separate data structures for files and directories makes tree operations complex.

---

### Phase 2: Permission System with Bit Flags (15-20 minutes)

**What to do**:
```python
class Permission(IntFlag):
    NONE = 0
    EXECUTE = 1    # 001
    WRITE = 2      # 010
    READ = 4       # 100
    
    # Combinations
    READ_WRITE = READ | WRITE
    ALL = READ | WRITE | EXECUTE

class FileMetadata:
    def __init__(self, name, type, owner):
        self.owner_permissions = Permission.ALL
        self.group_permissions = Permission.READ_EXECUTE
        self.other_permissions = Permission.READ
    
def has_permission(self, user, permission):
    if user == self.metadata.owner:
        return bool(self.metadata.owner_permissions & permission)
    # ... check group and others
```

**Why Bit Flags**:
- **Memory efficient**: 3 bits instead of 3 boolean fields
- **Fast operations**: Bitwise AND/OR operations are O(1)
- **Standard Unix model**: rwx permissions familiar to developers
- **Easy combinations**: Can check multiple permissions at once

**Interview Tip**: Explain that Unix uses octal notation (755 = rwxr-xr-x) where each digit represents owner/group/others permissions.

---

### Phase 3: Path Resolution Algorithm (20-25 minutes)

**What to do**:
```python
class PathResolver:
    @staticmethod
    def normalize_path(path: str) -> str:
        """Resolve . and .. in path"""
        components = path.split("/")
        stack = []
        
        for component in components:
            if component == "" or component == ".":
                continue
            elif component == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(component)
        
        return "/" + "/".join(stack) if stack else "/"
    
    def resolve_path(self, path: str) -> Optional[FileSystemNode]:
        """Navigate from root to target node"""
        normalized = PathResolver.normalize_path(path)
        
        # Check cache
        if normalized in self.path_cache:
            return self.path_cache[normalized]
        
        # Traverse tree
        components = [c for c in normalized.split("/") if c]
        current = self.root
        
        for component in components:
            if not current.is_directory():
                return None
            current = current.get_child(component)
            if current is None:
                return None
        
        # Cache result
        self.path_cache[normalized] = current
        return current
```

**Why Stack for Normalization**:
- **Handles .. elegantly**: Just pop from stack
- **Handles .** Just skip
- **Time**: O(n) where n is path length
- **Space**: O(d) where d is path depth

**Critical Detail**: Path cache dramatically improves performance from O(depth) to O(1), but requires invalidation on any structural change.

**When it fails**: Symlinks create additional complexity - may need to track visited nodes to detect cycles.

---

### Phase 4: File Operations with Permission Checking (25-30 minutes)

**What to do**:
```python
def create_file(self, path: str, user: str, content: str = ""):
    # 1. Parse path into parent directory and filename
    parent_path, filename = PathResolver.split_path(path)
    
    # 2. Resolve parent directory
    parent_node = self.resolve_path(parent_path)
    if not parent_node or not parent_node.is_directory():
        return False, "Parent directory not found"
    
    # 3. Check write permission on parent
    if not parent_node.has_permission(user, Permission.WRITE):
        return False, "Permission denied"
    
    # 4. Check if file already exists
    if parent_node.get_child(filename):
        return False, "File already exists"
    
    # 5. Create file node and add to parent
    file_node = File(FileMetadata(filename, FileType.FILE, user), 
                     parent_node, content)
    parent_node.add_child(file_node)
    
    # 6. Invalidate path cache
    self._invalidate_path_cache(parent_path)
    
    return True, "File created"
```

**Error Recovery Strategy**:
- Always check permissions before modifications
- Validate path existence before operations
- Check for name conflicts
- Invalidate cache after structural changes

**Interview Insight**: Every file operation follows this pattern: resolve → validate → check permission → modify → invalidate cache.

---

### Phase 5: Tree Traversal Algorithms (15-20 minutes)

**What to do**:
```python
# Depth-First Search (Recursive)
def traverse_dfs(self, path: str, user: str) -> List[str]:
    node = self.resolve_path(path)
    result = []
    self._dfs_helper(node, user, result)
    return result

def _dfs_helper(self, node, user, result):
    result.append(node.get_path())
    
    if node.is_directory():
        if node.has_permission(user, Permission.READ | Permission.EXECUTE):
            for child in sorted(node.children.values(), key=lambda n: n.get_name()):
                self._dfs_helper(child, user, result)

# Breadth-First Search (Iterative)
def traverse_bfs(self, path: str, user: str) -> List[str]:
    node = self.resolve_path(path)
    result = []
    queue = deque([node])
    
    while queue:
        current = queue.popleft()
        result.append(current.get_path())
        
        if current.is_directory():
            if current.has_permission(user, Permission.READ | Permission.EXECUTE):
                for child in sorted(current.children.values(), key=lambda n: n.get_name()):
                    queue.append(child)
    
    return result
```

**DFS vs BFS**:
- **DFS**: Better for finding files deep in tree, uses recursion (O(h) space)
- **BFS**: Better for finding files in shallow levels, uses queue (O(w) space where w=width)
- **Both**: O(n) time to visit all n nodes

**When to use each**:
- DFS: `find` command, directory size calculation
- BFS: Listing files by level, finding nearby files

---

### Phase 6: Move Operation with Cycle Detection (20-25 minutes)

**What to do**:
```python
def move(self, src_path: str, dest_path: str, user: str):
    # 1. Resolve source and destination
    src_node = self.resolve_path(src_path)
    dest_node = self.resolve_path(dest_path)
    
    # 2. Validate
    if not src_node or not dest_node or not dest_node.is_directory():
        return False, "Invalid source or destination"
    
    # 3. Check permissions
    if not src_node.parent.has_permission(user, Permission.WRITE):
        return False, "Cannot remove from source"
    if not dest_node.has_permission(user, Permission.WRITE):
        return False, "Cannot add to destination"
    
    # 4. Cycle detection - prevent moving dir into its own subtree
    if src_node.is_directory() and self._is_ancestor(src_node, dest_node):
        return False, "Cannot move directory to its own descendant"
    
    # 5. Check name conflict
    if dest_node.get_child(src_node.get_name()):
        return False, "Name already exists in destination"
    
    # 6. Perform move
    src_node.parent.remove_child(src_node.get_name())
    dest_node.add_child(src_node)
    
    return True, "Moved successfully"

def _is_ancestor(self, potential_ancestor, node):
    """Check if potential_ancestor is ancestor of node"""
    current = node
    while current:
        if current == potential_ancestor:
            return True
        current = current.parent
    return False
```

**Why Cycle Detection Matters**:
- Without it: `mv /a /a/b` would create cycle: /a → /a/b → /a → ...
- Detection: Traverse up from destination; if we hit source, it's a cycle
- Time: O(depth)

---

## Critical Knowledge Points

### 1. Why Composite Pattern for File System?

**Without Composite Pattern**:
```python
class FileSystem:
    def get_size(self, path):
        if is_file(path):
            return file_sizes[path]
        else:
            # Need separate logic for directories
            total = 0
            for child in get_children(path):
                total += self.get_size(child)  # Recursive
            return total
```

**With Composite Pattern**:
```python
class FileSystemNode:
    @abstractmethod
    def get_size(self) -> int:
        pass

class File(FileSystemNode):
    def get_size(self):
        return len(self._content)

class Directory(FileSystemNode):
    def get_size(self):
        return sum(child.get_size() for child in self.children.values())

# Usage - same interface for files and directories
size = node.get_size()  # Works for both!
```

**Benefits**:
- **Uniform interface**: Treat files and directories the same
- **Recursive operations**: Natural recursion through tree
- **Extensibility**: Easy to add new node types (symlinks, devices)
- **Simplicity**: No type checking needed

---

### 2. Path Resolution Algorithm Explained

**Algorithm**:
```python
def normalize_path(path):
    # Input: "/a/./b/../c"
    # Step 1: Split by "/" → ["", "a", ".", "b", "..", "c"]
    
    stack = []
    for component in path.split("/"):
        if component in ["", "."]:
            continue  # Skip empty and current directory
        elif component == "..":
            if stack:
                stack.pop()  # Go up one level
        else:
            stack.append(component)  # Add to path
    
    # Step 2: Join → "a/c"
    return "/" + "/".join(stack)
```

**Time**: O(n) where n = path length  
**Space**: O(d) where d = depth (stack size)

**Why it works**: Stack naturally handles parent directory references by popping when seeing "..".

**Edge cases**:
- `/a/b/../../c` → `/c` (can go above starting point)
- `/a/b/../..` → `/` (going to root)
- `/../..` → `/` (can't go above root)

**Alternative: Tree-based resolution**:
```python
def resolve(path):
    current = root
    for component in path.split("/"):
        if component == "..":
            current = current.parent or current  # Stay at root if no parent
        elif component and component != ".":
            current = current.get_child(component)
    return current
```

**Trade-off**: Tree-based requires tree access, stack-based works on path string alone.

---

### 3. Permission System Implementation

**Bit Flag Representation**:
```python
# Binary representation
READ    = 0b100 = 4
WRITE   = 0b010 = 2
EXECUTE = 0b001 = 1

# Check if has READ permission
has_read = (permissions & READ) != 0

# Grant WRITE permission
permissions = permissions | WRITE

# Revoke EXECUTE permission
permissions = permissions & ~EXECUTE

# Check multiple permissions
has_read_write = (permissions & (READ | WRITE)) == (READ | WRITE)
```

**Why Bit Flags**:
- **Space efficient**: 1 integer instead of 3 booleans
- **Fast operations**: Bitwise operations are single CPU instructions
- **Atomic updates**: Single integer write is atomic

**Real Unix Permissions**:
```
-rwxr-xr--  1 alice users  4096 Nov  6 10:00 file.txt
 │││││││││
 ││││││││└─ others: read
 │││││││└── others: write (no)
 ││││││└─── others: execute (no)
 │││││└──── group: read
 ││││└───── group: write (no)
 │││└────── group: execute
 ││└─────── owner: read
 │└──────── owner: write
 └───────── owner: execute
```

**Octal notation**: `chmod 754` = `111 101 100` = `rwx r-x r--`

---

### 4. Cache Invalidation Strategy

**Problem**: After structural changes, cached paths may be stale.

**Solution**: Invalidate affected paths
```python
def _invalidate_path_cache(self, path: str):
    """Invalidate path and all descendants"""
    normalized = normalize_path(path)
    
    # Remove all paths that start with this path
    keys_to_remove = [k for k in self.path_cache.keys() 
                     if k.startswith(normalized)]
    
    for key in keys_to_remove:
        del self.path_cache[key]
```

**Operations requiring invalidation**:
- Create file/directory (invalidate parent)
- Delete (invalidate parent and subtree)
- Rename (invalidate old parent)
- Move (invalidate old parent and new parent)

**Trade-off**:
- **With cache**: O(1) path resolution, but must invalidate on changes
- **Without cache**: O(d) path resolution always, no invalidation needed

**When cache helps**: Read-heavy workloads (listing, searching)  
**When cache hurts**: Write-heavy workloads (many creates/deletes)

---

## Expected Interview Questions & Answers

### Q1: How would you implement symbolic links (symlinks)?

**Answer**:
Symbolic links are special file system nodes that point to another path. When accessed, they redirect to the target path.

**Implementation**:
```python
class Symlink(FileSystemNode):
    def __init__(self, metadata, parent, target_path):
        super().__init__(metadata, parent)
        self.target_path = target_path
        self.metadata.type = FileType.SYMLINK
    
    def get_target(self, file_system):
        """Resolve symlink to actual target"""
        return file_system.resolve_path(self.target_path)
    
    def is_directory(self):
        # Symlinks take type of their target
        return False  # Or resolve and check target

# Path resolution with symlinks
def resolve_path(self, path, visited=None):
    if visited is None:
        visited = set()
    
    node = self._resolve_without_symlinks(path)
    
    # Follow symlinks
    while isinstance(node, Symlink):
        # Detect cycles
        if node.id in visited:
            raise Exception("Symlink cycle detected")
        visited.add(node.id)
        
        # Resolve target
        node = node.get_target(self)
    
    return node
```

**Key Challenges**:
1. **Cycle detection**: Symlink A → B → A creates infinite loop
2. **Broken links**: Target may not exist
3. **Permissions**: Check permissions on symlink AND target
4. **Relative vs absolute**: Symlink target can be relative to symlink location

**Follow-up**: Hardlinks are different - they reference the same inode, not a path. Requires inode table to track multiple directory entries pointing to same data.

---

### Q2: How do you calculate directory size efficiently?

**Answer**:
Directory size is the sum of all file sizes in its subtree. There are several approaches with different trade-offs:

**Approach 1: On-demand calculation (current implementation)**:
```python
def get_size(self):
    total = 0
    for child in self.children.values():
        total += child.get_size()  # Recursive
    return total
```

**Pros**: Always accurate, no extra space  
**Cons**: O(n) time for n files in subtree - slow for large directories

**Approach 2: Cached size**:
```python
class Directory:
    def __init__(self, metadata, parent):
        super().__init__(metadata, parent)
        self.children = {}
        self._cached_size = 0  # Cache
    
    def add_child(self, child):
        self.children[child.get_name()] = child
        self._cached_size += child.get_size()
        
        # Update all ancestors
        current = self.parent
        while current:
            current._cached_size += child.get_size()
            current = current.parent
    
    def get_size(self):
        return self._cached_size  # O(1)!
```

**Pros**: O(1) size retrieval  
**Cons**: Must update all ancestors on every change - O(depth) cost

**Approach 3: Lazy evaluation with dirty flag**:
```python
class Directory:
    def __init__(self, metadata, parent):
        self._cached_size = 0
        self._size_dirty = True  # Flag
    
    def add_child(self, child):
        self.children[child.get_name()] = child
        self._mark_size_dirty()  # O(depth)
    
    def _mark_size_dirty(self):
        current = self
        while current and not current._size_dirty:
            current._size_dirty = True
            current = current.parent
    
    def get_size(self):
        if self._size_dirty:
            self._cached_size = sum(c.get_size() for c in self.children.values())
            self._size_dirty = False
        return self._cached_size
```

**Pros**: Fast when size accessed multiple times, efficient updates  
**Cons**: More complex, first access after change is still O(n)

**Best approach**: Depends on workload
- Read-heavy: Use cached size (approach 2 or 3)
- Write-heavy: Use on-demand (approach 1)
- Mixed: Use lazy evaluation (approach 3)

---

### Q3: How would you implement `du` command (disk usage)?

**Answer**:
The `du` command shows disk usage of files and directories. It needs to:
1. Traverse directory tree
2. Calculate sizes
3. Optionally show individual files or just summary

**Implementation**:
```python
def disk_usage(self, path: str, user: str, show_files: bool = False) -> Dict[str, int]:
    """
    Calculate disk usage for path and its children
    
    Args:
        path: Starting path
        user: User running command (for permissions)
        show_files: If True, show individual file sizes
    
    Returns:
        Dict mapping paths to sizes
    """
    node = self.resolve_path(path)
    if not node:
        return {}
    
    # Check permission
    if not node.has_permission(user, Permission.READ | Permission.EXECUTE):
        return {}
    
    result = {}
    
    def calculate(node):
        # Get size
        size = node.get_size()
        
        # Record if requested
        if show_files or node.is_directory():
            result[node.get_path()] = size
        
        # Recurse for directories
        if node.is_directory():
            for child in node.children.values():
                if child.has_permission(user, Permission.READ):
                    calculate(child)
    
    calculate(node)
    return result

# Usage
fs = FileSystem()
# Create some files...

usage = fs.disk_usage("/home", "alice", show_files=True)
for path, size in sorted(usage.items(), key=lambda x: x[1], reverse=True):
    print(f"{size:>10} bytes  {path}")

# Output:
#      15000 bytes  /home
#       8000 bytes  /home/alice
#       5000 bytes  /home/alice/documents
#       3000 bytes  /home/alice/documents/report.txt
#       2000 bytes  /home/alice/readme.txt
#       7000 bytes  /home/bob
```

**Optimizations**:
1. **Parallel traversal**: Use thread pool to traverse subdirectories concurrently
2. **Caching**: Cache sizes if tree doesn't change
3. **Pruning**: Skip directories user doesn't have permission for

**Follow-up**: How to handle hardlinks? Need inode tracking to avoid counting same file multiple times.

---

### Q4: How would you implement a file watcher that notifies on changes?

**Answer**:
A file watcher monitors the file system and triggers callbacks when files/directories change. This uses the Observer pattern.

**Implementation**:
```python
class FileSystemEvent(Enum):
    CREATED = "created"
    DELETED = "deleted"
    MODIFIED = "modified"
    MOVED = "moved"

class FileSystemObserver(ABC):
    @abstractmethod
    def on_event(self, event_type: FileSystemEvent, path: str, **kwargs):
        pass

class FileSystem:
    def __init__(self):
        # ... existing code ...
        self.observers: List[FileSystemObserver] = []
    
    def add_observer(self, observer: FileSystemObserver):
        self.observers.append(observer)
    
    def _notify_observers(self, event_type: FileSystemEvent, path: str, **kwargs):
        for observer in self.observers:
            observer.on_event(event_type, path, **kwargs)
    
    def create_file(self, path: str, user: str, content: str = ""):
        # ... existing logic ...
        
        # Notify observers
        self._notify_observers(FileSystemEvent.CREATED, path)
        
        return True, "File created"
    
    def delete(self, path: str, user: str, recursive: bool = False):
        # ... existing logic ...
        
        # Notify observers
        self._notify_observers(FileSystemEvent.DELETED, path)
        
        return True, "Deleted"
    
    def write_file(self, path: str, user: str, content: str):
        # ... existing logic ...
        
        # Notify observers
        self._notify_observers(FileSystemEvent.MODIFIED, path)
        
        return True, "Written"

# Example observer - logging
class LoggingObserver(FileSystemObserver):
    def on_event(self, event_type, path, **kwargs):
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] {event_type.value}: {path}")

# Example observer - indexing for search
class SearchIndexObserver(FileSystemObserver):
    def __init__(self):
        self.index = {}  # filename -> [paths]
    
    def on_event(self, event_type, path, **kwargs):
        filename = path.split("/")[-1]
        
        if event_type == FileSystemEvent.CREATED:
            if filename not in self.index:
                self.index[filename] = []
            self.index[filename].append(path)
        
        elif event_type == FileSystemEvent.DELETED:
            if filename in self.index:
                self.index[filename].remove(path)
                if not self.index[filename]:
                    del self.index[filename]
    
    def search(self, filename: str) -> List[str]:
        return self.index.get(filename, [])

# Usage
fs = FileSystem()
fs.add_observer(LoggingObserver())

search_index = SearchIndexObserver()
fs.add_observer(search_index)

fs.create_file("/test.txt", "alice", "content")
# Output: [2024-11-06T10:00:00] created: /test.txt

results = search_index.search("test.txt")
# Returns: ["/test.txt"]
```

**Use cases**:
- **Build systems**: Recompile when source files change
- **IDEs**: Refresh project tree when files added/removed
- **Cloud sync**: Upload changed files to cloud
- **Search indexing**: Keep search index up-to-date

---

### Q5: How do you handle concurrent access to the file system?

**Answer**:
Concurrent access requires careful synchronization to prevent race conditions. There are several strategies:

**Strategy 1: Coarse-grained locking (current implementation)**:
```python
class FileSystem:
    def __init__(self):
        self.lock = RLock()  # Single lock for entire file system
    
    def create_file(self, path, user, content):
        with self.lock:  # Lock entire operation
            # ... implementation ...
```

**Pros**: Simple, prevents all race conditions  
**Cons**: Poor concurrency - only one operation at a time

**Strategy 2: Fine-grained locking**:
```python
class FileSystemNode:
    def __init__(self, metadata, parent):
        self.metadata = metadata
        self.parent = parent
        self.lock = RLock()  # Per-node lock

class FileSystem:
    def create_file(self, path, user, content):
        parent_path, filename = split_path(path)
        parent = self.resolve_path(parent_path)
        
        with parent.lock:  # Lock only parent directory
            # ... create file ...
```

**Pros**: Better concurrency - operations on different directories don't block  
**Cons**: Complex, risk of deadlocks with multiple locks

**Strategy 3: Read-write locks**:
```python
from threading import Lock, Condition

class ReadWriteLock:
    def __init__(self):
        self.readers = 0
        self.writers = 0
        self.lock = Lock()
        self.can_read = Condition(self.lock)
        self.can_write = Condition(self.lock)
    
    def acquire_read(self):
        with self.lock:
            while self.writers > 0:
                self.can_read.wait()
            self.readers += 1
    
    def release_read(self):
        with self.lock:
            self.readers -= 1
            if self.readers == 0:
                self.can_write.notify()
    
    def acquire_write(self):
        with self.lock:
            while self.readers > 0 or self.writers > 0:
                self.can_write.wait()
            self.writers += 1
    
    def release_write(self):
        with self.lock:
            self.writers -= 1
            self.can_write.notify()
            self.can_read.notify_all()

class FileSystem:
    def __init__(self):
        self.rw_lock = ReadWriteLock()
    
    def read_file(self, path, user):
        self.rw_lock.acquire_read()
        try:
            # ... read operation ...
        finally:
            self.rw_lock.release_read()
    
    def create_file(self, path, user, content):
        self.rw_lock.acquire_write()
        try:
            # ... write operation ...
        finally:
            self.rw_lock.release_write()
```

**Pros**: Multiple readers can run concurrently  
**Cons**: Writers still block all operations

**Best approach for production**:
- Use fine-grained locking with careful lock ordering to prevent deadlocks
- Lock parent directory when modifying children
- Use read-write locks for file content
- Consider lock-free data structures for metadata

---

### Q6: How would you implement file system quotas?

**Answer**:
Quotas limit disk space usage per user or directory. Need to track usage and enforce limits.

**Implementation**:
```python
@dataclass
class Quota:
    max_size: int  # bytes
    max_files: int
    used_size: int = 0
    used_files: int = 0
    
    def can_add_file(self, size: int) -> bool:
        return (self.used_size + size <= self.max_size and
                self.used_files + 1 <= self.max_files)
    
    def add_file(self, size: int):
        self.used_size += size
        self.used_files += 1
    
    def remove_file(self, size: int):
        self.used_size -= size
        self.used_files -= 1

class FileSystem:
    def __init__(self):
        # ... existing code ...
        self.user_quotas: Dict[str, Quota] = {}
    
    def set_quota(self, user: str, max_size: int, max_files: int):
        """Set quota for user"""
        self.user_quotas[user] = Quota(max_size, max_files)
    
    def create_file(self, path: str, user: str, content: str = ""):
        with self.lock:
            # Check quota
            if user in self.user_quotas:
                quota = self.user_quotas[user]
                file_size = len(content.encode('utf-8'))
                
                if not quota.can_add_file(file_size):
                    return False, "Quota exceeded"
            
            # ... existing create file logic ...
            
            # Update quota
            if user in self.user_quotas:
                self.user_quotas[user].add_file(file_size)
            
            return True, "File created"
    
    def delete(self, path: str, user: str, recursive: bool = False):
        with self.lock:
            node = self.resolve_path(path)
            
            # ... existing delete logic ...
            
            # Update quota
            if user in self.user_quotas:
                size = node.get_size()
                file_count = self._count_files(node)
                self.user_quotas[user].used_size -= size
                self.user_quotas[user].used_files -= file_count
    
    def _count_files(self, node: FileSystemNode) -> int:
        """Count number of files in subtree"""
        if not node.is_directory():
            return 1
        
        count = 0
        for child in node.children.values():
            count += self._count_files(child)
        return count
    
    def get_quota_usage(self, user: str) -> Dict[str, Any]:
        """Get quota usage for user"""
        if user not in self.user_quotas:
            return {}
        
        quota = self.user_quotas[user]
        return {
            "used_size": quota.used_size,
            "max_size": quota.max_size,
            "used_files": quota.used_files,
            "max_files": quota.max_files,
            "size_percentage": (quota.used_size / quota.max_size) * 100,
            "files_percentage": (quota.used_files / quota.max_files) * 100
        }

# Usage
fs = FileSystem()
fs.set_quota("alice", max_size=1_000_000, max_files=100)  # 1 MB, 100 files

# Alice creates files
fs.create_file("/home/alice/file1.txt", "alice", "x" * 500_000)  # 500 KB - OK
fs.create_file("/home/alice/file2.txt", "alice", "x" * 600_000)  # 600 KB - FAIL (exceeds 1 MB)

# Check usage
usage = fs.get_quota_usage("alice")
print(f"Used: {usage['used_size']} / {usage['max_size']} bytes ({usage['size_percentage']:.1f}%)")
```

**Additional considerations**:
1. **Directory quotas**: Limit size of specific directories
2. **Grace periods**: Allow temporary quota violations with warnings
3. **Hard vs soft limits**: Soft limit warns, hard limit blocks
4. **Efficient tracking**: Use cached sizes to avoid recalculating on every quota check

---

### Q7: How would you implement a `find` command?

**Answer**:
The `find` command searches for files matching criteria (name, size, type, modification time, etc.). Requires flexible filtering during traversal.

**Implementation**:
```python
from typing import Callable

class SearchCriteria:
    """Fluent interface for building search criteria"""
    
    def __init__(self):
        self.filters: List[Callable[[FileSystemNode], bool]] = []
    
    def name_matches(self, pattern: str):
        """Match filename pattern (supports wildcards)"""
        import re
        regex = pattern.replace("*", ".*").replace("?", ".")
        self.filters.append(lambda node: re.match(regex, node.get_name()))
        return self
    
    def type_is(self, file_type: FileType):
        """Match file type"""
        self.filters.append(lambda node: node.metadata.type == file_type)
        return self
    
    def size_greater_than(self, size: int):
        """Match files larger than size"""
        self.filters.append(lambda node: node.get_size() > size)
        return self
    
    def modified_after(self, timestamp: datetime):
        """Match files modified after timestamp"""
        self.filters.append(lambda node: node.metadata.modified_at > timestamp)
        return self
    
    def matches(self, node: FileSystemNode) -> bool:
        """Check if node matches all criteria"""
        return all(f(node) for f in self.filters)

class FileSystem:
    def find(self, start_path: str, criteria: SearchCriteria, 
             user: str) -> List[str]:
        """
        Find files matching criteria
        
        Similar to Unix: find /path -name "*.txt" -type f -size +1M
        """
        with self.lock:
            start_node = self.resolve_path(start_path)
            if not start_node:
                return []
            
            results = []
            self._find_helper(start_node, criteria, user, results)
            return results
    
    def _find_helper(self, node: FileSystemNode, criteria: SearchCriteria,
                    user: str, results: List[str]):
        """Recursive find helper"""
        # Check permissions
        if not node.has_permission(user, Permission.READ):
            return
        
        # Check if matches criteria
        if criteria.matches(node):
            results.append(node.get_path())
        
        # Recurse into directories
        if node.is_directory():
            directory = node  # Type: Directory
            if directory.has_permission(user, Permission.EXECUTE):
                for child in directory.children.values():
                    self._find_helper(child, criteria, user, results)

# Usage examples
fs = FileSystem()
# ... create files ...

# Find all .txt files
criteria = SearchCriteria().name_matches("*.txt").type_is(FileType.FILE)
results = fs.find("/home", criteria, "alice")
# Returns: ["/home/alice/readme.txt", "/home/alice/notes.txt", ...]

# Find large files (> 1 MB)
from datetime import datetime, timedelta
criteria = (SearchCriteria()
    .type_is(FileType.FILE)
    .size_greater_than(1_000_000))
results = fs.find("/", criteria, "root")

# Find recently modified files (last 24 hours)
yesterday = datetime.now() - timedelta(days=1)
criteria = SearchCriteria().modified_after(yesterday)
results = fs.find("/home", criteria, "alice")

# Complex query - large .log files modified in last week
week_ago = datetime.now() - timedelta(days=7)
criteria = (SearchCriteria()
    .name_matches("*.log")
    .size_greater_than(10_000_000)  # > 10 MB
    .modified_after(week_ago))
results = fs.find("/var/log", criteria, "root")
```

**Optimizations**:
1. **Early termination**: Stop searching if found enough results
2. **Parallel search**: Search subdirectories in parallel
3. **Index-based search**: Build index for common queries (by name, by extension)
4. **Pruning**: Skip directories based on criteria (e.g., if searching for files modified today, skip old directories)

---

## Testing Strategy

### Unit Tests

**Test path normalization**:
```python
def test_path_normalization():
    assert PathResolver.normalize_path("/a/./b") == "/a/b"
    assert PathResolver.normalize_path("/a/../b") == "/b"
    assert PathResolver.normalize_path("/a/b/../../c") == "/c"
    assert PathResolver.normalize_path("/../..") == "/"
    assert PathResolver.normalize_path("/a/b/../c/./d") == "/a/c/d"

def test_split_path():
    assert PathResolver.split_path("/a/b/c") == ("/a/b", "c")
    assert PathResolver.split_path("/a") == ("/", "a")
    assert PathResolver.split_path("/") == ("", "/")
```

**Test permissions**:
```python
def test_permission_checks():
    # Create file with specific permissions
    metadata = FileMetadata("test.txt", FileType.FILE, "alice")
    metadata.owner_permissions = Permission.READ_WRITE
    metadata.other_permissions = Permission.READ
    
    file = File(metadata, None, "content")
    
    # Owner can read and write
    assert file.has_permission("alice", Permission.READ) == True
    assert file.has_permission("alice", Permission.WRITE) == True
    
    # Others can only read
    assert file.has_permission("bob", Permission.READ) == True
    assert file.has_permission("bob", Permission.WRITE) == False
```

**Test node operations**:
```python
def test_directory_add_remove():
    dir_metadata = FileMetadata("dir", FileType.DIRECTORY, "root")
    directory = Directory(dir_metadata, None)
    
    file_metadata = FileMetadata("file.txt", FileType.FILE, "root")
    file = File(file_metadata, directory, "content")
    
    # Add child
    success, msg = directory.add_child(file)
    assert success == True
    assert "file.txt" in directory.children
    
    # Remove child
    success, msg = directory.remove_child("file.txt")
    assert success == True
    assert "file.txt" not in directory.children
```

---

### Integration Tests

**Test complete file operations**:
```python
def test_create_read_write_delete():
    fs = FileSystem()
    user = "alice"
    
    # Create directory
    fs.create_directory("/home", "root")
    fs.create_directory("/home/alice", user)
    
    # Create file
    success, msg = fs.create_file("/home/alice/test.txt", user, "initial")
    assert success == True
    
    # Read file
    success, content = fs.read_file("/home/alice/test.txt", user)
    assert success == True
    assert content == "initial"
    
    # Write file
    success, msg = fs.write_file("/home/alice/test.txt", user, "updated")
    assert success == True
    
    # Read again
    success, content = fs.read_file("/home/alice/test.txt", user)
    assert content == "updated"
    
    # Delete
    success, msg = fs.delete("/home/alice/test.txt", user)
    assert success == True
    
    # Verify deleted
    node = fs.resolve_path("/home/alice/test.txt")
    assert node is None
```

**Test move operations**:
```python
def test_move_prevents_cycles():
    fs = FileSystem()
    user = "root"
    
    # Create: /a/b/c
    fs.create_directory("/a", user)
    fs.create_directory("/a/b", user)
    fs.create_directory("/a/b/c", user)
    
    # Try to move /a to /a/b/c (would create cycle)
    success, msg = fs.move("/a", "/a/b/c", user)
    assert success == False
    assert "descendant" in msg.lower()
```

---

### Load Testing

**Test with large directory tree**:
```python
import time

def test_performance_large_tree():
    fs = FileSystem()
    user = "root"
    
    # Create 10,000 files in nested structure
    start = time.time()
    
    for i in range(100):
        dir_path = f"/dir{i}"
        fs.create_directory(dir_path, user)
        
        for j in range(100):
            file_path = f"{dir_path}/file{j}.txt"
            fs.create_file(file_path, user, f"content {i}-{j}")
    
    creation_time = time.time() - start
    print(f"Created 10,000 files in {creation_time:.2f}s")
    
    # Test path resolution
    start = time.time()
    for i in range(1000):
        fs.resolve_path(f"/dir{i % 100}/file{i % 100}.txt")
    resolution_time = time.time() - start
    print(f"Resolved 1,000 paths in {resolution_time:.2f}s")
    
    # Test traversal
    start = time.time()
    results = fs.traverse_dfs("/", user)
    traversal_time = time.time() - start
    print(f"Traversed {len(results)} nodes in {traversal_time:.2f}s")
```

---

## Production Considerations

### 1. Persistence

**Current implementation**: In-memory only  
**Production needs**: Persist to disk

```python
import pickle
import json

class PersistentFileSystem(FileSystem):
    def __init__(self, storage_path: str):
        super().__init__()
        self.storage_path = storage_path
        self._load()
    
    def _save(self):
        """Save file system tree to disk"""
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.root, f)
    
    def _load(self):
        """Load file system tree from disk"""
        try:
            with open(self.storage_path, 'rb') as f:
                self.root = pickle.load(f)
                self._rebuild_cache()
        except FileNotFoundError:
            pass  # Use default root
    
    def _rebuild_cache(self):
        """Rebuild path cache after loading"""
        self.path_cache = {"/": self.root}
        self._rebuild_cache_helper(self.root)
    
    def _rebuild_cache_helper(self, node):
        if node.is_directory():
            for child in node.children.values():
                self.path_cache[child.get_path()] = child
                self._rebuild_cache_helper(child)
    
    def create_file(self, path, user, content=""):
        result = super().create_file(path, user, content)
        if result[0]:
            self._save()  # Persist after changes
        return result
    
    # Similar for delete, rename, move, write_file...
```

**Better approach for production**: Use database
```python
# SQLite schema
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    name TEXT,
    type TEXT,
    owner TEXT,
    size INTEGER,
    created_at TIMESTAMP,
    modified_at TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES nodes(id)
);

CREATE TABLE file_content (
    node_id TEXT PRIMARY KEY,
    content BLOB,
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

CREATE INDEX idx_parent ON nodes(parent_id);
CREATE INDEX idx_name ON nodes(name);
```

---

### 2. Distributed File System

**Challenges**:
- How to shard file system across multiple servers?
- How to ensure consistency?
- How to handle server failures?

**Approach: Shard by directory**:
```python
class DistributedFileSystem:
    def __init__(self, shard_servers: List[str]):
        self.shard_servers = shard_servers
        self.metadata_server = MetadataServer()
    
    def _get_shard(self, path: str) -> str:
        """Determine which server holds this path"""
        # Hash path to determine shard
        shard_id = hash(path) % len(self.shard_servers)
        return self.shard_servers[shard_id]
    
    def create_file(self, path, user, content):
        # Get shard
        shard = self._get_shard(path)
        
        # Create on shard server
        response = requests.post(f"http://{shard}/create", json={
            "path": path,
            "user": user,
            "content": content
        })
        
        # Update metadata
        if response.ok:
            self.metadata_server.record_file(path, shard)
        
        return response.ok, response.text
```

---

### 3. Monitoring

**Key metrics to track**:
```python
class FileSystemMetrics:
    def __init__(self):
        self.operations_count = defaultdict(int)
        self.operation_latencies = defaultdict(list)
        self.error_count = defaultdict(int)
    
    def record_operation(self, operation: str, latency_ms: float, success: bool):
        self.operations_count[operation] += 1
        self.operation_latencies[operation].append(latency_ms)
        
        if not success:
            self.error_count[operation] += 1
    
    def get_stats(self) -> Dict:
        stats = {}
        for op in self.operations_count:
            latencies = self.operation_latencies[op]
            stats[op] = {
                "count": self.operations_count[op],
                "errors": self.error_count[op],
                "avg_latency_ms": sum(latencies) / len(latencies),
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
                "error_rate": self.error_count[op] / self.operations_count[op]
            }
        return stats
```

---

### 4. Security

**Threats**:
- Path traversal attacks (`../../etc/passwd`)
- Permission bypass
- Quota bypass
- Denial of service (creating too many files)

**Mitigations**:
```python
class SecureFileSystem(FileSystem):
    def create_file(self, path, user, content=""):
        # Validate path
        if not self._is_safe_path(path):
            return False, "Invalid path"
        
        # Rate limiting
        if not self._check_rate_limit(user):
            return False, "Rate limit exceeded"
        
        # Audit logging
        self._audit_log("CREATE_FILE", user, path)
        
        return super().create_file(path, user, content)
    
    def _is_safe_path(self, path):
        """Prevent path traversal attacks"""
        normalized = PathResolver.normalize_path(path)
        
        # Check for suspicious patterns
        if ".." in path or path != normalized:
            return False
        
        # Check path length
        if len(path) > 4096:
            return False
        
        return True
    
    def _audit_log(self, action, user, path):
        """Log all file system operations"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "path": path
        }
        # Write to audit log...
```

---

## Summary

### Do's ✅
- Use Composite pattern for uniform file/directory interface
- Implement path normalization before resolution
- Check permissions on every operation
- Use bit flags for efficient permission storage
- Cache path resolutions for performance
- Invalidate cache after structural changes
- Support both DFS and BFS traversal

### Don'ts ❌
- Don't allow operations without permission checks
- Don't forget to update parent directory on child changes
- Don't create cycles (moving directory to its descendant)
- Don't store absolute paths in nodes (breaks on move)
- Don't skip path normalization (security risk)
- Don't use global lock for all operations (poor concurrency)

### Key Takeaways
1. **Composite Pattern**: Essential for treating files and directories uniformly
2. **Path Resolution**: Normalize first, then traverse tree with caching
3. **Permissions**: Unix-style bit flags are efficient and familiar
4. **Tree Traversal**: DFS for deep searches, BFS for shallow
5. **Cycle Detection**: Essential for move operation - traverse up from destination to check if it hits source
6. **Cache Invalidation**: Must invalidate after any structural change
7. **Concurrency**: Fine-grained locking or read-write locks for better performance

---

**Time to Master**: 3-4 hours  
**Difficulty**: Medium-Hard  
**Key Patterns**: Composite, Observer  
**Critical Skills**: Tree algorithms, path parsing, permission systems, concurrency
