"""
File System - Single File Implementation
For coding interviews and production-ready reference

Features:
- Hierarchical directory structure (tree-based)
- File and directory operations (create, delete, rename, move, copy)
- Unix-style permissions (read, write, execute for owner/group/others)
- Path resolution (absolute and relative paths)
- Efficient traversal (DFS and BFS)
- File content management
- Search functionality (by name, extension, size)

Interview Focus:
- Tree data structure for hierarchy
- Composite pattern for files and directories
- Path parsing and resolution algorithms
- Permission bit manipulation
- Graph traversal algorithms (DFS/BFS)
- String manipulation for path operations
- Caching for performance optimization

Author: Interview Prep
Date: 2024
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, deque
from threading import RLock
import time
import uuid
from datetime import datetime


# ============================================================================
# SECTION 1: MODELS - Core data classes and enums
# ============================================================================

class Permission(IntFlag):
    """
    Unix-style permissions using bit flags
    
    Interview Focus: Why use bit flags? Efficient storage and checking.
    Each permission is a bit: rwx rwx rwx (owner, group, others)
    """
    NONE = 0
    EXECUTE = 1      # 001
    WRITE = 2        # 010
    READ = 4         # 100
    
    # Common combinations
    READ_WRITE = READ | WRITE
    READ_EXECUTE = READ | EXECUTE
    ALL = READ | WRITE | EXECUTE


class FileType(Enum):
    """File system node types"""
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


@dataclass
class FileMetadata:
    """
    Metadata for file system nodes
    
    Interview Focus: What metadata is essential for a file system?
    - Size, timestamps, owner, permissions are standard
    """
    name: str
    type: FileType
    size: int = 0  # in bytes
    owner: str = "root"
    group: str = "root"
    
    # Permissions: owner, group, others
    owner_permissions: Permission = Permission.ALL
    group_permissions: Permission = Permission.READ_EXECUTE
    other_permissions: Permission = Permission.READ
    
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    
    def update_modified_time(self):
        """Update modification timestamp"""
        self.modified_at = datetime.now()
    
    def update_accessed_time(self):
        """Update access timestamp"""
        self.accessed_at = datetime.now()


# ============================================================================
# SECTION 2: FILE SYSTEM NODES - Composite Pattern
# ============================================================================

class FileSystemNode(ABC):
    """
    Abstract base for file system nodes
    
    Composite Pattern: Uniform interface for files and directories
    
    Interview Focus: Why Composite Pattern?
    - Files and directories share operations (delete, rename, get metadata)
    - Directories can contain both files and other directories
    - Enables recursive operations on entire trees
    """
    
    def __init__(self, metadata: FileMetadata, parent: Optional['Directory'] = None):
        self.metadata = metadata
        self.parent = parent
        self.id = str(uuid.uuid4())
    
    @abstractmethod
    def get_size(self) -> int:
        """Get size of node (files: content size, directories: sum of children)"""
        pass
    
    @abstractmethod
    def is_directory(self) -> bool:
        """Check if node is a directory"""
        pass
    
    def get_name(self) -> str:
        """Get node name"""
        return self.metadata.name
    
    def get_path(self) -> str:
        """
        Get absolute path from root
        
        Interview Focus: How to construct path efficiently?
        Traverse up to root, then reverse.
        
        Time Complexity: O(depth)
        Space Complexity: O(depth)
        """
        if self.parent is None:
            return "/" if self.metadata.name == "/" else "/" + self.metadata.name
        
        path_parts = []
        current = self
        
        while current.parent is not None:
            path_parts.append(current.metadata.name)
            current = current.parent
        
        # Root node
        if not path_parts:
            return "/"
        
        return "/" + "/".join(reversed(path_parts))
    
    def has_permission(self, user: str, permission: Permission) -> bool:
        """
        Check if user has specific permission
        
        Interview Focus: How to implement permission checking?
        - Check owner first, then group, then others
        - Use bitwise AND to check specific permission
        """
        metadata = self.metadata
        
        # Owner check
        if user == metadata.owner:
            return bool(metadata.owner_permissions & permission)
        
        # Group check (simplified - assume user has group)
        if user in [metadata.group]:  # In reality, check user's groups
            return bool(metadata.group_permissions & permission)
        
        # Others
        return bool(metadata.other_permissions & permission)


class File(FileSystemNode):
    """
    File node in the file system
    
    Key Features:
    - Stores content as string
    - Tracks size automatically
    - Supports read/write operations
    
    Interview Focus: Why separate File from Directory?
    - Different behaviors (files have content, directories have children)
    - Type safety
    - Clear separation of concerns
    """
    
    def __init__(self, metadata: FileMetadata, parent: Optional['Directory'] = None, 
                 content: str = ""):
        super().__init__(metadata, parent)
        self._content = content
        self.metadata.size = len(content.encode('utf-8'))
        self.metadata.type = FileType.FILE
    
    def get_size(self) -> int:
        """Get file size in bytes"""
        return self.metadata.size
    
    def is_directory(self) -> bool:
        """Files are not directories"""
        return False
    
    def read(self, user: str) -> Tuple[bool, str]:
        """
        Read file content
        
        Interview Focus: Permission checking before operation
        """
        if not self.has_permission(user, Permission.READ):
            return False, f"Permission denied: {user} cannot read {self.get_name()}"
        
        self.metadata.update_accessed_time()
        return True, self._content
    
    def write(self, user: str, content: str) -> Tuple[bool, str]:
        """
        Write content to file
        
        Interview Focus: Update size and timestamps after write
        """
        if not self.has_permission(user, Permission.WRITE):
            return False, f"Permission denied: {user} cannot write to {self.get_name()}"
        
        self._content = content
        self.metadata.size = len(content.encode('utf-8'))
        self.metadata.update_modified_time()
        
        return True, f"Written {self.metadata.size} bytes to {self.get_name()}"
    
    def append(self, user: str, content: str) -> Tuple[bool, str]:
        """Append content to file"""
        if not self.has_permission(user, Permission.WRITE):
            return False, f"Permission denied: {user} cannot write to {self.get_name()}"
        
        self._content += content
        self.metadata.size = len(self._content.encode('utf-8'))
        self.metadata.update_modified_time()
        
        return True, f"Appended to {self.get_name()}"


class Directory(FileSystemNode):
    """
    Directory node in the file system
    
    Key Features:
    - Contains children (files and subdirectories)
    - Supports directory operations (add, remove, list)
    - Size is sum of all children
    
    Interview Focus: How to efficiently manage children?
    - Use dict for O(1) lookup by name
    - Keep track of total size
    """
    
    def __init__(self, metadata: FileMetadata, parent: Optional['Directory'] = None):
        super().__init__(metadata, parent)
        self.children: Dict[str, FileSystemNode] = {}
        self.metadata.type = FileType.DIRECTORY
    
    def get_size(self) -> int:
        """
        Get total size of directory (sum of all children recursively)
        
        Time Complexity: O(n) where n is total nodes in subtree
        Space Complexity: O(h) where h is tree height (recursion stack)
        
        Interview Focus: Recursive calculation vs cached value trade-off
        """
        total_size = 0
        for child in self.children.values():
            total_size += child.get_size()
        return total_size
    
    def is_directory(self) -> bool:
        """Directories are directories"""
        return True
    
    def add_child(self, child: FileSystemNode) -> Tuple[bool, str]:
        """
        Add child node to directory
        
        Interview Focus: What checks are needed?
        - Name uniqueness
        - Parent assignment
        """
        child_name = child.get_name()
        
        if child_name in self.children:
            return False, f"Node '{child_name}' already exists in {self.get_name()}"
        
        self.children[child_name] = child
        child.parent = self
        self.metadata.update_modified_time()
        
        return True, f"Added {child_name} to {self.get_name()}"
    
    def remove_child(self, name: str) -> Tuple[bool, str]:
        """Remove child by name"""
        if name not in self.children:
            return False, f"Node '{name}' not found in {self.get_name()}"
        
        del self.children[name]
        self.metadata.update_modified_time()
        
        return True, f"Removed {name} from {self.get_name()}"
    
    def get_child(self, name: str) -> Optional[FileSystemNode]:
        """
        Get child by name
        
        Time Complexity: O(1)
        """
        return self.children.get(name)
    
    def list_children(self, user: str) -> List[str]:
        """
        List all children names
        
        Interview Focus: Need read+execute permission for directory listing
        """
        if not self.has_permission(user, Permission.READ | Permission.EXECUTE):
            return []
        
        self.metadata.update_accessed_time()
        return list(self.children.keys())
    
    def is_empty(self) -> bool:
        """Check if directory is empty"""
        return len(self.children) == 0


# ============================================================================
# SECTION 3: PATH RESOLVER - Path parsing and navigation
# ============================================================================

class PathResolver:
    """
    Resolves paths in the file system
    
    Responsibilities:
    - Parse absolute and relative paths
    - Navigate directory structure
    - Handle special path components (., .., ~)
    
    Interview Focus: How to parse and resolve paths efficiently?
    """
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalize path by resolving . and ..
        
        Interview Focus: Classic algorithm using stack
        
        Time Complexity: O(n) where n is path length
        Space Complexity: O(n)
        
        Examples:
        - "/a/./b" -> "/a/b"
        - "/a/../b" -> "/b"
        - "/a/b/../../c" -> "/c"
        """
        if not path:
            return "/"
        
        # Split path into components
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
        
        # Build normalized path
        if not stack:
            return "/"
        return "/" + "/".join(stack)
    
    @staticmethod
    def split_path(path: str) -> Tuple[str, str]:
        """
        Split path into parent directory and filename
        
        Examples:
        - "/a/b/c" -> ("/a/b", "c")
        - "/a" -> ("/", "a")
        - "/" -> ("", "/")
        """
        normalized = PathResolver.normalize_path(path)
        
        if normalized == "/":
            return "", "/"
        
        parts = normalized.rsplit("/", 1)
        if len(parts) == 1:
            return "/", parts[0]
        
        parent = parts[0] if parts[0] else "/"
        filename = parts[1]
        
        return parent, filename
    
    @staticmethod
    def is_absolute_path(path: str) -> bool:
        """Check if path is absolute"""
        return path.startswith("/")


# ============================================================================
# SECTION 4: FILE SYSTEM - Main controller
# ============================================================================

class FileSystem:
    """
    Main file system controller
    
    Responsibilities:
    - Manage root directory
    - Provide file/directory operations
    - Handle path resolution
    - Maintain file system integrity
    
    Thread Safety: Uses RLock for all operations
    
    Interview Focus: How to coordinate file system operations?
    - Path-based API for user-friendly interface
    - Internal node-based operations for efficiency
    - Permission checking on every operation
    """
    
    def __init__(self):
        """Initialize file system with root directory"""
        # Create root directory
        root_metadata = FileMetadata(
            name="/",
            type=FileType.DIRECTORY,
            owner="root",
            group="root"
        )
        self.root = Directory(root_metadata, parent=None)
        
        # Current working directory (for relative paths)
        self.cwd = self.root
        
        # Path cache for performance (path -> node)
        self.path_cache: Dict[str, FileSystemNode] = {"/": self.root}
        
        # Thread safety
        self.lock = RLock()
    
    # ========================================================================
    # PATH RESOLUTION
    # ========================================================================
    
    def resolve_path(self, path: str) -> Optional[FileSystemNode]:
        """
        Resolve path to file system node
        
        Interview Focus: Path resolution algorithm
        - Normalize path first
        - Check cache for performance
        - Traverse tree from root
        
        Time Complexity: O(depth) for cache miss, O(1) for cache hit
        Space Complexity: O(depth) for path components
        """
        with self.lock:
            # Normalize path
            normalized = PathResolver.normalize_path(path)
            
            # Check cache
            if normalized in self.path_cache:
                return self.path_cache[normalized]
            
            # Root case
            if normalized == "/":
                return self.root
            
            # Split path and traverse
            components = [c for c in normalized.split("/") if c]
            current = self.root
            
            for component in components:
                if not current.is_directory():
                    return None
                
                current = current.get_child(component)
                if current is None:
                    return None
            
            # Cache the result
            self.path_cache[normalized] = current
            return current
    
    def _invalidate_path_cache(self, path: str):
        """
        Invalidate cache entries for path and descendants
        
        Interview Focus: Cache invalidation strategy
        """
        normalized = PathResolver.normalize_path(path)
        
        # Remove path and all descendants from cache
        keys_to_remove = [k for k in self.path_cache.keys() 
                         if k.startswith(normalized)]
        
        for key in keys_to_remove:
            del self.path_cache[key]
    
    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================
    
    def create_file(self, path: str, user: str, content: str = "") -> Tuple[bool, str]:
        """
        Create a new file
        
        Interview Focus: How to create a file?
        1. Parse path to get parent directory and filename
        2. Check if parent exists and is a directory
        3. Check write permission on parent
        4. Create file node
        5. Add to parent
        """
        with self.lock:
            # Parse path
            parent_path, filename = PathResolver.split_path(path)
            
            # Get parent directory
            parent_node = self.resolve_path(parent_path)
            if parent_node is None:
                return False, f"Directory not found: {parent_path}"
            
            if not parent_node.is_directory():
                return False, f"Not a directory: {parent_path}"
            
            parent = parent_node  # Type: Directory
            
            # Check write permission on parent
            if not parent.has_permission(user, Permission.WRITE):
                return False, f"Permission denied: cannot create file in {parent_path}"
            
            # Check if file already exists
            if parent.get_child(filename) is not None:
                return False, f"File already exists: {path}"
            
            # Create file
            file_metadata = FileMetadata(
                name=filename,
                type=FileType.FILE,
                owner=user,
                group=user
            )
            file_node = File(file_metadata, parent, content)
            
            # Add to parent
            success, msg = parent.add_child(file_node)
            if success:
                self._invalidate_path_cache(parent_path)
                return True, f"File created: {path}"
            
            return False, msg
    
    def create_directory(self, path: str, user: str) -> Tuple[bool, str]:
        """
        Create a new directory
        
        Interview Focus: Similar to create_file but creates directory
        """
        with self.lock:
            # Parse path
            parent_path, dirname = PathResolver.split_path(path)
            
            # Get parent directory
            parent_node = self.resolve_path(parent_path)
            if parent_node is None:
                return False, f"Directory not found: {parent_path}"
            
            if not parent_node.is_directory():
                return False, f"Not a directory: {parent_path}"
            
            parent = parent_node  # Type: Directory
            
            # Check write permission
            if not parent.has_permission(user, Permission.WRITE):
                return False, f"Permission denied: cannot create directory in {parent_path}"
            
            # Check if already exists
            if parent.get_child(dirname) is not None:
                return False, f"Directory already exists: {path}"
            
            # Create directory
            dir_metadata = FileMetadata(
                name=dirname,
                type=FileType.DIRECTORY,
                owner=user,
                group=user
            )
            dir_node = Directory(dir_metadata, parent)
            
            # Add to parent
            success, msg = parent.add_child(dir_node)
            if success:
                self._invalidate_path_cache(parent_path)
                return True, f"Directory created: {path}"
            
            return False, msg
    
    def delete(self, path: str, user: str, recursive: bool = False) -> Tuple[bool, str]:
        """
        Delete file or directory
        
        Interview Focus: Recursive deletion
        - Non-recursive: directory must be empty
        - Recursive: delete all children first (DFS)
        
        Key Challenges:
        - Permission checking
        - Preventing root deletion
        - Handling non-empty directories
        """
        with self.lock:
            # Cannot delete root
            if path == "/":
                return False, "Cannot delete root directory"
            
            # Resolve node
            node = self.resolve_path(path)
            if node is None:
                return False, f"Path not found: {path}"
            
            # Check parent permission
            if node.parent is None:
                return False, "Cannot delete root"
            
            parent = node.parent
            if not parent.has_permission(user, Permission.WRITE):
                return False, f"Permission denied: cannot delete in {parent.get_path()}"
            
            # Check if directory is empty
            if node.is_directory():
                directory = node  # Type: Directory
                if not directory.is_empty() and not recursive:
                    return False, f"Directory not empty: {path}. Use recursive=True"
            
            # Remove from parent
            success, msg = parent.remove_child(node.get_name())
            if success:
                self._invalidate_path_cache(path)
                return True, f"Deleted: {path}"
            
            return False, msg
    
    def rename(self, old_path: str, new_name: str, user: str) -> Tuple[bool, str]:
        """
        Rename file or directory
        
        Interview Focus: Rename vs Move
        - Rename: same directory, different name
        - Move: different directory, may have different name
        """
        with self.lock:
            # Resolve node
            node = self.resolve_path(old_path)
            if node is None:
                return False, f"Path not found: {old_path}"
            
            # Cannot rename root
            if node.parent is None:
                return False, "Cannot rename root directory"
            
            parent = node.parent
            
            # Check write permission on parent
            if not parent.has_permission(user, Permission.WRITE):
                return False, f"Permission denied: cannot rename in {parent.get_path()}"
            
            # Check if new name already exists
            if parent.get_child(new_name) is not None:
                return False, f"Name already exists: {new_name}"
            
            # Remove old entry
            old_name = node.get_name()
            parent.remove_child(old_name)
            
            # Update name
            node.metadata.name = new_name
            node.metadata.update_modified_time()
            
            # Add with new name
            parent.add_child(node)
            
            # Invalidate cache
            self._invalidate_path_cache(parent.get_path())
            
            return True, f"Renamed {old_name} to {new_name}"
    
    def move(self, src_path: str, dest_path: str, user: str) -> Tuple[bool, str]:
        """
        Move file or directory to new location
        
        Interview Focus: Move algorithm
        1. Resolve source node
        2. Check if moving to descendant (would create cycle)
        3. Remove from old parent
        4. Add to new parent
        
        Key Challenges:
        - Cycle detection
        - Permission checking on both source and destination
        - Handling name conflicts
        """
        with self.lock:
            # Resolve source
            src_node = self.resolve_path(src_path)
            if src_node is None:
                return False, f"Source not found: {src_path}"
            
            # Cannot move root
            if src_node.parent is None:
                return False, "Cannot move root directory"
            
            # Resolve destination
            dest_node = self.resolve_path(dest_path)
            if dest_node is None or not dest_node.is_directory():
                return False, f"Destination directory not found: {dest_path}"
            
            dest_dir = dest_node  # Type: Directory
            
            # Check permissions
            if not src_node.parent.has_permission(user, Permission.WRITE):
                return False, "Permission denied: cannot move from source directory"
            
            if not dest_dir.has_permission(user, Permission.WRITE):
                return False, "Permission denied: cannot move to destination directory"
            
            # Check if moving to descendant (would create cycle)
            if src_node.is_directory() and self._is_ancestor(src_node, dest_dir):
                return False, "Cannot move directory to its own descendant"
            
            # Check name conflict
            src_name = src_node.get_name()
            if dest_dir.get_child(src_name) is not None:
                return False, f"Name conflict: {src_name} already exists in destination"
            
            # Remove from old parent
            old_parent = src_node.parent
            old_parent.remove_child(src_name)
            
            # Add to new parent
            dest_dir.add_child(src_node)
            
            # Invalidate cache
            self._invalidate_path_cache(old_parent.get_path())
            self._invalidate_path_cache(dest_path)
            
            return True, f"Moved {src_path} to {dest_path}"
    
    def _is_ancestor(self, potential_ancestor: FileSystemNode, 
                    node: FileSystemNode) -> bool:
        """
        Check if potential_ancestor is an ancestor of node
        
        Interview Focus: Cycle detection algorithm
        Traverse up from node to root, check if we hit potential_ancestor
        """
        current = node
        while current is not None:
            if current == potential_ancestor:
                return True
            current = current.parent
        return False
    
    def read_file(self, path: str, user: str) -> Tuple[bool, str]:
        """Read file content"""
        with self.lock:
            node = self.resolve_path(path)
            if node is None:
                return False, f"File not found: {path}"
            
            if node.is_directory():
                return False, f"Is a directory: {path}"
            
            file = node  # Type: File
            return file.read(user)
    
    def write_file(self, path: str, user: str, content: str) -> Tuple[bool, str]:
        """Write content to file"""
        with self.lock:
            node = self.resolve_path(path)
            if node is None:
                return False, f"File not found: {path}"
            
            if node.is_directory():
                return False, f"Is a directory: {path}"
            
            file = node  # Type: File
            return file.write(user, content)
    
    # ========================================================================
    # TRAVERSAL AND SEARCH
    # ========================================================================
    
    def list_directory(self, path: str, user: str) -> Tuple[bool, List[str]]:
        """
        List directory contents
        
        Interview Focus: Directory listing with permissions
        """
        with self.lock:
            node = self.resolve_path(path)
            if node is None:
                return False, []
            
            if not node.is_directory():
                return False, []
            
            directory = node  # Type: Directory
            children = directory.list_children(user)
            return True, children
    
    def traverse_dfs(self, path: str, user: str) -> List[str]:
        """
        Depth-first traversal of file system
        
        Interview Focus: DFS algorithm for tree traversal
        
        Time Complexity: O(n) where n is total nodes
        Space Complexity: O(h) where h is tree height (recursion)
        """
        with self.lock:
            node = self.resolve_path(path)
            if node is None:
                return []
            
            result = []
            self._dfs_helper(node, user, result)
            return result
    
    def _dfs_helper(self, node: FileSystemNode, user: str, result: List[str]):
        """DFS helper - recursive"""
        result.append(node.get_path())
        
        if node.is_directory():
            directory = node  # Type: Directory
            if directory.has_permission(user, Permission.READ | Permission.EXECUTE):
                for child_name in sorted(directory.children.keys()):
                    child = directory.children[child_name]
                    self._dfs_helper(child, user, result)
    
    def traverse_bfs(self, path: str, user: str) -> List[str]:
        """
        Breadth-first traversal of file system
        
        Interview Focus: BFS algorithm using queue
        
        Time Complexity: O(n) where n is total nodes
        Space Complexity: O(w) where w is maximum width of tree
        """
        with self.lock:
            node = self.resolve_path(path)
            if node is None:
                return []
            
            result = []
            queue = deque([node])
            
            while queue:
                current = queue.popleft()
                result.append(current.get_path())
                
                if current.is_directory():
                    directory = current  # Type: Directory
                    if directory.has_permission(user, Permission.READ | Permission.EXECUTE):
                        for child_name in sorted(directory.children.keys()):
                            child = directory.children[child_name]
                            queue.append(child)
            
            return result
    
    def search_by_name(self, name: str, start_path: str = "/", 
                      user: str = "root") -> List[str]:
        """
        Search for files/directories by name
        
        Interview Focus: Search algorithm with DFS
        """
        with self.lock:
            start_node = self.resolve_path(start_path)
            if start_node is None:
                return []
            
            results = []
            self._search_helper(start_node, name, user, results)
            return results
    
    def _search_helper(self, node: FileSystemNode, name: str, 
                      user: str, results: List[str]):
        """Search helper - recursive DFS"""
        if node.get_name() == name or name in node.get_name():
            results.append(node.get_path())
        
        if node.is_directory():
            directory = node  # Type: Directory
            if directory.has_permission(user, Permission.READ | Permission.EXECUTE):
                for child in directory.children.values():
                    self._search_helper(child, name, user, results)
    
    def get_info(self, path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Get detailed information about file or directory
        
        Interview Focus: Metadata retrieval
        """
        with self.lock:
            node = self.resolve_path(path)
            if node is None:
                return False, {}
            
            info = {
                "path": node.get_path(),
                "name": node.get_name(),
                "type": node.metadata.type.value,
                "size": node.get_size(),
                "owner": node.metadata.owner,
                "group": node.metadata.group,
                "permissions": {
                    "owner": str(node.metadata.owner_permissions),
                    "group": str(node.metadata.group_permissions),
                    "others": str(node.metadata.other_permissions)
                },
                "created_at": node.metadata.created_at.isoformat(),
                "modified_at": node.metadata.modified_at.isoformat(),
                "accessed_at": node.metadata.accessed_at.isoformat()
            }
            
            if node.is_directory():
                directory = node  # Type: Directory
                info["children_count"] = len(directory.children)
            
            return True, info


# ============================================================================
# DEMO - Demonstration and testing code
# ============================================================================

def print_separator(title: str = ""):
    """Print visual separator"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    else:
        print(f"{'='*70}\n")


def demo_basic_operations():
    """Demonstrate basic file system operations"""
    print_separator("Basic File System Operations")
    
    fs = FileSystem()
    user = "alice"
    
    # Create directories - need to use root or proper owner
    print("\n🔹 Creating directories:")
    success, msg = fs.create_directory("/home", "root")
    print(f"  /home: {msg}")
    
    # Alice creates her own directory (root creates it and changes owner)
    success, msg = fs.create_directory("/home/alice", "root")
    print(f"  /home/alice: {msg}")
    
    # Now alice can create subdirectories in her own directory
    # But first, let's get the alice directory and change its owner
    alice_dir = fs.resolve_path("/home/alice")
    if alice_dir:
        alice_dir.metadata.owner = user
    
    success, msg = fs.create_directory("/home/alice/documents", user)
    print(f"  /home/alice/documents: {msg}")
    
    # Create files
    print("\n🔹 Creating files:")
    success, msg = fs.create_file("/home/alice/readme.txt", user, 
                                  "Welcome to my home directory!")
    print(f"  readme.txt: {msg}")
    
    success, msg = fs.create_file("/home/alice/documents/notes.txt", user,
                                  "Important notes here")
    print(f"  notes.txt: {msg}")
    
    # List directory
    print("\n🔹 Listing /home/alice:")
    success, children = fs.list_directory("/home/alice", user)
    if success:
        for child in children:
            print(f"  - {child}")
    
    # Read file
    print("\n🔹 Reading /home/alice/readme.txt:")
    success, content = fs.read_file("/home/alice/readme.txt", user)
    if success:
        print(f"  Content: {content}")
    
    return fs


def demo_file_operations():
    """Demonstrate file read/write operations"""
    print_separator("File Operations")
    
    fs = FileSystem()
    user = "bob"
    
    # Create and write to file
    print("\n🔹 Creating and writing to file:")
    fs.create_directory("/tmp", "root")
    fs.create_file("/tmp/test.txt", user, "Initial content")
    
    # Read
    success, content = fs.read_file("/tmp/test.txt", user)
    print(f"  Initial content: {content if success else 'Error: ' + content}")
    
    # Write
    fs.write_file("/tmp/test.txt", user, "Updated content with more text")
    success, content = fs.read_file("/tmp/test.txt", user)
    print(f"  Updated content: {content if success else 'Error: ' + content}")
    
    # Get file info
    print("\n🔹 File information:")
    success, info = fs.get_info("/tmp/test.txt")
    if success:
        print(f"  Path: {info['path']}")
        print(f"  Size: {info['size']} bytes")
        print(f"  Owner: {info['owner']}")
        print(f"  Type: {info['type']}")


def demo_rename_and_move():
    """Demonstrate rename and move operations"""
    print_separator("Rename and Move Operations")
    
    fs = FileSystem()
    user = "charlie"
    
    # Setup
    fs.create_directory("/projects", user)
    fs.create_directory("/archive", user)
    fs.create_file("/projects/old_project.txt", user, "Old project data")
    
    print("\n🔹 Initial structure:")
    paths = fs.traverse_dfs("/", user)
    for path in paths:
        print(f"  {path}")
    
    # Rename
    print("\n🔹 Renaming old_project.txt to new_project.txt:")
    success, msg = fs.rename("/projects/old_project.txt", "new_project.txt", user)
    print(f"  {msg}")
    
    # Move
    print("\n🔹 Moving new_project.txt to /archive:")
    success, msg = fs.move("/projects/new_project.txt", "/archive", user)
    print(f"  {msg}")
    
    print("\n🔹 Final structure:")
    paths = fs.traverse_dfs("/", user)
    for path in paths:
        print(f"  {path}")


def demo_permissions():
    """Demonstrate permission system"""
    print_separator("Permission System")
    
    fs = FileSystem()
    owner = "alice"
    other_user = "bob"
    
    # Create file as alice
    fs.create_directory("/home", "root")
    # Give alice ownership of /home so she can create her directory
    home_dir = fs.resolve_path("/home")
    home_dir.metadata.owner = owner
    home_dir.metadata.owner_permissions = Permission.ALL
    
    fs.create_directory("/home/alice", owner)
    fs.create_file("/home/alice/private.txt", owner, "Private data")
    
    print("\n🔹 File created by alice")
    
    # Alice can read
    success, content = fs.read_file("/home/alice/private.txt", owner)
    print(f"  Alice reads: {'✓ ' + content if success else '✗ Permission denied'}")
    
    # Bob tries to read (should succeed - others have read permission by default)
    success, content = fs.read_file("/home/alice/private.txt", other_user)
    print(f"  Bob reads: {'✓ ' + content if success else '✗ Permission denied'}")
    
    # Bob tries to write (should fail)
    success, msg = fs.write_file("/home/alice/private.txt", other_user, "Hacked!")
    print(f"  Bob writes: {'✓ Success' if success else '✗ ' + msg}")
    
    # Bob tries to delete (should fail - no write permission on parent)
    success, msg = fs.delete("/home/alice/private.txt", other_user)
    print(f"  Bob deletes: {'✓ Success' if success else '✗ ' + msg}")


def demo_traversal_and_search():
    """Demonstrate traversal and search"""
    print_separator("Traversal and Search")
    
    fs = FileSystem()
    user = "root"
    
    # Create complex structure
    fs.create_directory("/projects", user)
    fs.create_directory("/projects/web", user)
    fs.create_directory("/projects/mobile", user)
    fs.create_file("/projects/web/index.html", user, "<html></html>")
    fs.create_file("/projects/web/style.css", user, "body{}")
    fs.create_file("/projects/mobile/app.py", user, "print('hello')")
    fs.create_file("/projects/README.md", user, "# Projects")
    
    print("\n🔹 DFS Traversal:")
    paths = fs.traverse_dfs("/projects", user)
    for path in paths:
        print(f"  {path}")
    
    print("\n🔹 BFS Traversal:")
    paths = fs.traverse_bfs("/projects", user)
    for path in paths:
        print(f"  {path}")
    
    print("\n🔹 Search for 'app':")
    results = fs.search_by_name("app", "/", user)
    for result in results:
        print(f"  Found: {result}")
    
    print("\n🔹 Search for 'README':")
    results = fs.search_by_name("README", "/", user)
    for result in results:
        print(f"  Found: {result}")


def demo_delete_operations():
    """Demonstrate delete operations"""
    print_separator("Delete Operations")
    
    fs = FileSystem()
    user = "root"
    
    # Create structure
    fs.create_directory("/test", user)
    fs.create_directory("/test/subdir", user)
    fs.create_file("/test/file1.txt", user, "content")
    fs.create_file("/test/subdir/file2.txt", user, "content")
    
    print("\n🔹 Initial structure:")
    paths = fs.traverse_dfs("/test", user)
    for path in paths:
        print(f"  {path}")
    
    # Try to delete non-empty directory without recursive
    print("\n🔹 Try deleting non-empty /test (no recursive):")
    success, msg = fs.delete("/test", user, recursive=False)
    print(f"  {msg}")
    
    # Delete with recursive
    print("\n🔹 Delete /test with recursive:")
    success, msg = fs.delete("/test", user, recursive=True)
    print(f"  {msg}")
    
    print("\n🔹 Final structure (/ only):")
    paths = fs.traverse_dfs("/", user)
    for path in paths:
        print(f"  {path}")


def run_demo():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("  FILE SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Hierarchical structure, Permissions, Traversal")
    print("="*70)
    
    demo_basic_operations()
    demo_file_operations()
    demo_rename_and_move()
    demo_permissions()
    demo_traversal_and_search()
    demo_delete_operations()
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    Usage:
        python 11_file_system.py
    """
    run_demo()
