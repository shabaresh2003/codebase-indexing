import os
import hashlib
import logging
from git import Repo

# Tree-sitter imports
from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_go
import tree_sitter_rust

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Tree-sitter Languages
LANGUAGES = {
    "python": Language(tree_sitter_python.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "typescript": Language(tree_sitter_typescript.language("typescript")),
    "tsx": Language(tree_sitter_typescript.language("tsx")),
    "go": Language(tree_sitter_go.language()),
    "rust": Language(tree_sitter_rust.language())
}

# Map file extensions to languages
EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust"
}

# Node types to extract per language
EXTRACT_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "method_definition", "class_declaration", "arrow_function"},
    "typescript": {"function_declaration", "method_definition", "class_declaration", "arrow_function"},
    "tsx": {"function_declaration", "method_definition", "class_declaration", "arrow_function"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item"}
}

def get_file_hash(filepath):
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def clone_or_pull_repo(github_url, base_dir="./repos"):
    """Clones a repo if it doesn't exist, or pulls latest changes if it does."""
    os.makedirs(base_dir, exist_ok=True)
    repo_name = github_url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    repo_path = os.path.join(base_dir, repo_name)
    
    if os.path.exists(repo_path):
        logger.info(f"Repository {repo_name} exists locally. Pulling latest changes...")
        repo = Repo(repo_path)
        origin = repo.remotes.origin
        origin.pull()
    else:
        logger.info(f"Cloning repository {github_url} to {repo_path}...")
        Repo.clone_from(github_url, repo_path)
        
    return repo_path, repo_name

def extract_chunks(filepath, language_key):
    """Extracts function and class chunks from a file using Tree-sitter."""
    try:
        with open(filepath, 'rb') as f:
            source_bytes = f.read()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return []

    parser = Parser(LANGUAGES[language_key])
    tree = parser.parse(source_bytes)
    
    chunks = []
    target_types = EXTRACT_NODE_TYPES.get(language_key, set())
    
    # Recursive walk
    def traverse(node):
        if node.type in target_types:
            # Extract source code for this node
            code_bytes = source_bytes[node.start_byte:node.end_byte]
            code_text = code_bytes.decode('utf-8', errors='replace')
            
            # Find the name of the function/class (heuristic: find first child named 'name' or 'identifier')
            name = "unknown"
            for child in node.children:
                # Often the identifier is called 'identifier' or 'name'
                if "identifier" in child.type or "name" in child.type:
                    name_bytes = source_bytes[child.start_byte:child.end_byte]
                    name = name_bytes.decode('utf-8', errors='replace')
                    break
            
            chunks.append({
                "name": name,
                "code": code_text,
                "file": filepath,
                "lineno": node.start_point[0] + 1, # tree-sitter points are 0-indexed
                "docstring": "", # Simple heuristic extraction omitted for multi-language
                "type": node.type
            })
            
        # Continue traversal
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return chunks

def index_repo(repo_path, previous_hashes=None):
    """
    Walks through the repository, extracting chunks for changed/new files.
    Returns:
        - new_chunks: list of dicts
        - new_hashes: dict mapping filepath to file hash
        - deleted_files: list of filepaths that were deleted since last index
    """
    if previous_hashes is None:
        previous_hashes = {}
        
    new_chunks = []
    new_hashes = {}
    current_files = set()
    
    valid_extensions = set(EXTENSION_MAP.keys())
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories like .git
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in valid_extensions:
                filepath = os.path.join(root, file)
                current_files.add(filepath)
                
                file_hash = get_file_hash(filepath)
                new_hashes[filepath] = file_hash
                
                # Check if file is new or modified
                if filepath not in previous_hashes or previous_hashes[filepath] != file_hash:
                    logger.info(f"Parsing {filepath}...")
                    language_key = EXTENSION_MAP[ext]
                    file_chunks = extract_chunks(filepath, language_key)
                    new_chunks.extend(file_chunks)

    # Determine deleted files
    deleted_files = [f for f in previous_hashes if f not in current_files]
    
    return new_chunks, new_hashes, deleted_files
