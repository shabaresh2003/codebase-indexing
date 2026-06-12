import os
import ast
import hashlib
from git import Repo
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def extract_chunks(filepath):
    """Extracts function and class chunks from a Python file using AST."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return []
        
    try:
        tree = ast.parse(source)
    except Exception as e:
        logger.warning(f"Failed to parse {filepath} (likely invalid Python): {e}")
        return []

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                code = ast.get_source_segment(source, node)
                if not code:
                    continue
                
                # Prepend docstring if available (improves embeddings)
                docstring = ast.get_docstring(node) or ""
                
                chunks.append({
                    "name": node.name,
                    "code": code,
                    "file": filepath,
                    "lineno": node.lineno,
                    "docstring": docstring,
                    "type": type(node).__name__
                })
            except Exception as e:
                logger.warning(f"Failed to extract source for node {node.name} in {filepath}: {e}")
                
    return chunks

def index_repo(repo_path, previous_hashes=None):
    """
    Walks through the repository, extracting chunks for changed/new Python files.
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
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories like .git
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                current_files.add(filepath)
                
                file_hash = get_file_hash(filepath)
                new_hashes[filepath] = file_hash
                
                # Check if file is new or modified
                if filepath not in previous_hashes or previous_hashes[filepath] != file_hash:
                    logger.info(f"Parsing {filepath}...")
                    file_chunks = extract_chunks(filepath)
                    new_chunks.extend(file_chunks)

    # Determine deleted files
    deleted_files = [f for f in previous_hashes if f not in current_files]
    
    return new_chunks, new_hashes, deleted_files
