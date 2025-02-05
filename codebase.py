import os
import logging
import asyncio
import re
import aiofiles
from typing import List, Dict, Any
from collections import deque

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

async def read_file_snippet(file_path, start_line=0, num_lines=10):
    """
    Asynchronously reads a snippet of lines from a file starting at a specific line.
    """
    lines = []
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Skip to start_line
            for _ in range(start_line):
                await f.readline()
            # Read requested lines
            for _ in range(num_lines):
                line = await f.readline()
                if not line:
                    break
                lines.append(line.strip())
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return f"Error reading file: {str(e)}"
    return '\n'.join(lines)

async def read_file_content(file_path: str, max_size: int = 100 * 1024) -> str:
    """
    Read the full content of a file, with size limit.
    Returns the file content or a truncated version if too large.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = await f.read(max_size + 1)
            if len(content) > max_size:
                return content[:max_size] + "\n... (file truncated due to size)"
            return content
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return f"Error reading file: {str(e)}"

async def scan_file_content(file_path: str, keywords: set) -> List[Dict[str, str]]:
    """
    Scan a file for keywords, returning snippets of context.
    This version uses asynchronous iteration and a deque for a sliding window.
    """
    snippets = []
    context_window = deque(maxlen=5)  # Store up to 5 lines as context
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            current_line = 0
            async for line in f:
                context_window.append(line.strip())
                lower_line = line.lower()
                if any(keyword in lower_line for keyword in keywords):
                    snippet = {
                        'line_range': f"{max(0, current_line - len(context_window) + 1)} - {current_line}",
                        'context': '\n'.join(context_window)
                    }
                    snippets.append(snippet)
                current_line += 1
        return snippets
    except Exception as e:
        logging.error(f"Error scanning file {file_path}: {e}")
        return []

async def process_file(file_path: str, code_extensions: set, text_extensions: set, 
                       keywords: set, semaphore: asyncio.Semaphore, is_target_file: bool = False) -> Dict[str, Any]:
    """
    Process a single file by gathering metadata and scanning for relevant content.
    """
    async with semaphore:
        try:
            stat = await asyncio.to_thread(os.stat, file_path)
            file_size = stat.st_size
            last_modified = stat.st_mtime
            file_extension = os.path.splitext(file_path)[1].lower()
            
            MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
            if file_size > MAX_FILE_SIZE:
                return {
                    'path': file_path,
                    'size': file_size,
                    'extension': file_extension,
                    'importance': 'high' if is_target_file else 'low',
                    'last_modified': last_modified,
                    'snippets': [],
                    'skip_reason': 'File too large'
                }
            
            # Check if file is binary by reading an initial chunk in binary mode
            async with aiofiles.open(file_path, 'rb') as f:
                sample = await f.read(8192)
                try:
                    sample.decode('utf-8')
                except UnicodeDecodeError:
                    return {
                        'path': file_path,
                        'size': file_size,
                        'extension': file_extension,
                        'importance': 'high' if is_target_file else 'low',
                        'last_modified': last_modified,
                        'snippets': [],
                        'skip_reason': 'Binary file'
                    }

            # Determine file importance
            if is_target_file:
                importance = 'high'
            elif file_extension in code_extensions:
                importance = 'medium'
            elif file_extension in text_extensions:
                importance = 'low'
            else:
                importance = 'low'

            snippets = []
            if importance != 'low':
                snippets = await scan_file_content(file_path, keywords)

            return {
                'path': file_path,
                'size': file_size,
                'extension': file_extension,
                'importance': importance,
                'last_modified': last_modified,
                'snippets': snippets
            }
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return {'path': file_path, 'error': str(e)}

async def explore_codebase(root_dir: str = '.', task_description: str = '', 
                          get_cache=None, set_cache=None) -> List[Dict[str, Any]]:
    """
    Explores codebase focusing on potentially relevant files based on task keywords.
    """
    keywords = set(re.findall(r'\b\w+\b', task_description.lower()))
    keywords = {word for word in keywords if len(word) > 3}
    
    file_patterns = set(re.findall(r'\b\w+\.[a-zA-Z]+\b|\b\w+(?=\s+file)\b|\b\w+(?=\s+changes)\b', task_description.lower()))
    
    ignored_directories = {'.git', 'node_modules', '__pycache__', 'venv', '.pytest_cache'}
    code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
    text_extensions = code_extensions.union({'.json', '.txt', '.md', '.html', '.css', '.xml', '.csv'})
    
    semaphore = asyncio.Semaphore(100)
    tasks = []

    async def return_cached(result):
        return result

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignored_directories]
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path).lower()
            file_name_no_ext = os.path.splitext(file_name)[0]
            
            is_target_file = any(
                pattern in file_name or pattern == file_name_no_ext
                for pattern in file_patterns
            )
            
            if get_cache and set_cache:
                cache_key = f"{file_path}:{os.path.getmtime(file_path)}"
                cached_result = await get_cache(cache_key)
                
                if cached_result:
                    # Print a message indicating the file is being read from the persistent cache (SQLite)
                    print(f"[INFO] File read from persistent cache: {file_path} (no tokens used)")
                    if is_target_file:
                        cached_result['importance'] = 'high'
                    tasks.append(asyncio.create_task(return_cached(cached_result)))
                    continue
            
            tasks.append(process_file(
                file_path, 
                code_extensions, 
                text_extensions, 
                keywords, 
                semaphore,
                is_target_file
            ))
    
    codebase_summary = await asyncio.gather(*tasks)
    
    if get_cache and set_cache:
        for file_summary in codebase_summary:
            if 'error' not in file_summary:
                cache_key = f"{file_summary['path']}:{file_summary['last_modified']}"
                await set_cache(cache_key, file_summary)
    
    relevant_files = [
        f for f in codebase_summary 
        if isinstance(f, dict) and (f.get('snippets', []) or f.get('importance') == 'high')
    ]
    
    relevant_files.sort(
        key=lambda x: (
            x.get('importance') != 'high',
            -len(x.get('snippets', []))
        )
    )
    
    return relevant_files
