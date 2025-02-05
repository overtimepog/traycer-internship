import os
import logging
import asyncio
import re
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
import json
from concurrent.futures import ProcessPoolExecutor
from persistent_cache import get_cache, set_cache  # Import the persistent cache module
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from pygments.styles.monokai import MonokaiStyle
from rich.syntax import Syntax, SyntaxTheme
from rich.style import Style

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        print(f"{func.__name__} called {wrapper.calls} times this session.")
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

import aiofiles

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

async def scan_file_content(file_path, keywords):
    """
    Line | 49-80 `cli.py`
    Quickly scan a file for relevant keywords using chunked reading to reduce memory usage.
    Returns a list of dictionaries containing:
    - line_range: string showing the range of lines included in the snippet (e.g. "205 - 209")
    - context: the actual text content from those lines
    
    Each match includes 2 lines before and 2 lines after for context. All matches in the file
    are returned, with no limit on the number of snippets.
    """
    snippets = []
    line_buffer = []  # Buffer to store context lines
    current_line = 0
    chunk_size = 8192  # 8KB chunks
    
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Read file in chunks to reduce memory usage
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                    
                lines = chunk.split('\n')
                # Handle line spanning across chunks
                if line_buffer and lines:
                    line_buffer[-1] += lines[0]
                    lines = lines[1:]
                
                line_buffer.extend(lines)
                
                # Process complete lines, keeping a few lines as buffer for context
                while len(line_buffer) > 4:  # Keep 4 lines for context
                    line = line_buffer[0]
                    if any(keyword.lower() in line.lower() for keyword in keywords):
                        # Get context (2 lines before and 2 after)
                        context_start = max(0, len(line_buffer) - 5)
                        context = '\n'.join(line_buffer[context_start:len(line_buffer)])
                        snippets.append({
                            'line_range': f"{current_line - 2} - {current_line + 2}",
                            'context': context
                        })
                    line_buffer.pop(0)
                    current_line += 1
            # Process any remaining lines in the buffer
            for idx, line in enumerate(line_buffer):
                if any(keyword.lower() in line.lower() for keyword in keywords):
                    start = max(0, idx - 2)
                    end = min(len(line_buffer), idx + 3)
                    context = '\n'.join(line_buffer[start:end])
                    start_line = current_line + start + 1  # converting to 1-based numbering
                    end_line = current_line + end
                    snippets.append({
                        'line_range': f"{start_line} - {end_line}",
                        'context': context
                    })
                    
        return snippets
    except Exception as e:
        logging.error(f"Error scanning file {file_path}: {str(e)}")
        return []

async def process_file(file_path, code_extensions, text_extensions, keywords, semaphore):
    """
    Line | 92-115 `cli.py`
    Process a single file by gathering metadata and scanning for relevant content.
    Implements early filtering for binary and large files.
    """
    async with semaphore:
        try:
            file_size = await asyncio.to_thread(os.path.getsize, file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            last_modified = await asyncio.to_thread(os.path.getmtime, file_path)
            
            # Early filtering for large files (>10MB) or binary content
            MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB #TODO: figure out some way to shrink large files so they can be sent as context to the ai model
            if file_size > MAX_FILE_SIZE:
                return {
                    'path': file_path,
                    'size': file_size,
                    'extension': file_extension,
                    'importance': 'low',
                    'last_modified': last_modified,
                    'snippets': [],
                    'skip_reason': 'File too large'
                }
            
            # Check for binary content (read first 8KB)
            async with aiofiles.open(file_path, 'rb') as f:
                sample = await f.read(8192)
                try:
                    sample.decode('utf-8')
                except UnicodeDecodeError:
                    return {
                        'path': file_path,
                        'size': file_size,
                        'extension': file_extension,
                        'importance': 'low',
                        'last_modified': last_modified,
                        'snippets': [],
                        'skip_reason': 'Binary file'
                    }
            
            # Determine importance based on file type
            if file_extension in code_extensions:
                importance = 'high'
            elif file_extension in text_extensions:
                importance = 'medium'
            else:
                importance = 'low'
            
            # Only scan content for potentially relevant files
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
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return {'path': file_path, 'error': str(e)}

async def explore_codebase(root_dir='.', task_description=''):
    """
    Line | 119-157 `cli.py`
    Explores codebase focusing on potentially relevant files based on task keywords.
    Uses ProcessPoolExecutor for CPU-bound tasks and persistent caching for file data.
    """
    """
    Explores codebase focusing on potentially relevant files based on task keywords.
    """
    # Extract keywords from task description
    keywords = set(re.findall(r'\b\w+\b', task_description.lower()))
    keywords = {word for word in keywords if len(word) > 3}  # Filter out short words
    
    ignored_directories = {'.git', 'node_modules', '__pycache__', 'venv', '.pytest_cache'}
    code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
    text_extensions = code_extensions.union({'.json', '.txt', '.md', '.html', '.css', '.xml', '.csv'})
    
    semaphore = asyncio.Semaphore(100)
    tasks = []
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignored_directories]
        for file in files:
            file_path = os.path.join(root, file)
            cache_key = f"{file_path}:{os.path.getmtime(file_path)}"
            cached_result = await get_cache(cache_key)
            
            if cached_result:
                tasks.append(asyncio.create_task(
                    asyncio.sleep(0, result=cached_result)
                ))
            else:
                # Remove ProcessPoolExecutor usage; schedule process_file directly
                tasks.append(process_file(file_path, code_extensions, text_extensions, keywords, semaphore))
    
    codebase_summary = await asyncio.gather(*tasks)
    
    # Update cache with new results
    for file_summary in codebase_summary:
        if 'error' not in file_summary:
            cache_key = f"{file_summary['path']}:{file_summary['last_modified']}"
            await set_cache(cache_key, file_summary)
    
    # Filter out files with no relevant snippets
    relevant_files = [f for f in codebase_summary if isinstance(f, dict) and f.get('snippets', [])]
    return relevant_files

async def batch_relevance_check(files, task_description):
    """
    Check relevance of multiple files in a single AI call to reduce API usage.
    """
    if not files:
        return []
    
    file_summaries = []
    for i in range(0, len(files), 5):  # Process in batches of 5
        batch = files[i:i+5]
        message = {
            "role": "user",
            "content": f"""Task: {task_description}

Files to analyze:
{json.dumps([{
    'path': f['path'],
    'snippets': f['snippets']
} for f in batch], indent=2)}

For each file, respond with a JSON array of objects containing:
- path: file path
- relevance: "high", "medium", or "low"
"""
        }
        
        response = await client.messages.create(
            max_tokens=1024,
            messages=[message],
            model="claude-3-5-sonnet-latest"
        )
        console.print(f"[dim]Received response: {response.content[0].text}[/dim]")
        
        # Handle different response content types
        try:
            if isinstance(response.content, TextBlock):
                # Convert TextBlock to string, clean it to ensure it contains only the JSON part
                content_str = str(response.content[0].text)
                print(f"Received response: {content_str}")
                # Find JSON array in the response
                json_start = content_str.find('[')
                json_end = content_str.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_str = content_str[json_start:json_end]
                    parsed_content = json.loads(json_str)
                    if isinstance(parsed_content, list):
                        file_summaries.extend(parsed_content)
                    else:
                        file_summaries.extend([{'path': f['path'], 'relevance': 'medium'} for f in batch])
                else:
                    # If no JSON array found, use medium relevance
                    file_summaries.extend([{'path': f['path'], 'relevance': 'medium'} for f in batch])
            else:
                # Unknown content type, fallback to medium relevance
                file_summaries.extend([{'path': f['path'], 'relevance': 'medium'} for f in batch])
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logging.error(f"Error processing response: {str(e)}")
            console.print(f"[red]Error processing response: {str(e)}[/red]")
            # Fallback to medium relevance if parsing fails
            file_summaries.extend([{'path': f['path'], 'relevance': 'medium'} for f in batch])
    
    return file_summaries

async def generate_task_plan(task_description, codebase_summary):
    """
    Generate a focused task plan based on relevant files and their contents.
    """
    relevance_results = await batch_relevance_check(codebase_summary, task_description)
    print(f"Received relevance results for {len(relevance_results)} files.")
    print("Results:", json.dumps(relevance_results, indent=2))
    
    # Filter for high relevance files
    filtered_results = [r for r in relevance_results if isinstance(r, dict) and r.get('relevance') == 'high']
    
    # Combine file information with relevance results
    enhanced_summary = []
    for file in codebase_summary:
        relevance = next((r['relevance'] for r in filtered_results if r['path'] == file['path']), 'low')
        summary = {
            'path': file['path'],
            'importance': file['importance'],
            'relevance': relevance,
            'snippets': file['snippets']
        }
        enhanced_summary.append(summary)
    
    message = {
        "role": "user",
        "content": f"""Task: {task_description}

Relevant Files Analysis:
{json.dumps(enhanced_summary, indent=2)}

Please provide a JSON object that strictly adheres to the following format. The JSON must have exactly three top-level keys: 'explanation', 'files_modified', and 'codebase_analysis'. 'explanation' should be a string that briefly describes the task and approach. 'files_modified' should be an array where each element is either a string (representing the file path) or an object with the keys 'path' and 'description' that explain why the file needs modification. 'codebase_analysis' should be either a string or an object containing the keys 'current_state' and 'recommendations'. No additional keys are allowed."""
    }
    
    response = await client.messages.create(
        max_tokens=1024,
        messages=[message],
        model="claude-3-5-sonnet-latest"
    )
    # Parse the response content
    try:
        content_str = str(response.content[0].text) if isinstance(response.content, TextBlock) else response.content[0].text
        
        # Try to find JSON object in the response
        json_start = content_str.find('{')
        json_end = content_str.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = content_str[json_start:json_end]
            try:
                parsed_content = json.loads(json_str)
                return json.dumps(parsed_content, indent=2)
            except json.JSONDecodeError:
                print("Warning: Failed to parse JSON from response")
        
        # If no valid JSON found, return the raw content for validation
        return content_str
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        # Return a basic error response in the expected format
        return json.dumps({
            "explanation": "Error processing AI response",
            "files_modified": [],
            "codebase_analysis": f"Unable to analyze due to error: {str(e)}"
        }, indent=2)

@count_calls
def transform_plan_format(plan_obj: dict) -> dict:
    """
    Convert a plan object with unexpected keys (e.g. "current_state", "recommendations")
    into the expected format with keys:
    - explanation
    - files_modified
    - codebase_analysis
    """
    #TODO: keep track of how many times this function is called per session
    print("Transforming plan format...")
    # If already in expected format, return as is.
    if {"explanation", "files_modified", "codebase_analysis"}.issubset(plan_obj.keys()):
        return plan_obj
    new_plan = {
        "explanation": "The plan outlines optimizations based on current analysis.",
        "files_modified": [],
        "codebase_analysis": json.dumps(plan_obj, indent=2)
    }
    return new_plan

def validate_json_schema(data: dict) -> bool:
    """Validate that the JSON data follows the required schema."""
    required_schema = {
        "explanation": {
            "type": str,
            "required": True
        },
        "files_modified": {
            "type": list,
            "required": True,
            "items": {
                "type": (str, dict),
                "dict_keys": ["path", "description"]
            }
        },
        "codebase_analysis": {
            "type": (str, dict),
            "required": True,
            "dict_keys": ["current_state", "recommendations"] if isinstance(data.get("codebase_analysis"), dict) else None
        }
    }
    
    for key, schema in required_schema.items():
        if schema["required"] and key not in data:
            return False
        
        if key in data:
            value = data[key]
            if not isinstance(value, schema["type"]):
                return False
                
            if isinstance(value, list) and "items" in schema:
                for item in value:
                    if not isinstance(item, schema["items"]["type"]):
                        return False
                    if isinstance(item, dict) and "dict_keys" in schema["items"]:
                        if not all(k in item for k in schema["items"]["dict_keys"]):
                            return False
                            
            if isinstance(value, dict) and schema.get("dict_keys"):
                if not all(k in value for k in schema["dict_keys"]):
                    return False
    
    return True

async def validate_ai_response(response: str):
    """Validate and format AI response with retry logic."""
    MAX_ATTEMPTS = 3
    attempts = 0
    
    while attempts < MAX_ATTEMPTS:
        try:
            response_json = json.loads(response)
            
            # Check if response has the required structure
            if validate_json_schema(response_json):
                return json.dumps(response_json, indent=2)
            
            # If schema validation fails, try to fix the response
            logging.info(f"Attempt {attempts + 1}: JSON schema validation failed, requesting correction")
            response = await fix_json_plan(response, "Response does not match required schema")
            if not response:
                break
                
            attempts += 1
            
        except json.JSONDecodeError as e:
            logging.error(f"Attempt {attempts + 1}: JSON parsing failed: {str(e)}")
            response = await fix_json_plan(response, str(e))
            if not response:
                break
                
            attempts += 1
            
        except Exception as e:
            logging.error(f"Unexpected error during validation: {str(e)}")
            break
    
    # Return fallback response if all attempts fail
    return json.dumps({
        "explanation": "Error processing AI response after multiple attempts",
        "files_modified": [],
        "codebase_analysis": {
            "current_state": "Error state",
            "recommendations": [f"Failed to process response after {attempts} attempts"]
        }
    }, indent=2)

async def fix_json_plan(raw_plan: str, error_details: str = None) -> str:
    """
    Attempt to fix invalid JSON plan by asking the AI for a correction.
    
    Args:
        raw_plan: The invalid JSON string that needs fixing
        error_details: Optional error message explaining why the JSON was invalid
    
    Returns:
        str: A valid JSON string containing exactly the keys 'explanation', 'files_modified', and 'codebase_analysis'
    """
    error_context = f" Error: {error_details}" if error_details else ""
    
    message = {
        "role": "user",
        "content": f"""The following plan data failed validation.{error_context}
Please provide a corrected JSON object that STRICTLY follows this schema:

{{
    "explanation": "string describing the task and approach",
    "files_modified": [
        // Each item must be either a string or an object with "path" and "description"
        "path/to/file.ext",
        {{
            "path": "path/to/file.ext",
            "description": "why this file needs modification"
        }}
    ],
    "codebase_analysis": {{
        "current_state": "description of current implementation",
        "recommendations": [
            "list of specific recommendations"
        ]
    }}
}}

IMPORTANT:
1. The response MUST be valid JSON
2. ONLY these three top-level keys are allowed
3. The types must match exactly as shown
4. No additional keys are permitted

Raw content to fix:
{raw_plan}"""
    }
    
    try:
        response = await client.messages.create(
            max_tokens=1024,
            messages=[message],
            model="claude-3-5-sonnet-latest"
        )
        
        # Extract JSON from response
        content = response.content[0].text
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            fixed_json = json.loads(json_str)
            # Check that the fixed JSON contains exactly the required keys
            required_keys = {"explanation", "files_modified", "codebase_analysis"}
            if set(fixed_json.keys()) == required_keys:
                return json.dumps(fixed_json, indent=2)
        
        logging.error("AI response did not return valid JSON with exactly the required keys")
        return None
    except Exception as e:
        logging.error(f"Error fixing JSON plan: {str(e)}")
        return None

async def correct_recommended_changes(raw_changes: dict) -> dict:
    """
    Request corrected format for recommended changes from AI.
    
    Args:
        raw_changes: Dictionary containing the raw recommended changes
        
    Returns:
        dict: Corrected changes dictionary with proper format
    """
    message = {
        "role": "user",
        "content": f"""The recommended changes did not adhere to the expected JSON format. 
        Each change should be an object with keys 'location', 'suggestion', and 'benefit'.
        Please provide a corrected JSON object.
        
        Raw data: {json.dumps(raw_changes)}"""
    }
    
    try:
        response = await client.messages.create(
            max_tokens=1024,
            messages=[message],
            model="claude-3-5-sonnet-latest"
        )
        
        content = response.content[0].text
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            corrected = json.loads(json_str)
            
            # Verify the corrected format
            if isinstance(corrected, dict) and all(
                isinstance(change, dict) and 
                all(key in change for key in ['location', 'suggestion', 'benefit'])
                for change in corrected.values()
            ):
                return corrected
                
        logging.warning("AI response did not contain properly formatted changes")
        return raw_changes
    except Exception as e:
        logging.error(f"Error correcting recommended changes: {str(e)}")
        return raw_changes

async def format_recommended_changes(changes: dict) -> Table:
    """Format the recommended changes section as a table."""
    table = Table(title="Recommended Changes", show_header=True, header_style="bold magenta")
    table.add_column("Location", style="cyan")
    table.add_column("Suggestion", style="green")
    table.add_column("Benefit", style="yellow")
    
    if not changes:
        table.add_row("No changes", "No suggestions", "N/A")
        return table
    
    # Verify format and correct if needed
    needs_correction = False
    for key, change in changes.items():
        if not isinstance(change, dict) or not all(
            key in change for key in ['location', 'suggestion', 'benefit']
        ):
            needs_correction = True
            break
    
    if needs_correction:
        changes = await correct_recommended_changes(changes)
    
    # Add rows to table
    for key, change in changes.items():
        if isinstance(change, dict):
            location = change.get('location', 'Unknown location')
            suggestion = change.get('suggestion', 'No suggestion provided')
            benefit = change.get('benefit', 'No benefit specified')
            table.add_row(location, suggestion, benefit)
        else:
            # Handle case where change is still not a dictionary after correction
            table.add_row(str(key), str(change), "N/A")
    
    return table

def format_current_implementation(implementation: dict) -> Panel:
    """
    Format the current implementation section in a Panel.
    """
    from rich.panel import Panel  # Ensure Panel is imported
    if not implementation:
        return Panel("No implementation details provided", title="Current Implementation", border_style="blue")
    content = "\n".join(f"{key}: {value}" for key, value in implementation.items())
    return Panel(content, title="Current Implementation", border_style="blue")

async def display_json_data(json_data: dict):
    """Display the JSON data with rich formatting."""
    console.clear()
    console.print("\n[bold cyan]JSON Data Summary[/bold cyan]", justify="center")
    console.print("=" * 80, justify="center")
    
    # Display current implementation if available
    if current_impl := json_data.get('current_implementation'):
        console.print(format_current_implementation(current_impl))
        console.print()
    
    # Display files to be modified if available
    if files_modified := json_data.get('files_modified'):
        files_table = Table(title="Files to be Modified", show_header=True, header_style="bold magenta")
        files_table.add_column("File Path", style="cyan")
        files_table.add_column("Description", style="green")
        
        if isinstance(files_modified, list):
            for file in files_modified:
                if isinstance(file, str):
                    files_table.add_row(file, "")
                elif isinstance(file, dict):
                    files_table.add_row(
                        file.get('path', 'Unknown'),
                        file.get('description', 'No description provided')
                    )
        console.print(files_table)
        console.print()
    
    # Display recommended changes if available
    if recommended_changes := json_data.get('recommended_changes'):
        if isinstance(recommended_changes, dict) and recommended_changes:
            changes_table = await format_recommended_changes(recommended_changes)
            console.print(changes_table)
            console.print()
    
    console.print("\n[bold green]Please review the formatted JSON data.[/bold green]")


async def display_final_plan(plan: str):
    """Display the final plan as bullet points rather than raw JSON."""
    try:
        with Status("[bold blue]Formatting task plan...", console=console):
            # Convert the plan input into a dictionary.
            if isinstance(plan, list):
                if isinstance(plan[0], TextBlock):
                    plan_data = json.loads(plan[0].text)
                elif isinstance(plan[0], dict):
                    plan_data = plan[0]
                else:
                    plan_data = json.loads(plan[0])
            elif isinstance(plan, str):
                plan_data = json.loads(plan)
            elif isinstance(plan, dict):
                plan_data = plan
            else:
                raise TypeError("Plan must be a string, a list containing a string, or a dictionary.")

            console.clear()
            console.print("\n[bold cyan]Task Plan Summary[/bold cyan]", justify="center")
            console.print("=" * 80, justify="center")
            console.print("\n[white]Based on the code analysis, here are the key areas for optimization:[/white]\n")
            
            # Display the explanation as a panel.
            explanation = plan_data.get("explanation", "No explanation provided")
            console.print(Panel(explanation, title="Task Explanation", border_style="blue"))
            console.print()
            
            # Display files to be modified as a bullet list.
            files_modified = plan_data.get("files_modified", [])
            if files_modified:
                files_list = "\n".join(
                    f"- {f}" if isinstance(f, str)
                    else f"- {f.get('path', 'Unknown')}: {f.get('description', 'No description provided')}"
                    for f in files_modified
                )
            else:
                files_list = "No files to be modified."
            console.print(Panel(files_list, title="Files to be Modified", border_style="magenta"))
            console.print()

            # Improved dynamic formatting for recommended_changes
            if "recommended_changes" in plan_data:
                changes = plan_data["recommended_changes"]
                bullet_changes = []
                for category, items in changes.items():
                    bullet_changes.append(f"• [bold]{category}[/bold]:")
                    if isinstance(items, dict):
                        count = 1
                        for k, v in items.items():
                            bullet_changes.append(f"    {count}. {k}: {v}")
                            count += 1
                    elif isinstance(items, list):
                        for index, item in enumerate(items, start=1):
                            bullet_changes.append(f"    {index}. {item}")
                    else:
                        bullet_changes.append(f"    1. {items}")
                analysis_str = "\n".join(bullet_changes)
                console.print(Panel(analysis_str, title="Recommended Changes", border_style="green"))
            else:
                # Restored original codebase_analysis formatting
                analysis = plan_data.get("codebase_analysis", {})
                if isinstance(analysis, dict):
                    analysis_lines = []
                    for section, details in analysis.items():
                        analysis_lines.append(f"- {section}:")
                        if isinstance(details, list):
                            for detail in details:
                                analysis_lines.append(f"    • {detail}")
                        elif isinstance(details, dict):
                            for key, value in details.items():
                                analysis_lines.append(f"    • {key}: {value}")
                        else:
                            analysis_lines.append(f"    • {details}")
                    analysis_str = "\n".join(analysis_lines)
                elif isinstance(analysis, str):
                    analysis_str = analysis
                else:
                    analysis_str = "No codebase analysis available."
                console.print(Panel(analysis_str, title="Codebase Analysis", border_style="green"))
            
            # Format additional dynamic sections if available
            additional_keys = ["current_issues", "recommended_improvements", "implementation_approach"]
            for key in additional_keys:
                if key in plan_data:
                    content = plan_data[key]
                    section_lines = []
                    section_lines.append(f"- {key}:")
                    if isinstance(content, list):
                        for i, item in enumerate(content, start=1):
                            section_lines.append(f"    {i}. {item}")
                    elif isinstance(content, dict):
                        for i, (subkey, subvalue) in enumerate(content.items(), start=1):
                            section_lines.append(f"    {i}. {subkey}: {subvalue}")
                    else:
                        section_lines.append(f"    1. {content}")
                    section_str = "\n".join(section_lines)
                    console.print(Panel(section_str, title=key.replace('_', ' ').title(), border_style="cyan"))

            console.print("\n[bold green]Please review the plan and proceed with the necessary actions.[/bold green]")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Unexpected error in display_final_plan: {str(e)}\nTraceback:\n{tb}")
        console.print(Panel(
            f"[red]An unexpected error occurred:[/red]\n{str(e)}",
            title="Error",
            border_style="red"
        ))

async def main():
    from persistent_cache import init_persistent_cache  # Newly added import
    await init_persistent_cache()  # Initialize the persistent cache once the event loop is running
    task_description = input("Enter the task description: ")
    codebase_summary = await explore_codebase(task_description=task_description)
    console.print(f"[green]Found {len(codebase_summary)} relevant files in the codebase.[/green]")
    plan = await generate_task_plan(task_description, codebase_summary)
    valid_plan = await validate_ai_response(plan)
    await display_final_plan(valid_plan)

if __name__ == "__main__":
    asyncio.run(main())
