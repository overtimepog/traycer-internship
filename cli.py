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
from codebase import (
    explore_codebase,
    read_file_content,
    read_file_snippet,
    scan_file_content,
    process_file
)

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def get_api_key():
    """Get the Anthropic API key from environment variables or .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Load environment variables from .env file
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[red]Error: ANTHROPIC_API_KEY not found in environment variables or .env file[/red]")
            console.print("[yellow]Please set your Anthropic API key in a .env file or as an environment variable.[/yellow]")
            console.print("You can create a .env file with the following content:")
            console.print("[green]ANTHROPIC_API_KEY=your-api-key-here[/green]")
            raise ValueError("Missing ANTHROPIC_API_KEY")
        return api_key
    except ImportError:
        console.print("[yellow]python-dotenv not installed. Installing...[/yellow]")
        import subprocess
        subprocess.check_call(["pip", "install", "python-dotenv"])
        from dotenv import load_dotenv
        load_dotenv()
        return get_api_key()

try:
    client = AsyncAnthropic(
        api_key=get_api_key()
    )
except Exception as e:
    console.print(f"[red]Failed to initialize Anthropic client: {str(e)}[/red]")
    raise

def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        print(f"{func.__name__} called {wrapper.calls} times this session.")
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

import aiofiles

async def process_file(file_path, code_extensions, text_extensions, keywords, semaphore, is_target_file=False):
    """
    Process a single file by gathering metadata and scanning for relevant content.
    Implements early filtering for binary and large files.
    """
    async with semaphore:
        try:
            file_size = await asyncio.to_thread(os.path.getsize, file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            last_modified = await asyncio.to_thread(os.path.getmtime, file_path)
            
            # Early filtering for large files (>10MB) or binary content
            MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
            # TODO: figure out some way to shrink large files so they can be sent as context to the ai model
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
                        'importance': 'high' if is_target_file else 'low',
                        'last_modified': last_modified,
                        'snippets': [],
                        'skip_reason': 'Binary file'
                    }
            
            # Determine importance based on file type and target status
            if is_target_file:
                importance = 'high'
            elif file_extension in code_extensions:
                importance = 'medium'
            elif file_extension in text_extensions:
                importance = 'low'
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

async def scan_file_content(file_path, keywords):
    """
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

async def batch_relevance_check(files, task_description):
    """
    Check relevance of multiple files in a single AI call to reduce API usage.
    """
    if not files:
        return []
    
    file_summaries = []
    for i in range(0, len(files), 5):  # Process in batches of 5
        batch = files[i:i+5]
        # Add full file content as a separate step to avoid f-string issues
        batch_data = []
        for f in batch:
            full_content = await read_file_content(f['path'])
            batch_data.append({
                'path': f['path'],
                'snippets': f['snippets'],
                'full_content': full_content
            })
            
        message = {
            "role": "user",
            "content": f"""Task: {task_description}

Files to analyze:
{json.dumps(batch_data, indent=2)}

For each file, respond with a JSON array of objects containing:
- path: file path
- relevance: "high", "medium", or "low"
- needs_more_context: (optional) if you need more context, specify the line numbers or areas you'd like to see
"""
        }
        
        response = await client.messages.create(
            max_tokens=4096,  # Increased token limit for larger responses
            messages=[message],
            model="claude-3-5-sonnet-latest"
        )
        
        # Handle different response content types
        try:
            if isinstance(response.content, TextBlock):
                content_str = str(response.content[0].text)
                print(f"Received response: {content_str}")
                json_start = content_str.find('[')
                json_end = content_str.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_str = content_str[json_start:json_end]
                    parsed_content = json.loads(json_str)
                    
                    # Handle requests for more context
                    for item in parsed_content:
                        if item.get('needs_more_context'):
                            additional_context = await get_additional_context(
                                item['path'], 
                                item['needs_more_context']
                            )
                            # Make another API call with additional context
                            item['relevance'] = await get_relevance_with_context(
                                item['path'],
                                additional_context,
                                task_description
                            )
                    
                    file_summaries.extend(parsed_content)
                else:
                    file_summaries.extend([{'path': f['path'], 'relevance': 'medium'} for f in batch])
            else:
                file_summaries.extend([{'path': f['path'], 'relevance': 'medium'} for f in batch])
        except Exception as e:
            logging.error(f"Error processing response: {str(e)}")
            console.print(f"[red]Error processing response: {str(e)}[/red]")
            file_summaries.extend([{'path': f['path'], 'relevance': 'medium'} for f in batch])
    
    return file_summaries

async def get_additional_context(file_path: str, context_request: str) -> dict:
    """
    Get additional context from a file based on the AI's request.
    
    Args:
        file_path: Path to the file
        context_request: String describing what context is needed (e.g., "lines 50-100" or "function X")
    
    Returns:
        dict: Contains the requested context and metadata
    """
    try:
        if 'lines' in context_request.lower():
            # Extract line numbers from request
            matches = re.findall(r'lines?\s*(\d+)(?:\s*-\s*(\d+))?', context_request.lower())
            if matches:
                start = int(matches[0][0])
                end = int(matches[0][1]) if matches[0][1] else start + 20
                content = await read_file_snippet(file_path, start - 1, end - start + 1)
                return {
                    'type': 'lines',
                    'range': f"{start}-{end}",
                    'content': content
                }
        else:
            # If no specific lines requested, provide more surrounding context
            content = await read_file_content(file_path)
            return {
                'type': 'full_file',
                'content': content
            }
    except Exception as e:
        logging.error(f"Error getting additional context: {str(e)}")
        return {
            'type': 'error',
            'error': str(e)
        }

async def get_relevance_with_context(file_path: str, context: dict, task_description: str) -> str:
    """
    Make another API call to determine relevance with additional context.
    """
    message = {
        "role": "user",
        "content": f"""Task: {task_description}

Additional context requested for {file_path}:
{json.dumps(context, indent=2)}

Please analyze this additional context and provide a final relevance rating ("high", "medium", or "low").
"""
    }
    
    try:
        response = await client.messages.create(
            max_tokens=1024,
            messages=[message],
            model="claude-3-5-sonnet-latest"
        )
        
        content = response.content[0].text.lower()
        if 'high' in content:
            return 'high'
        elif 'medium' in content:
            return 'medium'
        else:
            return 'low'
    except Exception as e:
        logging.error(f"Error getting relevance with context: {str(e)}")
        return 'medium'

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

Please provide a JSON object that strictly adheres to the following format. The JSON must have exactly three top-level keys: 'explanation', 'files_modified', and 'codebase_analysis'.

- 'explanation' should be a string that briefly describes the task and approach.
- 'files_modified' must be an array where each element is an object with the following keys:
    - 'path': the file path
    - 'changes': an array of change objects, each with:
        - 'line_range': a string indicating the line numbers to be modified (e.g., "503-506")
        - 'action': a string describing the type of modification ("Replace", "Rewrite", "Remove", etc.)
        - 'description': a string explaining what change should be made and why
        - 'code': (optional) the actual code changes to be made, if applicable
- 'codebase_analysis' should be an object containing:
    - 'current_state': a string describing the current implementation
    - 'recommendations': an array of specific recommendations for improvement

No additional keys are allowed in the JSON response. The response must be valid JSON."""
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
                "type": dict,
                "required_keys": ["path", "changes"],
                "changes_schema": {
                    "type": list,
                    "items": {
                        "type": dict,
                        "required_keys": ["line_range", "action", "description"]
                    }
                }
            }
        },
        "codebase_analysis": {
            "type": dict,
            "required": True,
            "required_keys": ["current_state", "recommendations"]
        }
    }
    
    # Validate top-level structure
    for key, schema in required_schema.items():
        if schema["required"] and key not in data:
            return False
            
        if key in data:
            value = data[key]
            if not isinstance(value, schema["type"]):
                return False
                
            # Validate files_modified array
            if key == "files_modified":
                for file_mod in value:
                    if not isinstance(file_mod, dict):
                        return False
                    if not all(k in file_mod for k in schema["items"]["required_keys"]):
                        return False
                    # Validate changes array
                    changes = file_mod.get("changes", [])
                    if not isinstance(changes, list):
                        return False
                    for change in changes:
                        if not isinstance(change, dict):
                            return False
                        if not all(k in change for k in schema["items"]["changes_schema"]["items"]["required_keys"]):
                            return False
                            
            # Validate codebase_analysis structure
            elif key == "codebase_analysis":
                if not all(k in value for k in schema["required_keys"]):
                    return False
                if not isinstance(value["recommendations"], (list, str)):
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
        // Each item must be either a string or an object with "path" and "changes"
        "path/to/file.ext",
        {{
            "path": "path/to/file.ext",
            "changes": [
                {{
                    "line_range": "line_range",
                    "action": "action",
                    "description": "why this file needs modification"
                }}
            ]
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
            
            # Display files to be modified with their changes
            files_modified = plan_data.get("files_modified", [])
            if files_modified:
                files_table = Table(title="Files to be Modified", show_header=True, header_style="bold magenta")
                files_table.add_column("File Path", style="cyan")
                files_table.add_column("Changes", style="green")
                
                for file in files_modified:
                    if isinstance(file, dict):
                        path = file.get('path', 'Unknown')
                        changes = file.get('changes', [])
                        changes_text = ""
                        for change in changes:
                            changes_text += f"• Lines {change.get('line_range', 'N/A')}: "
                            changes_text += f"{change.get('action', 'N/A')} - "
                            changes_text += f"{change.get('description', 'No description')}\n"
                        files_table.add_row(path, changes_text.strip())
                    else:
                        files_table.add_row(str(file), "No changes specified")
                
                console.print(files_table)
                console.print()

            # Display codebase analysis
            analysis = plan_data.get("codebase_analysis", {})
            if isinstance(analysis, dict):
                current_state = analysis.get('current_state', 'No current state information available')
                recommendations = analysis.get('recommendations', [])
                
                console.print(Panel(current_state, title="Current State", border_style="yellow"))
                console.print()
                
                if recommendations:
                    rec_table = Table(title="Recommendations", show_header=False, box=None)
                    rec_table.add_column("", style="green")
                    for rec in recommendations:
                        rec_table.add_row(f"• {rec}")
                    console.print(rec_table)
            else:
                console.print(Panel(str(analysis), title="Codebase Analysis", border_style="yellow"))

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
    from persistent_cache import init_persistent_cache, get_cache, set_cache
    await init_persistent_cache()
    task_description = input("Enter the task description: ")
    codebase_summary = await explore_codebase(
        task_description=task_description,
        get_cache=get_cache,
        set_cache=set_cache
    )
    console.print(f"[green]Found {len(codebase_summary)} relevant files in the codebase.[/green]")
    plan = await generate_task_plan(task_description, codebase_summary)
    valid_plan = await validate_ai_response(plan)
    await display_final_plan(valid_plan)

if __name__ == "__main__":
    asyncio.run(main())
