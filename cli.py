import os
import logging
import asyncio
import re
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
import json
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
    Quickly scan a file for relevant keywords and return all matching snippets.
    Returns a list of dictionaries containing:
    - line_range: string showing the range of lines included in the snippet (e.g. "205 - 209")
    - context: the actual text content from those lines
    
    Each match includes 2 lines before and 2 lines after for context. All matches in the file
    are returned, with no limit on the number of snippets.
    """
    snippets = []
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = await f.readlines()
            
        for i, line in enumerate(content):
            if any(keyword.lower() in line.lower() for keyword in keywords):
                start = max(0, i - 2)
                end = min(len(content), i + 3)
                context = ''.join(content[start:end])
                # Convert to 1-based line numbers for display
                start_line = start + 1
                end_line = end
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
    Process a single file by gathering metadata and scanning for relevant content.
    """
    async with semaphore:
        try:
            file_size = await asyncio.to_thread(os.path.getsize, file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            last_modified = await asyncio.to_thread(os.path.getmtime, file_path)
            
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

file_cache = {}

async def explore_codebase(root_dir='.', task_description=''):
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
            
            if cache_key in file_cache:
                tasks.append(asyncio.create_task(
                    asyncio.sleep(0, result=file_cache[cache_key])
                ))
            else:
                tasks.append(asyncio.create_task(
                    process_file(file_path, code_extensions, text_extensions, keywords, semaphore)
                ))
    
    codebase_summary = await asyncio.gather(*tasks)
    
    # Update cache with new results
    for file_summary in codebase_summary:
        if 'error' not in file_summary:
            cache_key = f"{file_summary['path']}:{file_summary['last_modified']}"
            file_cache[cache_key] = file_summary
    
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

Please provide a JSON response with:
- explanation: Brief task explanation
- files_modified: List of files that need modification
- codebase_analysis: Analysis of relevant code snippets"""
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

async def validate_ai_response(response: str):
    """Validate and format AI response."""
    try:
        response_json = json.loads(response)
        if all(key in response_json for key in ["explanation", "files_modified", "codebase_analysis"]):
            return json.dumps(response_json, indent=2)
    except json.JSONDecodeError:
        pass
    
    correction_message = {
        "role": "user",
        "content": "Please provide valid JSON with explanation, files_modified, and codebase_analysis keys."
    }
    corrected_response = await client.messages.create(
        max_tokens=1024,
        messages=[correction_message],
        model="claude-3-5-sonnet-latest"
    )
    return corrected_response.content

async def fix_json_plan(raw_plan: str, error_details: str = None) -> str:
    """
    Attempt to fix invalid JSON plan by asking the AI for a correction.
    
    Args:
        raw_plan: The invalid JSON string that needs fixing
        error_details: Optional error message explaining why the JSON was invalid
    
    Returns:
        str: A valid JSON string containing the required keys
    """
    error_context = f" Error: {error_details}" if error_details else ""
    
    message = {
        "role": "user",
        "content": f"""The following plan data failed JSON parsing.{error_context}
        Please fix and return a valid JSON object containing 'explanation', 'files_modified', and 'codebase_analysis' keys.
        
        Raw content:
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
            # Validate the fixed JSON has required keys
            fixed_json = json.loads(json_str)
            if all(key in fixed_json for key in ["explanation", "files_modified", "codebase_analysis"]):
                return json.dumps(fixed_json, indent=2)
        
        logging.error("AI response did not contain valid JSON with required keys")
        return None
    except Exception as e:
        logging.error(f"Error fixing JSON plan: {str(e)}")
        return None

def format_current_implementation(implementation: dict) -> Panel:
    """Format the current implementation section in a panel."""
    if not implementation:
        return Panel("No current implementation details available",
                    title="Current Implementation",
                    border_style="blue")
    
    try:
        # Handle both string and dictionary values
        content_lines = []
        for key, value in implementation.items():
            if isinstance(value, dict):
                # Format nested dictionary
                nested_content = "\n  ".join(f"{k}: {v}" for k, v in value.items())
                content_lines.append(f"{key}:\n  {nested_content}")
            else:
                content_lines.append(f"{key}: {value}")
        
        content = "\n".join(content_lines)
        return Panel(content, title="Current Implementation", border_style="blue")
    except Exception as e:
        return Panel(f"Error formatting implementation: {str(e)}",
                    title="Current Implementation",
                    border_style="red")

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
    """Display the final plan with rich formatting."""
    try:
        with Status("[bold blue]Formatting task plan...", console=console):
            # Parse the plan JSON
            
            #this needs to convert the list its in to a dictionary
            if isinstance(plan, list):
                if isinstance(plan[0], TextBlock):
                    plan_data = json.loads(plan[0].text)
                elif isinstance(plan[0], dict):
                    plan_data = plan[0]  # Directly use the dictionary if it's already parsed
                else:
                    plan_data = json.loads(plan[0])
            elif isinstance(plan, str):
                plan_data = json.loads(plan)
            elif isinstance(plan, dict):
                plan_data = plan  # Directly use the dictionary if it's already parsed
            else:
                raise TypeError("Plan must be a string, a list containing a string, or a dictionary.")
            
            # Clear the screen for better presentation
            console.clear()
            
            # Display title
            console.print("\n[bold cyan]Task Plan Summary[/bold cyan]", justify="center")
            console.print("=" * 80, justify="center")
            
            # Display each section
            if isinstance(plan_data, dict):
                console.print(Panel(plan_data.get('explanation', 'No explanation provided'), title="Task Explanation", border_style="blue"))
                console.print()
                
                console.print(Table(title="Files to be Modified", show_header=True, header_style="bold magenta"))
                console.print()
                
                codebase_analysis = plan_data.get('codebase_analysis', '{}')
                if isinstance(codebase_analysis, dict):
                    codebase_analysis = json.dumps(codebase_analysis, indent=2)
                console.print(Syntax(codebase_analysis, "json", theme="native", word_wrap=True, background_color=None))
            else:
                logging.error("plan_data is not a dictionary")
            
            console.print("\n[bold green]Please review the plan and proceed with the necessary actions.[/bold green]")
    
    except json.JSONDecodeError as e:
        # Attempt to fix invalid JSON using AI
        logging.warning(f"Invalid JSON plan detected: {str(e)}")
        console.print("[yellow]Attempting to fix invalid JSON plan...[/yellow]")
        
        fixed_plan = await fix_json_plan(str(plan), str(e))
        if fixed_plan:
            # Retry displaying with fixed plan
            plan_data = json.loads(fixed_plan)
            console.print("[green]Successfully fixed and parsed JSON plan[/green]")
            
            console.print(Panel(plan_data.get('explanation', 'No explanation provided'), title="Task Explanation", border_style="blue"))
            console.print()
            console.print(Table(title="Files to be Modified", show_header=True, header_style="bold magenta"))
            console.print()
            codebase_analysis = plan_data.get('codebase_analysis', '{}')
            if isinstance(codebase_analysis, dict):
                codebase_analysis = json.dumps(codebase_analysis, indent=2)
            console.print(Syntax(codebase_analysis, "json", theme="native", word_wrap=True, background_color=None))
        else:
            # Fallback to error display if fix failed
            logging.error("Failed to fix invalid JSON plan")
            console.print(Panel(
                "[red]Error: Invalid plan format and automatic fix failed[/red]\n\nRaw plan content:\n" + str(plan),
                title="Error",
                border_style="red"
            ))
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
    task_description = input("Enter the task description: ")
    codebase_summary = await explore_codebase(task_description=task_description)
    console.print(f"[green]Found {len(codebase_summary)} relevant files in the codebase.[/green]")
    plan = await generate_task_plan(task_description, codebase_summary)
    valid_plan = await validate_ai_response(plan)
    await display_final_plan(valid_plan)

if __name__ == "__main__":
    asyncio.run(main())
