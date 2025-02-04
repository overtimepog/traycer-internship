import os
import asyncio
import re
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
import json

# Initialize the Anthropic client with the API key from the environment
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
    relevant_files = [f for f in codebase_summary if f.get('snippets', [])]
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
        print(f"Received response: {response.content[0].text}")
        
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
            print(f"Error processing response: {str(e)}")
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
    filtered_results = [r for r in relevance_results if r.get('relevance') == 'high']
    
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

async def display_final_plan(plan: str):
    """Display the final plan."""
    print("\n--- Task Plan ---")
    print(plan)
    print("\nPlease review the plan and proceed with the necessary actions.")

async def main():
    task_description = input("Enter the task description: ")
    codebase_summary = await explore_codebase(task_description=task_description)
    print(f"Found {len(codebase_summary)} relevant files in the codebase.")
    plan = await generate_task_plan(task_description, codebase_summary)
    valid_plan = await validate_ai_response(plan)
    await display_final_plan(valid_plan)

if __name__ == "__main__":
    asyncio.run(main())
