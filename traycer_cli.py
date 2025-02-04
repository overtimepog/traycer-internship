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

async def read_file_snippet(file_path, num_lines=10):
    """
    Asynchronously reads the first few lines of a file and returns them as a string.
    Opens files using errors='replace' so that decoding issues (e.g. in partially binary files)
    do not cause crashes.
    """
    lines = []
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for _ in range(num_lines):
                line = await f.readline()
                if not line:
                    break
                lines.append(line.strip())
    except Exception as e:
        return f"Error reading file: {str(e)}"
    return '\n'.join(lines)

async def read_full_file(file_path):
    """
    Asynchronously reads the entire content of a file and returns it as a string.
    Opens files using errors='replace' so that decoding issues (e.g. in partially binary files)
    do not cause crashes.
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = await f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

async def process_file(file_path, code_extensions, text_extensions, semaphore):
    """
    Process a single file by gathering metadata (size, last modified, etc.) and,
    if applicable, reading its full contents.
    """
    async with semaphore:
        try:
            file_size = await asyncio.to_thread(os.path.getsize, file_path)
            # Normalize the file extension to lowercase for consistency.
            file_extension = os.path.splitext(file_path)[1].lower()
            last_modified = await asyncio.to_thread(os.path.getmtime, file_path)
            
            # Determine importance:
            #   - 'high' for known code files
            #   - 'medium' for other text-based files
            #   - 'low' for everything else (e.g. binary files)
            if file_extension in code_extensions:
                importance = 'high'
            elif file_extension in text_extensions:
                importance = 'medium'
            else:
                importance = 'low'
            
            # Read the full content only for text-based files.
            content = await read_full_file(file_path) if file_extension in text_extensions else ''
            
            return {
                'path': file_path,
                'size': file_size,
                'extension': file_extension,
                'importance': importance,
                'last_modified': last_modified,
                'content': content
            }
        except Exception as e:
            return {'path': file_path, 'error': str(e)}

file_cache = {}

async def explore_codebase(root_dir='.'):
    """
    Explores and indexes the entire file tree starting at root_dir. For each file,
    it collects metadata and, if the file is text-based (e.g. code, JSON, TXT, etc.),
    reads its full contents. The traversal ignores some specified directories.
    """
    # Directories to ignore during the walk.
    ignored_directories = {'.git', 'node_modules', '__pycache__', 'venv', '.pytest_cache'}
    
    # Define file type sets.
    code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
    # Extend text_extensions to include any file type that should have its content read.
    text_extensions = code_extensions.union({'.json', '.txt', '.md', '.html', '.css', '.xml', '.csv'})
    
    tasks = []
    # Limit the number of concurrent file processing tasks to avoid overloading the system.
    semaphore = asyncio.Semaphore(100)
    
    for root, dirs, files in os.walk(root_dir):
        # Skip any ignored directories.
        dirs[:] = [d for d in dirs if d not in ignored_directories]
        for file in files:
            file_path = os.path.join(root, file)
            if file_path not in file_cache:
                tasks.append(asyncio.create_task(
                    process_file(file_path, code_extensions, text_extensions, semaphore)
                ))
            else:
                print(f"Using cached data for {file_path}")
    
    # Run all file processing tasks concurrently.
    codebase_summary = await asyncio.gather(*tasks)
    # Update cache with new results
    for file_summary in codebase_summary:
        file_cache[file_summary['path']] = file_summary
    return codebase_summary

async def ai_based_relevance(file_summary, task_description):
    """
    Use AI to determine the relevance of a file based on its content and the task description.
    Returns a relevance rating of 'high', 'medium', or 'low'.
    """
    message = {
        "role": "user",
        "content": f"""Task: {task_description}

Snippet:
{file_summary.get('content', '')[:200]}  # Limit to first 200 characters

Please determine the relevance of this file to the task. Respond with one of "high", "medium", or "low"."""
    }
    response = await client.messages.create(
        max_tokens=50,
        messages=[message],
        model="claude-3-5-sonnet-latest"
    )
    relevance = response.content[0].text.strip().lower()
    return 'medium' if relevance not in ['high', 'low'] else relevance

async def generate_task_plan(task_description, codebase_summary):
    """
    Generate a task plan based on the user's description, incorporating file relevance ratings.
    """
    # Determine AI-based relevance for each file
    relevance_tasks = [ai_based_relevance(file, task_description) for file in codebase_summary]
    relevance_results = await asyncio.gather(*relevance_tasks)

    # Format the codebase summary into a more readable structure with AI-based relevance
    formatted_summary = []
    for file, relevance in zip(codebase_summary, relevance_results):
        summary = {
            'path': file['path'],
            'importance': file['importance'],
            'relevance': relevance
        }
        if 'functions' in file:
            summary['functions'] = file['functions']
        formatted_summary.append(summary)

    # Create a structured prompt with the enhanced codebase information
    message = {
        "role": "user",
        "content": f"""Task: {task_description}

Codebase Analysis:
- Highly Relevant Files: {[f['path'] for f in formatted_summary if f['relevance'] == 'high']}
- Moderately Relevant Files: {[f['path'] for f in formatted_summary if f['relevance'] == 'medium']}

Detailed File Information:
{formatted_summary}

Please provide a JSON response with the keys:
- explanation (a string)
- files_modified (a list of file paths)
- codebase_analysis (a string)

Ensure the response is valid JSON."""
    }
    response = await client.messages.create(
        max_tokens=1024,
        messages=[message],
        model="claude-3-5-sonnet-latest"
    )
    return response.content[0].text

async def validate_ai_response(response: str):
    """
    Function to validate the AI response and request corrections if needed.
    Ensures the response properly utilizes the codebase context and includes required sections.
    """
    try:
        response_json = json.loads(response)
        if all(key in response_json for key in ["explanation", "files_modified", "codebase_analysis"]):
            return response
    except json.JSONDecodeError:
        pass

    correction_message = {
        "role": "user",
        "content": """Your response is not valid JSON or is missing required keys.
        
Please provide a complete JSON response with the keys:
- explanation
- files_modified
- codebase_analysis"""
    }
    corrected_response = await client.messages.create(
        max_tokens=1024,
        messages=[correction_message],
        model="claude-3-5-sonnet-latest"
    )
    return corrected_response.content[0].text

async def display_final_plan(plan: str):
    """Function to display the final verified plan to the user"""
    print("\n--- Task Plan ---")
    print(plan)
    print("\nPlease review the plan and proceed with the necessary actions.")

async def main():
    # Main function to run the CLI tool
    task_description = input("Enter the task description: ")
    codebase_summary = await explore_codebase()
    print(f"Explored {len(codebase_summary)} files in the codebase.")
    plan = await generate_task_plan(task_description, codebase_summary)
    valid_plan = await validate_ai_response(plan)
    await display_final_plan(valid_plan)

if __name__ == "__main__":
    asyncio.run(main())
