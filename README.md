# Codebase Explorer and Task Plan Generator

## Overview

This project is a powerful, asynchronous tool designed to analyze a codebase, identify key files and code snippets through keyword scanning, and generate a detailed task plan with actionable recommendations. It leverages advanced file processing, persistent caching, and AI-driven analysis to assist developers in optimizing and improving their codebases.

## Key Features

- **Asynchronous File Processing**: Utilizes asynchronous I/O via aiofiles to efficiently process large codebases without blocking.
- **Keyword Scanning & Code Analysis**: Scans file contents for relevant keywords derived from a user-provided task description, extracting useful context snippets.
- **Persistent Caching**: Implements a persistent caching mechanism using SQLite (via aiosqlite), with an LRU eviction policy to manage cache size and improve performance.
- **AI-Driven Task Planning**: Integrates with the Anthropic AI API (using AsyncAnthropic) to produce comprehensive task plans that include recommendations and potential code modifications.
- **Rich CLI Interface**: Employs the Rich library to render formatted tables, panels, and syntax-highlighted text in the terminal for an enhanced user experience.
- **Robust Error Handling**: Includes detailed logging and error handling to ensure reliability during file processing and API interactions.

## Architecture & Components

### 1. Codebase Analysis (codebase.py)

- **File Reading and Snippet Extraction**:
  - `read_file_snippet`: Asynchronously reads a snippet of lines from a file starting at a given line number.
  - `read_file_content`: Reads the entirety (or a truncated version) of a file with a configurable maximum size.

- **Content Scanning**:
  - `scan_file_content`: Scans file content for specific keywords, returning context snippets along with the line ranges where they occur.

- **File Processing**:
  - `process_file`: Processes individual files, applying filters for binary or excessively large files, and extracts contextual snippets if the file is relevant.

- **Codebase Exploration**:
  - `explore_codebase`: Recursively traverses the directory structure, processes each file, applies caching to preserve results, and aggregates a summary of files that match the task criteria.

### 2. CLI Interface & Task Planning (cli.py)

- **API Key Management**:
  - Retrieves the Anthropic API key from environment variables or a `.env` file to authenticate API requests.

- **User Interaction**:
  - Prompts the user for a task description, which guides the codebase exploration and analysis process.

- **File Analysis & Task Generation**:
  - Invokes `explore_codebase` to assess and index relevant files.
  - Bundles file data and sends it to the Anthropic API to generate a detailed task plan containing explanations, recommended modifications, and a codebase analysis.

- **Task Plan Formatting & Display**:
  - The AI-generated plan is validated against a strict JSON schema and formatted into three core sections:
    - **explanation**: A brief overview of the task and approach.
    - **files_modified**: Detailed suggestions for file modifications, including line ranges, actions, and descriptions.
    - **codebase_analysis**: An assessment of the current codebase state along with targeted recommendations.
  - Utilizes Rich components (Tables, Panels, etc.) for a polished terminal display.

- **Retry Logic & Validation**:
  - Implements retry and correction mechanisms to ensure the AI response adheres to the expected JSON format.

### 3. Persistent Caching (persistent_cache.py)

- **SQLite-Based Cache**:
  - Initializes a SQLite database `cache.db` to store file analysis results, ensuring that frequently accessed data is quickly retrievable.

- **Cache Operations**:
  - **get_cache**: Retrieves cached data based on a unique key (generated from the file path and modified time).
  - **set_cache**: Stores processed file data in the cache, serializing it into JSON and managing cache size.

- **LRU Eviction Strategy**:
  - Enforces a maximum cache size and evicts the least recently used entries when necessary, using an evict batch strategy.

## Getting Started

### Prerequisites

- **Python Version**: Python 3.7 or higher
- **Required Python Packages**:
  - aiofiles
  - aiosqlite
  - rich
  - python-dotenv
  - anthropic
  - asyncio
  - logging
  - re

Install the necessary dependencies via pip:

```
pip install aiofiles aiosqlite rich python-dotenv anthropic
```

### Configuration

1. **Environment Variables**:
   - Create a `.env` file in the project root with the following content to set up your Anthropic API key:

   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

2. **Cache Setup**:
   - The persistent caching mechanism will automatically generate a `cache.db` file in the project directory. Ensure the directory has write permissions.

## Usage

- Run the CLI application using the following command:

```
python cli.py
```

- Upon execution, the application will prompt you to enter a task description. This description guides the analysis of your codebase. The steps include:
  - Scanning the codebase for relevant files based on keywords extracted from your task description.
  - Asynchronously processing each file and caching results to improve performance.
  - Bundling file data and sending it to the Anthropic AI API for a detailed task plan.
  - Displaying the final, formatted task plan with actionable recommendations directly in your terminal.

## Project Flow

1. **Task Input**: The user provides a task description.
2. **Codebase Exploration**: The tool scans and processes files across the codebase asynchronously, using caching to reduce redundant work.
3. **AI Analysis**: Relevant file data is sent to the Anthropic API, which returns a comprehensive task plan with recommendations.
4. **Plan Presentation**: The validated task plan is displayed using Rich formatting, outlining modifications and providing a codebase analysis.

## Contributing

Contributions are welcome! To contribute:

- Fork the repository.
- Create a feature branch for your changes.
- Commit your modifications with clear, concise messages.
- Open a pull request detailing your changes and the problem they address.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Thanks to the Anthropic team for their AI API, which powers the task planning functionality.
- Appreciation to the developers of aiofiles, aiosqlite, and Rich for providing high-quality tools that are integral to this project. 