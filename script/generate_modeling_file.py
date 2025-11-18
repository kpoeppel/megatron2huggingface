#!/usr/bin/env python3
"""Script to auto-generate modeling_megatron.py from modeling_megatron.py and its modeling module.

This script combines the main modeling file with all its modular components
into a single self-contained file for easier distribution and deployment.

Usage:
    # Generate to default location (repo root)
    python script/generate_modeling_file.py
    
    # Generate to specific directory
    python script/generate_modeling_file.py /path/to/output/dir
    
    # From Python code
    from script.generate_modeling_file import generate_modeling_auto
    generate_modeling_auto(output_dir="/path/to/output/dir")

Or with the correct module environment:
    module use /appl/local/csc/modulefiles && module load pytorch
    python script/generate_modeling_file.py [output_dir]

The script will:
    1. Read src/megatron2huggingface/modeling_megatron.py
    2. Read all files from src/megatron2huggingface/modeling/
    3. Combine them into a single modeling_megatron.py file
    4. Remove duplicate imports and relative imports
    5. Order classes properly (base classes first, then model classes)
    6. Add a header comment indicating the file is auto-generated

Output:
    modeling_megatron.py - A single-file version ready for distribution
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Tuple, List, Optional


def read_file(filepath: Path) -> str:
    """Read a file and return its contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(filepath: Path, content: str) -> None:
    """Write content to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def extract_module_docstring(content: str) -> Tuple[str, str]:
    """Extract the module-level docstring from content.
    
    Returns:
        Tuple of (docstring, remaining content)
    """
    lines = content.split('\n')
    
    # Check if first non-empty line is a docstring
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote_type = '"""' if stripped.startswith('"""') else "'''"
            # Single-line docstring
            if stripped.endswith(quote_type) and len(stripped) > 6:
                docstring = '\n'.join(lines[:i+1])
                remaining = '\n'.join(lines[i+1:])
                return docstring, remaining
            # Multi-line docstring
            for j in range(i+1, len(lines)):
                if quote_type in lines[j]:
                    docstring = '\n'.join(lines[:j+1])
                    remaining = '\n'.join(lines[j+1:])
                    return docstring, remaining
        break
    
    return '', content


def extract_imports_and_code(content: str) -> Tuple[List[str], str]:
    """Extract imports and remaining code from a file.
    
    Handles multiline imports properly.
    
    Returns:
        Tuple of (list of import blocks, remaining code without imports)
    """
    lines = content.split('\n')
    import_blocks = []
    code_lines = []
    in_import = False
    current_import = []
    past_imports = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines and comments before we hit imports
        if not past_imports and (not stripped or stripped.startswith('#')):
            continue
        
        # Check if this is the start of an import
        if stripped.startswith(('import ', 'from ')):
            past_imports = True
            in_import = True
            current_import = [line]
            # Check if it's a multiline import
            if '(' in line and ')' not in line:
                continue
            elif ')' not in line:
                # Single line import
                import_blocks.append(line)
                in_import = False
                current_import = []
            continue
        
        # If we're in a multiline import, continue collecting lines
        if in_import:
            current_import.append(line)
            if ')' in line:
                # End of multiline import
                import_blocks.append('\n'.join(current_import))
                in_import = False
                current_import = []
            continue
        
        # If we've seen imports and this is not an import, it's code
        if past_imports:
            code_lines.append(line)
        
    return import_blocks, '\n'.join(code_lines)


def normalize_import(imp: str) -> str:
    """Normalize an import for comparison (remove extra whitespace)."""
    # For multiline imports, collapse to single line for comparison
    normalized = ' '.join(imp.split())
    return normalized


def remove_relative_imports(content: str) -> str:
    """Remove relative imports from the modeling submodules."""
    lines = content.split('\n')
    filtered_lines = []
    in_import = False
    
    for line in lines:
        stripped = line.strip()
        
        # Check if this is a relative import we should skip
        if stripped.startswith('from .'):
            # Start of a potentially multiline relative import
            if '(' in stripped and ')' not in stripped:
                in_import = True
                continue
            else:
                # Single line relative import, skip it
                continue
        
        # If we're in a multiline relative import, skip until we find the end
        if in_import:
            if ')' in stripped:
                in_import = False
            continue
        
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def process_modeling_module(modeling_dir: Path, verbose: bool = True) -> Tuple[List[str], str]:
    """Process all files in the modeling module.
    
    Args:
        modeling_dir: Path to the modeling module directory
        verbose: Whether to print progress messages
    
    Returns:
        Tuple of (combined imports, combined code)
    """
    # Define the order of files to maintain dependencies
    module_files = [
        'layer_norm.py',
        'embeddings.py', 
        'attention.py',
        'mlp.py',
        'moemlp.py',
    ]
    
    all_imports = []
    all_code_sections = []
    seen_imports = set()
    
    for filename in module_files:
        filepath = modeling_dir / filename
        if not filepath.exists():
            if verbose:
                print(f"Warning: {filepath} not found, skipping...")
            continue
        
        if verbose:
            print(f"Processing {filename}...")
        content = read_file(filepath)
        
        # Remove docstring
        _, content = extract_module_docstring(content)
        
        # Remove relative imports
        content = remove_relative_imports(content)
        
        # Extract imports and code
        imports, code = extract_imports_and_code(content)
        
        # Add imports (avoiding duplicates)
        for imp in imports:
            normalized = normalize_import(imp)
            if normalized not in seen_imports:
                all_imports.append(imp)
                seen_imports.add(normalized)
        
        # Add separator comment and code
        separator = f"\n# {'=' * 78}\n# {filename.replace('.py', '').upper()}\n# {'=' * 78}\n"
        all_code_sections.append(separator + code.strip())
    
    return all_imports, '\n'.join(all_code_sections)


def process_main_file(main_file: Path, modeling_imports: List[str], modeling_code: str, verbose: bool = True) -> str:
    """Process the main modeling_megatron.py file and combine with modeling module.
    
    Args:
        main_file: Path to the main modeling file
        modeling_imports: List of imports from modeling modules
        modeling_code: Combined code from modeling modules
        verbose: Whether to print progress messages
    
    Returns:
        Complete combined file content
    """
    if verbose:
        print(f"Processing main file: {main_file.name}...")
    content = read_file(main_file)
    
    # Extract docstring
    docstring, content_without_docstring = extract_module_docstring(content)
    
    # Extract imports and code from main file
    main_imports, main_code = extract_imports_and_code(content_without_docstring)
    
    # Filter out the import from .modeling and relative imports
    filtered_main_imports = []
    for imp in main_imports:
        # Skip imports from .modeling module
        if 'from .modeling import' in imp:
            continue
        filtered_main_imports.append(imp)
    
    # Combine all imports (remove duplicates while preserving order)
    # Separate future imports from regular imports
    future_imports = []
    regular_imports = []
    seen_imports = set()
    
    for imp in filtered_main_imports + modeling_imports:
        normalized = normalize_import(imp)
        if normalized not in seen_imports:
            # Future imports must come first
            if 'from __future__ import' in imp:
                future_imports.append(imp)
            else:
                regular_imports.append(imp)
            seen_imports.add(normalized)
    
    # Combine with future imports first
    all_imports = future_imports + regular_imports
    
    # Build the final file
    parts = []
    
    # Add header comment
    header_comment = (
        "# This file was auto-generated by script/generate_modeling_file.py\n"
        "# It combines src/megatron2huggingface/modeling_megatron.py with all files\n"
        "# from src/megatron2huggingface/modeling/ into a single self-contained file.\n"
        "#\n"
        "# DO NOT EDIT THIS FILE DIRECTLY - regenerate it using:\n"
        "#   python script/generate_modeling_file.py [output_dir]\n"
    )
    parts.append(header_comment)
    
    # Add future imports first (they must come before docstring)
    if future_imports:
        parts.append('\n'.join(future_imports))
        parts.append('')
    
    # Add module docstring
    if docstring:
        parts.append(docstring)
        parts.append('')
    
    # Add regular imports
    if regular_imports:
        parts.append('\n'.join(regular_imports))
        parts.append('')
        parts.append('')
    
    # Add code from modeling module files (these come first as they define base classes)
    parts.append(modeling_code)
    parts.append('')
    parts.append('')
    
    # Add separator for main file code
    parts.append(f"# {'=' * 78}")
    parts.append("# MAIN MODEL CLASSES")
    parts.append(f"# {'=' * 78}\n")
    
    # Add main file code (remove leading blank lines)
    main_code = main_code.lstrip()
    parts.append(main_code)
    
    return '\n'.join(parts)


def generate_modeling_auto(
    output_dir: Optional[str] = None,
    src_dir: Optional[str] = None,
    verbose: bool = True
) -> Path:
    """Generate the combined modeling_megatron.py file.
    
    Args:
        output_dir: Directory where to write the output file. 
                   If None, uses the repository root.
        src_dir: Source directory containing megatron2huggingface package.
                If None, auto-detects from script location.
        verbose: Whether to print progress messages.
    
    Returns:
        Path to the generated file.
    
    Raises:
        FileNotFoundError: If required source files are not found.
    """
    # Determine paths
    if src_dir is None:
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        src_dir = repo_root / 'src' / 'megatron2huggingface'
    else:
        src_dir = Path(src_dir)
        repo_root = src_dir.parent.parent
    
    if output_dir is None:
        output_dir = repo_root
    else:
        output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    main_file = src_dir / 'modeling_megatron.py'
    modeling_dir = src_dir / 'modeling'
    output_file = output_dir / 'modeling_megatron.py'
    
    if verbose:
        print("=" * 80)
        print("Generating modeling_megatron.py")
        print("=" * 80)
        print(f"Source directory: {src_dir}")
        print(f"Main file: {main_file}")
        print(f"Modeling module: {modeling_dir}")
        print(f"Output file: {output_file}")
        print()
    
    # Check that required files exist
    if not main_file.exists():
        raise FileNotFoundError(f"Main file not found: {main_file}")
    
    if not modeling_dir.exists():
        raise FileNotFoundError(f"Modeling directory not found: {modeling_dir}")
    
    # Process modeling module
    if verbose:
        print("Processing modeling module files...")
    modeling_imports, modeling_code = process_modeling_module(modeling_dir, verbose=verbose)
    
    # Process main file and combine
    combined_content = process_main_file(main_file, modeling_imports, modeling_code, verbose=verbose)
    
    # Write output
    if verbose:
        print(f"\nWriting output to: {output_file}")
    write_file(output_file, combined_content)
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"✓ Successfully generated {output_file.name}")
        print(f"  Total lines: {len(combined_content.splitlines())}")
        print(f"{'=' * 80}")
    
    return output_file


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate a single modeling_megatron.py file from modular components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate to repository root (default)
  python script/generate_modeling_file.py
  
  # Generate to specific directory
  python script/generate_modeling_file.py /path/to/output/dir
  
  # Quiet mode (no progress messages)
  python script/generate_modeling_file.py -q /path/to/output/dir
        """
    )
    parser.add_argument(
        'output_dir',
        nargs='?',
        default=None,
        help='Output directory (default: repository root)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    parser.add_argument(
        '--src-dir',
        default=None,
        help='Source directory containing megatron2huggingface package (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    try:
        generate_modeling_auto(
            output_dir=args.output_dir,
            src_dir=args.src_dir,
            verbose=not args.quiet
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    exit(main())


# Example usage from another Python script (e.g., model_convert.py):
#
# from pathlib import Path
# from script.generate_modeling_file import generate_modeling_auto
#
# # Generate to a specific output directory
# output_file = generate_modeling_auto(
#     output_dir="/path/to/output/directory",
#     verbose=False  # Suppress progress messages
# )
# print(f"Generated: {output_file}")
#
# # Or with custom source directory
# output_file = generate_modeling_auto(
#     output_dir="/path/to/output",
#     src_dir="/custom/src/megatron2huggingface",
#     verbose=True
# )
