"""
Utility functions for document formatting and output.

Provides consistent formatting for document content and metadata presentation.
"""

def print_docs(docs, include_page=False):
    """Print document content with source information."""
    for doc in docs:
        source_info = f"{doc.metadata['source']}"
        if include_page:
            source_info += f" (Page {doc.metadata['page']})"
        print(f"{source_info}:\n\n{doc.page_content}\n\n")

def format_docs(docs):
    """Format documents into a single string with metadata."""

    return "\n\n".join(f"Source: {doc.metadata['source']} (Page {doc.metadata['page']}):\n\n{doc.page_content}\n\n" for doc in docs)

def extract_source_ids(docs):
    """
    Extract source IDs from document metadata.
    
    Args:
        docs: List of documents
    
    Returns:
        list: List of source numbers (XX) from '[XX]_XXXX.XXXXX' format
    """
    source_ids = []
    for doc in docs:
        source = doc.metadata['source']
        # Extract [XX]_XXXX.XXXXX pattern from the full path
        if '\\' in source:  # Handle Windows paths
            filename = source.split('\\')[-1]
        else:  # Handle Unix paths
            filename = source.split('/')[-1]
            
        # Extract number between brackets
        if filename.startswith('[') and ']' in filename:
            number = filename[1:filename.index(']')]
            try:
                source_ids.append(int(number))  # Convert to integer
            except ValueError:
                continue  # Skip if not a valid number
            
    return source_ids
