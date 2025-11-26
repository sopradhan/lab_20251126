"""
Autonomous Ingestion Agent for LangGraph RAG System
Handles intelligent document ingestion, chunking, and storage using
autonomous processing pipelines and adaptive strategies.
"""

from typing import Dict, Any, List
import json
import os
import re
import uuid
from pathlib import Path
import sys

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try relative imports first
try:
    from ..tools.ingestion_tools import (
        chunk_document_tool,
        extract_metadata_tool,
        save_to_vectordb_tool,
        ingest_sqlite_table_tool
    )
    from ..services.llm_service import LLMService
    from ..services.vectordb_service import VectorDBService
    from ..config.env_config import EnvConfig
except ImportError:
    from tools.ingestion_tools import (
        chunk_document_tool,
        extract_metadata_tool,
        save_to_vectordb_tool,
        ingest_sqlite_table_tool
    )
    from services.llm_service import LLMService
    from services.vectordb_service import VectorDBService
    from config.env_config import EnvConfig


class IngestionAgent:
    """Agent responsible for document ingestion and processing."""
    
    def __init__(self, llm_service: LLMService, vectordb_service: VectorDBService):
        self.llm_service = llm_service
        self.vectordb_service = vectordb_service
        self.tools = [
            chunk_document_tool,
            extract_metadata_tool,
            save_to_vectordb_tool,
            ingest_sqlite_table_tool
        ]
        
    def get_system_prompt(self) -> str:
        return """You are a document ingestion specialist. Your job is to use the available tools to process and store documents.

WORKFLOW:
1. Receive document path or content
2. Use chunk_document_tool to break content into chunks
3. Use extract_metadata_tool to extract metadata
4. Use save_to_vectordb_tool to store in vector database
5. Report success with chunk count and document ID

Be thorough and provide detailed results."""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process ingestion request through appropriate workflow."""
        print("[IngestionAgent] Processing ingestion request...")
        try:
            request = state.get("request", "")
            request_type = state.get("request_type", "ingestion")
            
            print(f"  → Request type: {request_type}")
            
            # Only process if this is an ingestion request
            if request_type != "ingestion":
                print(f"  → Skipping (request type: {request_type})")
                return state
            
            # Determine ingestion type: SQLite table vs document
            is_sqlite_request = self._detect_sqlite_request(request)
            
            if is_sqlite_request:
                print(f"  → Detected SQLite table ingestion request")
                result = self._handle_sqlite_ingestion(request)
            else:
                print(f"  → Detected document ingestion request")
                result = self._handle_document_ingestion(request)
            
            state["ingestion_result"] = result
            state["ingestion_completed"] = True
            state["metrics"]["ingestion"] += 1
            print(f"  ✅ Ingestion completed")
            return state
        except Exception as e:
            print(f"  ❌ Ingestion error: {e}")
            import traceback
            traceback.print_exc()
            state["ingestion_result"] = f"Error: {str(e)}"
            state["ingestion_completed"] = True
            state["metrics"]["errors"] += 1
            return state
    
    def _detect_sqlite_request(self, request: str) -> bool:
        """Detect if request is for SQLite table ingestion."""
        sqlite_keywords = [
            "table_name", "incidents", "database", "sqlite", "sql",
            "text_columns", "metadata_columns", "chunk_strategy"        ]
        request_lower = request.lower()
        return any(keyword in request_lower for keyword in sqlite_keywords)
    
    def _handle_sqlite_ingestion(self, request: str) -> str:
        """Handle SQLite table ingestion workflow with dynamic parameters."""
        print(f"    → Parsing SQLite ingestion parameters...")
        
        # Get parameters from request or use interactive prompts
        params = self._extract_sqlite_parameters(request)
        
        # If parameters are incomplete, prompt user or use defaults
        if not params.get('table_name'):
            params = self._get_interactive_sqlite_params(params)
        
        table_name = params['table_name']
        text_columns = params['text_columns']
        metadata_columns = params['metadata_columns']
        chunk_strategy = params['chunk_strategy']
        
        print(f"    → Using parameters:")
        print(f"      → Table: {table_name}")
        print(f"      → Text columns: {text_columns}")
        print(f"      → Metadata columns: {metadata_columns}")
        print(f"      → Chunk strategy: {chunk_strategy}")
        
        print(f"    → Calling ingest_sqlite_table_tool...")
        
        # Call SQLite ingestion tool with dynamic parameters
        result = ingest_sqlite_table_tool(
            table_name=table_name,
            text_columns=text_columns,
            metadata_columns=metadata_columns,
            chunk_strategy=chunk_strategy
        )
        
        return result
    
    def _extract_sqlite_parameters(self, request: str) -> Dict[str, Any]:
        """Extract SQLite ingestion parameters from request string."""
        params = {
            'table_name': None,
            'text_columns': [],
            'metadata_columns': [],
            'chunk_strategy': 'per_record'
        }
        
        try:
            # Extract table name
            table_match = re.search(r"table_name:\s*['\"]?(\w+)['\"]?", request, re.IGNORECASE)
            if table_match:
                params['table_name'] = table_match.group(1)
            
            # Extract text columns
            text_cols_match = re.search(r"text_columns:\s*\[(.*?)\]", request, re.IGNORECASE)
            if text_cols_match:
                cols_str = text_cols_match.group(1)
                params['text_columns'] = [col.strip().strip('"\'') for col in cols_str.split(',')]
            
            # Extract metadata columns
            meta_cols_match = re.search(r"metadata_columns:\s*\[(.*?)\]", request, re.IGNORECASE)
            if meta_cols_match:
                cols_str = meta_cols_match.group(1)
                params['metadata_columns'] = [col.strip().strip('"\'') for col in cols_str.split(',')]
            
            # Extract chunk strategy
            strategy_match = re.search(r"chunk_strategy:\s*['\"]?(\w+)['\"]?", request, re.IGNORECASE)
            if strategy_match:
                params['chunk_strategy'] = strategy_match.group(1)
                
        except Exception as e:
            print(f"      ⚠️  Error extracting parameters: {e}")
            
        return params
    
    def _get_interactive_sqlite_params(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get SQLite parameters interactively or use smart defaults."""
        
        # Default configurations for known tables
        table_configs = {
            'incidents': {
                'text_columns': ['title', 'description'],
                'metadata_columns': ['severity', 'status', 'assigned_to', 'category'],
                'chunk_strategy': 'per_record'
            },
            'tickets': {
                'text_columns': ['subject', 'description', 'resolution'],
                'metadata_columns': ['priority', 'status', 'assignee', 'department'],
                'chunk_strategy': 'per_record'
            },
            'documents': {
                'text_columns': ['title', 'content', 'summary'],
                'metadata_columns': ['category', 'author', 'version', 'tags'],
                'chunk_strategy': 'content_based'
            },
            'knowledge_base': {
                'text_columns': ['question', 'answer', 'content'],
                'metadata_columns': ['category', 'difficulty', 'tags'],
                'chunk_strategy': 'per_record'
            }
        }
        
        # If we have a partial table name, try to match it
        if not current_params.get('table_name'):
            # Default to 'incidents' if no table specified
            current_params['table_name'] = 'incidents'
            print(f"      → No table specified, defaulting to: incidents")
        
        table_name = current_params['table_name']
        
        # Use smart defaults based on table name
        if table_name in table_configs:
            config = table_configs[table_name]
            if not current_params.get('text_columns'):
                current_params['text_columns'] = config['text_columns']
                print(f"      → Using default text columns for {table_name}: {config['text_columns']}")
            
            if not current_params.get('metadata_columns'):
                current_params['metadata_columns'] = config['metadata_columns']
                print(f"      → Using default metadata columns for {table_name}: {config['metadata_columns']}")
                
            if not current_params.get('chunk_strategy'):
                current_params['chunk_strategy'] = config['chunk_strategy']
                print(f"      → Using default chunk strategy for {table_name}: {config['chunk_strategy']}")
        else:
            # Generic defaults for unknown tables
            if not current_params.get('text_columns'):
                current_params['text_columns'] = ['title', 'description', 'content']
                print(f"      → Using generic text columns: {current_params['text_columns']}")
            
            if not current_params.get('metadata_columns'):
                current_params['metadata_columns'] = ['category', 'status', 'created_date']
                print(f"      → Using generic metadata columns: {current_params['metadata_columns']}")
        
        return current_params
    
    def _handle_document_ingestion(self, request: str) -> str:
        """Handle document ingestion workflow."""
        print(f"    → Chunking document...")
        chunk_result = chunk_document_tool(request)
        
        print(f"    → Extracting metadata...")
        metadata_result = extract_metadata_tool(request)
        
        # Generate unique doc_id
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"
        
        print(f"    → Saving to vector database (doc_id: {doc_id})...")
        save_result = save_to_vectordb_tool(chunk_result, doc_id, metadata_result)
        
        return save_result
