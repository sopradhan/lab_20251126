"""
Ingestion Tools - Document processing and database storage
Handles missing tool decorator gracefully with fallback pattern
"""

import json
import sqlite3
import datetime
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ..config.env_config import EnvConfig
except ImportError:
    from config.env_config import EnvConfig

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Smart import with fallback for tool decorator
def _get_tool_decorator():
    """
    Attempt to import tool decorator from various sources.
    Returns a no-op decorator if all imports fail.
    """
    # Try langchain_core tools first (preferred for LangGraph autonomous agentic RAG)
    try:
        from langchain_core.tools import tool
        print("✅ Using tool decorator from langchain_core.tools for LangGraph agent")
        return tool
    except ImportError:
        pass
    
    # Try langchain_core (for compatibility)
    try:
        from langchain_core.tools import tool
        print("✅ Using tool decorator from langchain_core.tools")
        return tool
    except ImportError:
        pass
    
    # Try langchain (fallback)
    try:
        from langchain.tools import tool
        print("✅ Using tool decorator from langchain.tools")
        return tool
    except ImportError:
        pass
    
    # Fallback to no-op
    print("⚠️  No tool decorator found, using no-op")
    def tool(fn):
        return fn
    return tool

tool = _get_tool_decorator()


# ============================================================================
# INGESTION TOOLS - Document chunking, metadata extraction, and storage
# ============================================================================

def create_metadata_table_if_not_exists(db_path: str):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            key TEXT,
            value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


@tool
def chunk_document_tool(text: str, strategy: str = "recursive", chunk_size: int = 500, overlap: int = 50) -> str:
    """
    Chunk document using specified strategy.
    
    Args:
        text: Document text to chunk
        strategy: Chunking strategy ("recursive" is default)
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        JSON with chunks and metadata
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        result = [
            {
                "chunk_id": f"chunk_{i}",
                "text": chunk,
                "strategy": strategy,
                "size": len(chunk),
                "index": i
            }
            for i, chunk in enumerate(chunks)
        ]
        
        return json.dumps({
            "success": True,
            "num_chunks": len(result),
            "chunks": result
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def extract_metadata_tool(text: str) -> str:
    """
    Extract metadata from document using LLM.
    
    Args:
        text: Document text to analyze
    
    Returns:
        JSON with extracted metadata
    """
    try:
        # Initialize LLM service
        from ..services.llm_service import LLMService
        import json
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent / "config" / "llm_config.json"
        with open(config_path, "r") as f:
            llm_config = json.load(f)
        llm_service = LLMService(llm_config)
        
        # Use LLM to extract metadata
        prompt = f"""Extract metadata from the following document text. Return JSON with:
        - title: Document title
        - summary: Brief summary (max 200 chars)
        - keywords: List of key terms
        - topics: List of topics
        - doc_type: Type of document
        
        Document text:
        {text[:2000]}
        
        Return only valid JSON."""
        
        response = llm_service.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        try:
            metadata = json.loads(content)
        except:
            # Fallback metadata
            metadata = {
                "title": "Document",
                "summary": text[:200],
                "keywords": ["document"],
                "topics": [],
                "doc_type": "incident"
            }
        
        return json.dumps({
            "success": True,
            "metadata": metadata
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def save_to_vectordb_tool(chunks: str, doc_id: str, metadata: str = None, rbac_namespace: str = "general",
                         healing_suggestions: str = "") -> str:
    """
    Hybrid storage: Save embeddings to vector DB and metadata to SQLite.
    
    This function implements the autonomous hybrid storage pattern:
    - Embeddings: Stored in ChromaDB for fast similarity search
    - Metadata: Stored in SQLite for efficient relational queries
    - Text content: Stored in both for redundancy and flexibility
    
    Args:
        chunks: JSON string of chunks to save
        doc_id: Document identifier
        metadata: JSON string of document metadata
        rbac_namespace: RBAC namespace for access control
        healing_suggestions: Optional healing/optimization suggestions
    
    Returns:
        JSON with save status and hybrid storage details
    """
    try:
        # Initialize services
        from ..services.llm_service import LLMService
        from ..services.vectordb_service import VectorDBService
        from ..config.env_config import EnvConfig
        import json
        from pathlib import Path
        import sqlite3
        import datetime
        
        config_path = Path(__file__).parent.parent / "config" / "llm_config.json"
        with open(config_path, "r") as f:
            llm_config = json.load(f)
        llm_service = LLMService(llm_config)
        
        # Initialize hybrid storage
        chroma_db_path = EnvConfig.get_chroma_db_path()
        sqlite_db_path = EnvConfig.get_db_path()
        vectordb_service = VectorDBService(persist_directory=chroma_db_path)
        
        # Ensure hybrid storage tables exist
        create_hybrid_storage_tables(sqlite_db_path)
        
        chunks_data = json.loads(chunks) if isinstance(chunks, str) else chunks
        
        if not chunks_data.get('success'):
            return json.dumps({"success": False, "error": "Invalid chunks data"})
        
        chunk_list = chunks_data.get('chunks', [])
        if not chunk_list:
            return json.dumps({"success": False, "error": "No chunks to save"})
        
        # Parse metadata
        doc_metadata = json.loads(metadata) if metadata else {}
        
        # Store document metadata in SQLite
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()
        
        # Store main document record
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (doc_id, title, description, created_date, updated_date, rbac_namespace, total_chunks)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                doc_metadata.get('title', f'Document {doc_id}'),
                doc_metadata.get('description', 'Auto-generated document'),
                datetime.datetime.now().isoformat(),
                datetime.datetime.now().isoformat(),
                rbac_namespace,
                len(chunk_list)
            ))
        except Exception as e:
            print(f"[WARNING] Failed to store document record: {e}")
        
        # Store document metadata in SQLite
        try:
            # Clear existing metadata for this document
            cursor.execute("DELETE FROM document_metadata WHERE doc_id = ?", (doc_id,))
            
            # Store new metadata
            for key, value in doc_metadata.items():
                cursor.execute("""
                    INSERT INTO document_metadata (doc_id, metadata_key, metadata_value, value_type)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, key, str(value), type(value).__name__))
        except Exception as e:
            print(f"[WARNING] Failed to store document metadata: {e}")
        
        saved_chunks = []
        
        # Process each chunk with hybrid storage
        for chunk in chunk_list:
            chunk_text = chunk['text']
            chunk_id = f"{doc_id}_{chunk['chunk_id']}"
            
            # Generate embedding for vector storage
            embedding = llm_service.embeddings.embed_query(chunk_text)
            
            # Prepare minimal metadata for ChromaDB (only essential search metadata)
            vector_metadata = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "rbac_namespace": rbac_namespace,
                "chunk_index": chunk['index'],
                "strategy": chunk.get('strategy', 'recursive'),
                "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            }
            
            # Store embedding + minimal metadata in ChromaDB
            vectordb_service.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[vector_metadata],
                documents=[chunk_text]
            )
            
            # Store detailed chunk metadata in SQLite
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunk_metadata 
                    (chunk_id, doc_id, chunk_index, chunk_text, chunk_size, 
                     strategy, rbac_namespace, created_date, embedding_dimension, healing_suggestions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    doc_id,
                    chunk['index'],
                    chunk_text,
                    chunk['size'],
                    chunk.get('strategy', 'recursive'),
                    rbac_namespace,
                    datetime.datetime.now().isoformat(),
                    len(embedding),
                    healing_suggestions
                ))
                
                # Store chunk-specific metadata
                for key, value in chunk.items():
                    if key not in ['text', 'chunk_id', 'index', 'size', 'strategy']:
                        cursor.execute("""
                            INSERT INTO chunk_metadata_details (chunk_id, metadata_key, metadata_value)
                            VALUES (?, ?, ?)
                        """, (chunk_id, key, str(value)))
                        
            except Exception as e:
                print(f"[WARNING] Failed to store chunk metadata for {chunk_id}: {e}")
            
            saved_chunks.append({
                "chunk_id": chunk_id,
                "text_length": len(chunk_text),
                "embedding_dim": len(embedding),
                "storage_type": "hybrid"
            })
        
        conn.commit()
        conn.close()
        
        return json.dumps({
            "success": True,
            "doc_id": doc_id,
            "chunks_saved": len(saved_chunks),
            "storage_strategy": "hybrid",
            "vector_db": "ChromaDB",
            "metadata_db": "SQLite",
            "chunks": saved_chunks,
            "rbac_namespace": rbac_namespace,
            "healing_suggestions": healing_suggestions
        })
        
    except Exception as e:
        import traceback
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def create_hybrid_storage_tables(db_path: str):
    """Create tables for hybrid storage system."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Documents table for main document records
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            created_date TEXT,
            updated_date TEXT,
            rbac_namespace TEXT DEFAULT 'general',
            total_chunks INTEGER DEFAULT 0,
            file_path TEXT,
            file_type TEXT,
            file_size INTEGER
        )
    ''')
    
    # Document metadata table (key-value pairs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            metadata_key TEXT,
            metadata_value TEXT,
            value_type TEXT,
            created_date TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE,
            UNIQUE(doc_id, metadata_key)
        )
    ''')
    
    # Chunk metadata table (detailed chunk information)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_metadata (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            chunk_index INTEGER,
            chunk_text TEXT,
            chunk_size INTEGER,
            strategy TEXT,
            rbac_namespace TEXT DEFAULT 'general',
            created_date TEXT,
            embedding_dimension INTEGER,
            healing_suggestions TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
        )
    ''')
    
    # Chunk metadata details (key-value pairs for chunks)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_metadata_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT,
            metadata_key TEXT,
            metadata_value TEXT,
            FOREIGN KEY (chunk_id) REFERENCES chunk_metadata (chunk_id) ON DELETE CASCADE
        )
    ''')
    
    # Access control table for RBAC
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_control (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resource_id TEXT,
            resource_type TEXT CHECK(resource_type IN ('document', 'chunk')),
            user_id TEXT,
            permission TEXT CHECK(permission IN ('read', 'write', 'admin')),
            granted_date TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(resource_id, user_id, permission)
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_metadata_doc_id ON document_metadata (doc_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_metadata_doc_id ON chunk_metadata (doc_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_metadata_rbac ON chunk_metadata (rbac_namespace)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_control_resource ON access_control (resource_id, resource_type)')
    
    conn.commit()
    conn.close()


@tool
            return json.dumps({"success": False, "error": "No chunks to save"})
        
        # Parse metadata
        doc_metadata = json.loads(metadata) if metadata else {}
        
        # Store metadata in SQLite
        if metadata:
            try:
                doc_metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                for k, v in doc_metadata.items():
                    cursor.execute(
                        "INSERT INTO document_metadata (doc_id, key, value) VALUES (?, ?, ?)",
                        (doc_id, k, str(v))
                    )
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[WARNING] Failed to store metadata in SQLite: {e}")
        
        saved_chunks = []
        for chunk in chunk_list:
            chunk_text = chunk['text']
            chunk_id = f"{doc_id}_{chunk['chunk_id']}"
            
            # Generate embedding
            embedding = llm_service.embeddings.embed_query(chunk_text)
            
            # Prepare metadata for storage
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "rbac_namespace": rbac_namespace,
                "chunk_index": chunk['index'],
                "chunk_size": chunk['size'],
                "strategy": chunk.get('strategy', 'recursive'),
                "healing_suggestions": healing_suggestions,
                **doc_metadata
            }
            
            # Save to vector database
            vectordb_service.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[chunk_metadata],
                documents=[chunk_text]
            )
            
            saved_chunks.append({
                "chunk_id": chunk_id,
                "text_length": len(chunk_text),
                "embedding_dim": len(embedding)
            })
        
        return json.dumps({
            "success": True,
            "doc_id": doc_id,
            "chunks_saved": len(saved_chunks),
            "total_chunks": len(chunk_list),
            "saved_chunks": saved_chunks
        })
        
        except Exception as e:
    return json.dumps({"success": False, "error": str(e)})


@tool
def ingest_sqlite_table_tool(table_name: str, text_columns: list, metadata_columns: list = None,
                            db_path: str = None,
                            chunk_size: int = 512, chunk_overlap: int = 50,
                            chunk_strategy: str = "per_record", where_clause: str = None) -> str:
    """
    Autonomous hybrid ingestion: SQLite table to vector DB + metadata storage.
    
    This function implements intelligent hybrid storage:
    - Source data: Read from SQLite table
    - Embeddings: Generated and stored in ChromaDB for similarity search
    - Metadata: Stored in structured SQLite tables for efficient querying
    - Text content: Chunked and stored in both systems
    
    Args:
        table_name: Name of SQLite table to ingest
        text_columns: List of column names to combine as searchable text
        metadata_columns: List of column names to attach as metadata
        db_path: Path to SQLite database (uses config if None)
        chunk_size: Semantic chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        chunk_strategy: "per_record" (one document per row) or "sequential"
        where_clause: Optional SQL WHERE clause for filtering records
    
    Returns:
        JSON with hybrid ingestion status and statistics
    """
    try:
        # Initialize services
        from ..services.llm_service import LLMService
        from ..services.vectordb_service import VectorDBService
        from ..config.env_config import EnvConfig
        import json
        from pathlib import Path
        import sqlite3
        import datetime
        
        config_path = Path(__file__).parent.parent / "config" / "llm_config.json"
        with open(config_path, "r") as f:
            llm_config = json.load(f)
        llm_service = LLMService(llm_config)
        
        # Initialize hybrid storage
        chroma_db_path = EnvConfig.get_chroma_db_path()
        if db_path is None:
            source_db_path = EnvConfig.get_db_path()
        else:
            source_db_path = db_path
            
        metadata_db_path = EnvConfig.get_db_path()  # Always use config path for metadata storage
        vectordb_service = VectorDBService(persist_directory=chroma_db_path)
        
        # Ensure hybrid storage tables exist
        create_hybrid_storage_tables(metadata_db_path)
        
        # Connect to source database
        conn = sqlite3.connect(source_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Validate table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            conn.close()
            return json.dumps({
                "success": False,
                "error": f"Table '{table_name}' does not exist in database"
            })
        
        # Build query with autonomous optimization
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        query += " ORDER BY id"  # Ensure consistent ordering
        
        print(f"[HybridIngestion] Executing query: {query}")
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return json.dumps({
                "success": False,
                "error": f"No records found in table '{table_name}'"
            })
        
        print(f"[HybridIngestion] Found {len(rows)} records in '{table_name}'")
        
        # Connect to metadata database
        meta_conn = sqlite3.connect(metadata_db_path)
        meta_cursor = meta_conn.cursor()
        
        # Process records based on chunk strategy
        documents = []
        
        if chunk_strategy == "per_record":
            # One document per database record - optimal for structured data
            for row in rows:
                row_dict = dict(row)
                record_id = row_dict.get('id', f'record_{len(documents)}')
                
                # Combine text columns into document content
                text_parts = []
                for col in text_columns:
                    if col in row_dict and row_dict[col]:
                        value = row_dict[col]
                        text_parts.append(f"{col.upper()}: {value}")
                
                document_text = "\n".join(text_parts)
                
                if document_text.strip():
                    # Extract metadata for hybrid storage
                    record_metadata = {"source_table": table_name, "record_id": record_id}
                    if metadata_columns:
                        for col in metadata_columns:
                            if col in row_dict:
                                record_metadata[col] = row_dict[col]
                    
                    # Store record metadata in SQLite immediately
                    doc_id = f"{table_name}_{record_id}"
                    try:
                        meta_cursor.execute("""
                            INSERT OR REPLACE INTO documents 
                            (doc_id, title, description, created_date, updated_date, rbac_namespace, file_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            doc_id,
                            f"{table_name} Record {record_id}",
                            f"Record from {table_name} table",
                            datetime.datetime.now().isoformat(),
                            datetime.datetime.now().isoformat(),
                            f"table_{table_name}",
                            "sqlite_record"
                        ))
                        
                        # Store record metadata
                        for key, value in record_metadata.items():
                            meta_cursor.execute("""
                                INSERT OR REPLACE INTO document_metadata 
                                (doc_id, metadata_key, metadata_value, value_type)
                                VALUES (?, ?, ?, ?)
                            """, (doc_id, key, str(value), type(value).__name__))
                            
                    except Exception as e:
                        print(f"[WARNING] Failed to store metadata for record {record_id}: {e}")
                    
                    documents.append({
                        "text": document_text,
                        "metadata": record_metadata,
                        "table": table_name,
                        "doc_id": doc_id
                    })
        
        elif chunk_strategy == "sequential":
            # Combine records sequentially - useful for narrative data
            combined_text_parts = []
            combined_metadata = {
                "record_count": 0, 
                "source_table": table_name,
                "strategy": "sequential"
            }
            
            for row in rows:
                row_dict = dict(row)
                text_parts = []
                
                for col in text_columns:
                    if col in row_dict and row_dict[col]:
                        value = row_dict[col]
                        text_parts.append(f"{col.upper()}: {value}")
                
                document_section = "\n".join(text_parts)
                if document_section.strip():
                    record_id = row_dict.get('id', combined_metadata["record_count"])
                    combined_text_parts.append(f"--- Record {record_id} ---\n{document_section}")
                    combined_metadata["record_count"] += 1
            
            if combined_text_parts:
                doc_id = f"{table_name}_combined"
                documents.append({
                    "text": "\n\n".join(combined_text_parts),
                    "metadata": combined_metadata,
                    "table": table_name,
                    "doc_id": doc_id
                })
                
                # Store combined document metadata
                try:
                    meta_cursor.execute("""
                        INSERT OR REPLACE INTO documents 
                        (doc_id, title, description, created_date, updated_date, rbac_namespace, file_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        doc_id,
                        f"{table_name} Combined Records",
                        f"Combined records from {table_name} table",
                        datetime.datetime.now().isoformat(),
                        datetime.datetime.now().isoformat(),
                        f"table_{table_name}",
                        "sqlite_combined"
                    ))
                except Exception as e:
                    print(f"[WARNING] Failed to store combined document metadata: {e}")
        
        else:
            meta_conn.close()
            return json.dumps({
                "success": False,
                "error": f"Unknown chunk_strategy: {chunk_strategy}. Use 'per_record' or 'sequential'"
            })
        
        if not documents:
            meta_conn.close()
            return json.dumps({
                "success": False,
                "error": "No documents created after processing records"
            })
        
        print(f"[HybridIngestion] Created {len(documents)} documents for hybrid storage")
        
        # Autonomous chunking with hybrid storage
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        ingestion_results = {
            "table_name": table_name,
            "records_processed": len(rows),
            "documents_created": len(documents),
            "storage_strategy": "hybrid",
            "vector_db": "ChromaDB", 
            "metadata_db": "SQLite",
            "chunks": []
        }
        
        # Process each document with hybrid storage
        total_chunks = 0
        for doc_idx, doc in enumerate(documents):
            # Chunk the document
            chunks = splitter.split_text(doc["text"])
            doc_chunks = 0
            
            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                
                # Generate chunk ID
                chunk_id = f"{doc['doc_id']}_chunk_{chunk_idx}_{int(datetime.datetime.now().timestamp() * 1000)}"
                
                # Generate embedding for vector storage
                embedding = llm_service.generate_embedding(chunk_text)
                
                # Prepare minimal metadata for ChromaDB (fast search)
                vector_metadata = {
                    "table": table_name,
                    "doc_id": doc["doc_id"],
                    "chunk_index": chunk_idx,
                    "chunk_strategy": chunk_strategy,
                    "rbac_namespace": f"table_{table_name}",
                    "source_records": len(rows) if chunk_strategy == "sequential" else 1,
                    "text_preview": chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text
                }
                
                # Store embedding in ChromaDB
                vectordb_service.collection.add(
                    ids=[chunk_id],
                    documents=[chunk_text],
                    embeddings=[embedding],
                    metadatas=[vector_metadata]
                )
                
                # Store detailed chunk metadata in SQLite
                try:
                    meta_cursor.execute("""
                        INSERT OR REPLACE INTO chunk_metadata 
                        (chunk_id, doc_id, chunk_index, chunk_text, chunk_size, 
                         strategy, rbac_namespace, created_date, embedding_dimension)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        doc["doc_id"],
                        chunk_idx,
                        chunk_text,
                        len(chunk_text),
                        chunk_strategy,
                        f"table_{table_name}",
                        datetime.datetime.now().isoformat(),
                        len(embedding)
                    ))
                    
                    # Store chunk-specific metadata details
                    for key, value in doc["metadata"].items():
                        meta_cursor.execute("""
                            INSERT INTO chunk_metadata_details (chunk_id, metadata_key, metadata_value)
                            VALUES (?, ?, ?)
                        """, (chunk_id, key, str(value)))
                        
                except Exception as e:
                    print(f"[WARNING] Failed to store chunk metadata for {chunk_id}: {e}")
                
                ingestion_results["chunks"].append({
                    "chunk_id": chunk_id,
                    "chunk_size": len(chunk_text),
                    "embedding_dimension": len(embedding),
                    "storage": "hybrid"
                })
                
                doc_chunks += 1
                total_chunks += 1
            
            # Update document chunk count
            try:
                meta_cursor.execute("""
                    UPDATE documents SET total_chunks = ? WHERE doc_id = ?
                """, (doc_chunks, doc["doc_id"]))
            except Exception as e:
                print(f"[WARNING] Failed to update chunk count for {doc['doc_id']}: {e}")
        
        meta_conn.commit()
        meta_conn.close()
        
        ingestion_results["total_chunks_created"] = total_chunks
        
        print(f"[HybridIngestion] Successfully processed {total_chunks} chunks")
        print(f"  → Vector embeddings stored in ChromaDB")
        print(f"  → Metadata stored in SQLite")
        
        return json.dumps({
            "success": True,
            **ingestion_results
        })
        
    except Exception as e:
        import traceback
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    try:
        # Initialize services
        from ..services.llm_service import LLMService
        from ..services.vectordb_service import VectorDBService
        from ..config.env_config import EnvConfig
        import json
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent / "config" / "llm_config.json"
        with open(config_path, "r") as f:
            llm_config = json.load(f)
        llm_service = LLMService(llm_config)
        vectordb_service = VectorDBService(persist_directory=EnvConfig.get_chroma_db_path())
        
        # Use config database path if not provided
        if db_path is None:
            db_path = EnvConfig.get_db_path()
        
        # Ensure metadata table exists
        create_metadata_table_if_not_exists(db_path)
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Validate table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            conn.close()
            return json.dumps({
                "success": False,
                "error": f"Table '{table_name}' does not exist in database"
            })
        
        # Build query
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        # Fetch all records
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return json.dumps({
                "success": False,
                "error": f"No records found in table '{table_name}'"
            })
        
        # Store metadata in SQLite for each record if metadata_columns are provided
        if metadata_columns:
            try:
                create_metadata_table_if_not_exists(db_path)
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                for row in rows:
                    doc_id = str(row["id"]) if "id" in row.keys() else None
                    if doc_id:
                        for k in metadata_columns:
                            v = row[k]
                            cursor.execute(
                                "INSERT INTO document_metadata (doc_id, key, value) VALUES (?, ?, ?)",
                                (doc_id, k, str(v))
                            )
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[WARNING] Failed to store metadata in SQLite: {e}")
        
        # Process records based on chunk strategy
        documents = []
        
        if chunk_strategy == "per_record":
            # One document per database record
            for row in rows:
                row_dict = dict(row)
                
                # Combine text columns into document content
                text_parts = []
                for col in text_columns:
                    if col in row_dict and row_dict[col]:
                        value = row_dict[col]
                        text_parts.append(f"{col}: {value}")
                
                document_text = "\n".join(text_parts)
                
                if document_text.strip():
                    # Extract metadata
                    metadata = {}
                    if metadata_columns:
                        for col in metadata_columns:
                            if col in row_dict:
                                metadata[col] = row_dict[col]
                    
                    documents.append({
                        "text": document_text,
                        "metadata": metadata,
                        "table": table_name
                    })
        
        elif chunk_strategy == "sequential":
            # Combine records sequentially
            combined_text_parts = []
            combined_metadata = {"record_count": 0}
            
            for row in rows:
                row_dict = dict(row)
                text_parts = []
                
                for col in text_columns:
                    if col in row_dict and row_dict[col]:
                        value = row_dict[col]
                        text_parts.append(f"{col}: {value}")
                
                document_section = "\n".join(text_parts)
                if document_section.strip():
                    combined_text_parts.append(f"--- Record ---\n{document_section}")
                    combined_metadata["record_count"] += 1
            
            if combined_text_parts:
                documents.append({
                    "text": "\n\n".join(combined_text_parts),
                    "metadata": combined_metadata,
                    "table": table_name
                })
        
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown chunk_strategy: {chunk_strategy}. Use 'per_record' or 'sequential'"
            })
        
        if not documents:
            return json.dumps({
                "success": False,
                "error": "No documents created after processing records"
            })
        
        # Chunk each document semantically
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        ingestion_results = {
            "table_name": table_name,
            "records_processed": len(rows),
            "documents_created": len(documents),
            "chunks": []
        }
        
        # Process each document
        chunk_counter = 0
        for doc_idx, doc in enumerate(documents):
            # Chunk the document
            chunks = splitter.split_text(doc["text"])
            
            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                
                # Generate chunk ID
                chunk_id = f"{table_name}_doc_{doc_idx}_chunk_{chunk_idx}_{int(datetime.datetime.now().timestamp() * 1000)}"
                
                # Generate embedding
                if llm_service:
                    embedding = llm_service.generate_embedding(chunk_text)
                else:
                    embedding = [0.0] * 384  # Default embedding dimension
                
                # Prepare chunk metadata
                chunk_metadata = {
                    "table": table_name,
                    "document_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "chunk_strategy": chunk_strategy,
                    "source_records": len(rows) if chunk_strategy == "sequential" else 1,
                    **doc["metadata"]
                }
                
                # Save to vector DB
                vectordb_service.collection.add(
                    ids=[chunk_id],
                    documents=[chunk_text],
                    embeddings=[embedding],
                    metadatas=[chunk_metadata]
                )
                
                ingestion_results["chunks"].append({
                    "chunk_id": chunk_id,
                    "chunk_size": len(chunk_text),
                    "metadata": chunk_metadata
                })
                
                chunk_counter += 1
        
        ingestion_results["total_chunks_created"] = chunk_counter
        
        return json.dumps({
            "success": True,
            **ingestion_results
        })
        
    except Exception as e:
        import traceback
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
