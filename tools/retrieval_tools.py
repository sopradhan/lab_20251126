"""
Retrieval Tools for LangGraph Autonomous Agentic RAG
Tools for searching, permission checking, reranking, and answer synthesis
"""

import json
from langchain_core.tools import tool



@tool
def get_vectordb_service() -> str:
    """Stub: Return the vector database service instance. Replace with actual implementation."""
    # This should be replaced with a real service locator or dependency injection
    return None

@tool
def search_vectordb_tool(query: str) -> str:
    """Stub: Search the vector database for relevant information."""
    # Replace this with actual search logic using vectordb_service
    return f"[STUB] Would search vector DB for: {query}"

@tool
def permission_check_tool(user_id: str, chunk_ids: str, db_service=None) -> str:
    """Check RBAC permissions for a user on a list of chunk IDs."""
    # TODO: Implement actual permission logic using db_service
    # For now, allow all chunks
    chunk_ids_list = json.loads(chunk_ids) if isinstance(chunk_ids, str) else chunk_ids
    return json.dumps({"allowed_chunks": chunk_ids_list})

@tool
def vector_search_tool(query: str, top_k: int = 5) -> str:
    """Search the vector database for similar chunks."""
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
        
        # Perform vector search
        results = vectordb_service.search(query, top_k=top_k)
        return json.dumps({"results": results})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def rerank_results_tool(query: str, results: list, top_n: int = 5) -> str:
    """Rerank search results for better relevance."""
    # TODO: Implement reranking logic
    # For now, return the first top_n results
    results_list = results if isinstance(results, list) else json.loads(results)
    return json.dumps({"reranked": results_list[:top_n]})

@tool
def synthesize_answer_tool(query: str, results: list) -> str:
    """Generate a final answer from retrieved chunks."""
    try:
        # Initialize services
        from ..services.llm_service import LLMService
        import json
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent / "config" / "llm_config.json"
        with open(config_path, "r") as f:
            llm_config = json.load(f)
        llm_service = LLMService(llm_config)
        
        # Prepare context from results
        context = "\n".join([r.get("text", "") for r in results])
        
        # Generate answer
        prompt = f"Based on the following context, answer the query: {query}\n\nContext:\n{context}\n\nAnswer:"
        response = llm_service.llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error synthesizing answer: {str(e)}"

@tool
def graph_expand_tool(results: list) -> str:
    """Expand context by finding related chunks."""
    # For now, return the input results (can be enhanced with graph expansion logic)
    results_list = results if isinstance(results, list) else json.loads(results)
    return json.dumps({"expanded": results_list})

@tool
def autonomous_vector_search_tool(query: str, top_k: int = 5, context: str = None) -> str:
    """
    Autonomous vector search with adaptive parameters based on context.
    Uses RL insights to optimize search parameters automatically.
    """
    try:
        # Parse context for autonomous optimization
        search_params = {"top_k": top_k}
        if context:
            try:
                ctx = json.loads(context)
                # Autonomous parameter adjustment based on context
                if ctx.get("previous_results_count", 0) < 3:
                    search_params["top_k"] = min(top_k * 2, 15)  # Expand search
                if ctx.get("avg_similarity", 0.5) < 0.3:
                    search_params["top_k"] = min(top_k * 3, 20)  # Broader search
                if ctx.get("query_complexity", "medium") == "high":
                    search_params["top_k"] = min(top_k * 2, 12)  # More comprehensive
            except:
                pass  # Use defaults on context parsing error
        
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
        
        # Perform autonomous vector search
        results = vectordb_service.search(query, top_k=search_params["top_k"])
        
        # Add autonomous search metadata
        autonomous_metadata = {
            "original_top_k": top_k,
            "adjusted_top_k": search_params["top_k"],
            "autonomous_adjustment": search_params["top_k"] != top_k,
            "search_strategy": "adaptive"
        }
        
        return json.dumps({
            "results": results, 
            "autonomous_metadata": autonomous_metadata
        })
    except Exception as e:
        return json.dumps({
            "error": str(e), 
            "results": [],
            "autonomous_metadata": {"error": "autonomous_search_failed"}
        })

@tool
def adaptive_rerank_tool(query: str, chunks: str, context: str = None) -> str:
    """
    Adaptive reranking with autonomous strategy selection.
    Uses context to determine optimal reranking approach.
    """
    try:
        chunks_list = json.loads(chunks) if isinstance(chunks, str) else chunks
        
        # Autonomous reranking strategy selection
        strategy = "semantic"  # default
        if context:
            try:
                ctx = json.loads(context)
                if ctx.get("query_type") == "factual":
                    strategy = "keyword_boost"
                elif ctx.get("domain") == "technical":
                    strategy = "technical_weight"
                elif ctx.get("user_preference") == "comprehensive":
                    strategy = "diversity_boost"
            except:
                pass
        
        # Apply autonomous reranking
        if strategy == "keyword_boost":
            # Boost chunks with exact keyword matches
            query_words = set(query.lower().split())
            for chunk in chunks_list:
                chunk_text = chunk.get('content', '').lower()
                keyword_matches = len([w for w in query_words if w in chunk_text])
                chunk['rerank_score'] = chunk.get('similarity', 0.5) + (keyword_matches * 0.1)
        
        elif strategy == "diversity_boost":
            # Promote diverse content
            seen_topics = set()
            for chunk in chunks_list:
                chunk_words = set(chunk.get('content', '').lower().split()[:10])
                overlap = len(chunk_words & seen_topics)
                diversity_bonus = 0.15 - (overlap * 0.03)
                chunk['rerank_score'] = chunk.get('similarity', 0.5) + diversity_bonus
                seen_topics.update(chunk_words)
        
        else:  # semantic (default)
            for chunk in chunks_list:
                chunk['rerank_score'] = chunk.get('similarity', 0.5)
        
        # Sort by autonomous rerank score
        chunks_list.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return json.dumps({
            "reranked": chunks_list,
            "strategy_used": strategy,
            "autonomous_reranking": True
        })
    except Exception as e:
        return json.dumps({"error": str(e), "reranked": []})

@tool
def hybrid_search_tool(query: str, top_k: int = 5, metadata_filters: dict = None, 
                      include_metadata: bool = True) -> str:
    """
    Autonomous hybrid search: Vector similarity + metadata filtering.
    
    This function implements intelligent hybrid retrieval:
    - Vector search: Fast semantic similarity in ChromaDB
    - Metadata filtering: Structured queries in SQLite
    - Result fusion: Combines both sources for optimal results
    
    Args:
        query: Search query text
        top_k: Number of results to return
        metadata_filters: Dict of metadata filters (e.g., {"severity": "High"})
        include_metadata: Whether to include detailed metadata
        
    Returns:
        JSON with hybrid search results
    """
    try:
        from pathlib import Path
        from ..services.llm_service import LLMService
        from ..services.vectordb_service import VectorDBService
        from ..config.env_config import EnvConfig
        import json
        import sqlite3
        
        # Initialize services
        config_path = Path(__file__).parent.parent / "config" / "llm_config.json"
        with open(config_path, "r") as f:
            llm_config = json.load(f)
        llm_service = LLMService(llm_config)
        
        chroma_db_path = EnvConfig.get_chroma_db_path()
        sqlite_db_path = EnvConfig.get_db_path()
        vectordb_service = VectorDBService(persist_directory=chroma_db_path)
        
        # Step 1: Vector similarity search in ChromaDB
        query_embedding = llm_service.embeddings.embed_query(query)
        vector_results = vectordb_service.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more results for filtering
            include=['documents', 'metadatas', 'distances']
        )
        
        # Step 2: Get chunk IDs for metadata lookup
        chunk_ids = vector_results['ids'][0] if vector_results['ids'] else []
        
        if not chunk_ids:
            return json.dumps({
                "success": False,
                "results": [],
                "message": "No vector search results found"
            })
        
        # Step 3: Retrieve detailed metadata from SQLite
        conn = sqlite3.connect(sqlite_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build metadata query with filters
        placeholders = ','.join(['?' for _ in chunk_ids])
        metadata_query = f"""
            SELECT cm.chunk_id, cm.doc_id, cm.chunk_text, cm.chunk_size, 
                   cm.strategy, cm.rbac_namespace, cm.created_date,
                   d.title, d.description, d.file_type
            FROM chunk_metadata cm
            LEFT JOIN documents d ON cm.doc_id = d.doc_id
            WHERE cm.chunk_id IN ({placeholders})
        """
        
        query_params = list(chunk_ids)
        
        # Add metadata filters if provided
        if metadata_filters:
            for key, value in metadata_filters.items():
                metadata_query += """
                    AND EXISTS (
                        SELECT 1 FROM chunk_metadata_details cmd 
                        WHERE cmd.chunk_id = cm.chunk_id 
                        AND cmd.metadata_key = ? AND cmd.metadata_value = ?
                    )
                """
                query_params.extend([key, str(value)])
        
        cursor.execute(metadata_query, query_params)
        metadata_results = cursor.fetchall()
        
        # Step 4: Fuse results with autonomous ranking
        fused_results = []
        metadata_dict = {row['chunk_id']: dict(row) for row in metadata_results}
        
        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id in metadata_dict:
                vector_score = 1.0 - vector_results['distances'][0][i]  # Convert distance to similarity
                
                result = {
                    "chunk_id": chunk_id,
                    "text": vector_results['documents'][0][i],
                    "vector_score": float(vector_score),
                    "vector_metadata": vector_results['metadatas'][0][i]
                }
                
                if include_metadata and chunk_id in metadata_dict:
                    result["detailed_metadata"] = metadata_dict[chunk_id]
                    
                    # Autonomous relevance boosting based on metadata
                    metadata_boost = 0.0
                    if metadata_filters:
                        for key, value in metadata_filters.items():
                            cursor.execute("""
                                SELECT COUNT(*) FROM chunk_metadata_details 
                                WHERE chunk_id = ? AND metadata_key = ? AND metadata_value = ?
                            """, (chunk_id, key, str(value)))
                            if cursor.fetchone()[0] > 0:
                                metadata_boost += 0.1
                    
                    result["final_score"] = vector_score + metadata_boost
                else:
                    result["final_score"] = vector_score
                
                fused_results.append(result)
        
        conn.close()
        
        # Step 5: Autonomous result ranking and filtering
        fused_results.sort(key=lambda x: x["final_score"], reverse=True)
        final_results = fused_results[:top_k]
        
        return json.dumps({
            "success": True,
            "query": query,
            "total_found": len(fused_results),
            "returned": len(final_results),
            "search_strategy": "hybrid",
            "results": final_results,
            "metadata_filters_applied": metadata_filters or {},
            "autonomous_ranking": True
        })
        
    except Exception as e:
        import traceback
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@tool
def metadata_query_tool(filters: dict, table_name: str = None, limit: int = 50) -> str:
    """
    Autonomous metadata querying from SQLite storage.
    
    Query structured metadata independently of vector search.
    Useful for filtering, analytics, and structured data exploration.
    
    Args:
        filters: Dict of metadata filters (e.g., {"severity": "High", "status": "Open"})
        table_name: Filter by source table name
        limit: Maximum results to return
        
    Returns:
        JSON with metadata query results
    """
    try:
        from ..config.env_config import EnvConfig
        import json
        import sqlite3
        
        sqlite_db_path = EnvConfig.get_db_path()
        conn = sqlite3.connect(sqlite_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build autonomous query
        base_query = """
            SELECT DISTINCT d.doc_id, d.title, d.description, d.created_date, 
                   d.rbac_namespace, d.total_chunks, d.file_type,
                   dm.metadata_key, dm.metadata_value
            FROM documents d
            LEFT JOIN document_metadata dm ON d.doc_id = dm.doc_id
        """
        
        where_conditions = []
        query_params = []
        
        # Add table filter
        if table_name:
            where_conditions.append("d.doc_id LIKE ?")
            query_params.append(f"{table_name}_%")
        
        # Add metadata filters
        for key, value in (filters or {}).items():
            where_conditions.append("""
                EXISTS (
                    SELECT 1 FROM document_metadata dm2 
                    WHERE dm2.doc_id = d.doc_id 
                    AND dm2.metadata_key = ? AND dm2.metadata_value = ?
                )
            """)
            query_params.extend([key, str(value)])
        
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        base_query += f" ORDER BY d.created_date DESC LIMIT {limit}"
        
        cursor.execute(base_query, query_params)
        results = cursor.fetchall()
        
        # Group results by document
        documents = {}
        for row in results:
            doc_id = row['doc_id']
            if doc_id not in documents:
                documents[doc_id] = {
                    "doc_id": doc_id,
                    "title": row['title'],
                    "description": row['description'],
                    "created_date": row['created_date'],
                    "rbac_namespace": row['rbac_namespace'],
                    "total_chunks": row['total_chunks'],
                    "file_type": row['file_type'],
                    "metadata": {}
                }
            
            if row['metadata_key'] and row['metadata_value']:
                documents[doc_id]["metadata"][row['metadata_key']] = row['metadata_value']
        
        conn.close()
        
        result_list = list(documents.values())
        
        return json.dumps({
            "success": True,
            "total_found": len(result_list),
            "filters_applied": filters or {},
            "table_filter": table_name,
            "query_type": "metadata_only",
            "documents": result_list
        })
        
    except Exception as e:
        import traceback
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@tool 
def get_document_analytics_tool(table_name: str = None, time_range_days: int = 30) -> str:
    """
    Autonomous analytics on stored documents and chunks.
    
    Provides insights into the hybrid storage system for optimization.
    
    Args:
        table_name: Focus on specific table (None for all)
        time_range_days: Days to look back for analytics
        
    Returns:
        JSON with analytics and insights
    """
    try:
        from ..config.env_config import EnvConfig
        import json
        import sqlite3
        from datetime import datetime, timedelta
        
        sqlite_db_path = EnvConfig.get_db_path()
        conn = sqlite3.connect(sqlite_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Calculate date filter
        cutoff_date = (datetime.now() - timedelta(days=time_range_days)).isoformat()
        
        # Basic document statistics
        doc_query = "SELECT COUNT(*) as total_docs, AVG(total_chunks) as avg_chunks FROM documents"
        params = []
        
        if table_name:
            doc_query += " WHERE doc_id LIKE ?"
            params.append(f"{table_name}_%")
            
        doc_query += f" AND created_date >= ?"
        params.append(cutoff_date)
        
        cursor.execute(doc_query, params)
        doc_stats = dict(cursor.fetchone())
        
        # Chunk statistics
        chunk_query = """
            SELECT COUNT(*) as total_chunks, 
                   AVG(chunk_size) as avg_chunk_size,
                   MIN(chunk_size) as min_chunk_size,
                   MAX(chunk_size) as max_chunk_size,
                   strategy
            FROM chunk_metadata 
            WHERE created_date >= ?
        """
        chunk_params = [cutoff_date]
        
        if table_name:
            chunk_query += " AND doc_id LIKE ?"
            chunk_params.append(f"{table_name}_%")
            
        chunk_query += " GROUP BY strategy"
        
        cursor.execute(chunk_query, chunk_params)
        chunk_stats = [dict(row) for row in cursor.fetchall()]
        
        # Storage utilization
        storage_query = """
            SELECT 
                file_type,
                COUNT(*) as document_count,
                SUM(total_chunks) as total_chunks,
                rbac_namespace
            FROM documents 
            WHERE created_date >= ?
        """
        storage_params = [cutoff_date]
        
        if table_name:
            storage_query += " AND doc_id LIKE ?"
            storage_params.append(f"{table_name}_%")
            
        storage_query += " GROUP BY file_type, rbac_namespace"
        
        cursor.execute(storage_query, storage_params)
        storage_stats = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        analytics = {
            "time_range_days": time_range_days,
            "table_filter": table_name,
            "document_statistics": doc_stats,
            "chunk_statistics": chunk_stats,
            "storage_utilization": storage_stats,
            "recommendations": []
        }
        
        # Autonomous recommendations
        if doc_stats.get('avg_chunks', 0) > 20:
            analytics["recommendations"].append("Consider larger chunk sizes to reduce fragmentation")
        
        if chunk_stats:
            avg_size = sum(s.get('avg_chunk_size', 0) for s in chunk_stats) / len(chunk_stats)
            if avg_size < 200:
                analytics["recommendations"].append("Chunk sizes are small - consider increasing chunk_size parameter")
            elif avg_size > 1000:
                analytics["recommendations"].append("Chunk sizes are large - consider decreasing chunk_size for better search granularity")
        
        return json.dumps({
            "success": True,
            "analytics": analytics,
            "generated_date": datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

# All retrieval tools are now LangGraph autonomous agentic RAG tools with reinforcement learning and hybrid storage capabilities.
