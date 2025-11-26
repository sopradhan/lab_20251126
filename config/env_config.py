import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class EnvConfig:
    """Load configuration from environment variables with dynamic path resolution"""
    
    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory"""
        return Path(__file__).parent.parent
    
    @staticmethod
    def get_db_path() -> str:
        """Get the incidents database path (configurable)"""
        default_path = str(EnvConfig.get_project_root() / "data" / "incidents.db")
        return os.getenv('INCIDENTS_DB_PATH', default_path)
    
    @staticmethod
    def get_legacy_db_path() -> str:
        """Get the legacy incident_iq database path"""
        default_path = str(EnvConfig.get_project_root() / "data" / "incident_iq.db")
        return os.getenv('LEGACY_DB_PATH', default_path)
    
    @staticmethod
    def get_chroma_db_path() -> str:
        """Get the ChromaDB path (configurable)"""
        default_path = str(EnvConfig.get_project_root() / "data" / "chroma_db")
        return os.getenv('CHROMA_DB_PATH', default_path)
    
    @staticmethod
    def get_data_dir() -> str:
        """Get the data directory path"""
        default_path = str(EnvConfig.get_project_root() / "data")
        return os.getenv('DATA_DIR', default_path)
    
    @staticmethod
    def get_rag_config_path() -> str:
        return os.getenv('RAG_CONFIG_PATH', 'config')
    
    @staticmethod
    def get_app_env() -> str:
        return os.getenv('APP_ENV', 'development')
    
    @staticmethod
    def get_log_level() -> str:
        return os.getenv('LOG_LEVEL', 'info')
