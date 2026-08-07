import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

database_cfg = config.get("database", {})
llm_cfg = config.get("llm", {})
embedding_cfg = config.get("embedding", {})
retriever_cfg = config.get("retriever", {})
data_cfg = config.get("data", {})
logging_cfg = config.get("logging", {})
agent_cfg = config.get("agent", {})
sandbox_cfg = config.get("sandbox", {})

# --- Database Settings ---
DB_NAME = database_cfg.get("db_name", "chat_meta.db")
HISTORY_DB_PATH = database_cfg.get("history_db_path", "sqlite:///memory_pure.sqlite")

# --- LLM and Embedding Model Settings ---
LLM_TYPE = llm_cfg.get("type", "ChatOpenAI")
LLM_MODEL_NAME = llm_cfg.get("model_name", "glm-4-flash")
EMBEDDING_TYPE = embedding_cfg.get("type", "HuggingFaceEmbeddings")
EMBEDDING_MODEL_NAME = embedding_cfg.get("model_name", "all-MiniLM-L6-v2")
ZAI_API_BASE = llm_cfg.get("zai_api_base", "https://open.bigmodel.cn/api/paas/v4/")

# --- Retriever Settings ---
RETRIEVER_TYPE = retriever_cfg.get("type", "HybridRRFRetriever")
VECTOR_SEARCH_K = retriever_cfg.get("vector_search_k", 6)
VECTOR_FETCH_K = retriever_cfg.get("vector_fetch_k", 24)
VECTOR_MMR_LAMBDA = retriever_cfg.get("vector_mmr_lambda", 0.4)
BM25_K = retriever_cfg.get("bm25_k", 8)
RRF_K = retriever_cfg.get("rrf_k", 60)
RERANK_MODEL_NAME = retriever_cfg.get("rerank_model_name", "BAAI/bge-reranker-v2-m3")
RERANK_CANDIDATES = retriever_cfg.get("rerank_candidates", 8)
RERANK_TOP_K = retriever_cfg.get("rerank_top_k", 4)
RERANK_MIN_SCORE = retriever_cfg.get("rerank_min_score", -100.0)
RERANK_SKIP_ENABLED = retriever_cfg.get("rerank_skip_enabled", True)
RERANK_SKIP_SHORT_QUERY_TOKENS = retriever_cfg.get("rerank_skip_short_query_tokens", 3)
RERANK_SKIP_RRF_ABS_MARGIN = retriever_cfg.get("rerank_skip_rrf_abs_margin", 0.003)
RERANK_SKIP_RRF_RATIO = retriever_cfg.get("rerank_skip_rrf_ratio", 1.25)
CHUNK_SIZE = retriever_cfg.get("chunk_size", 600)
CHUNK_OVERLAP = retriever_cfg.get("chunk_overlap", 120)
ENABLE_HYDE = retriever_cfg.get("enable_hyde", True)
HYDE_MAX_QUERY_LENGTH = retriever_cfg.get("hyde_max_query_length", 200)
HYDE_MAX_OUTPUT_CHARS = retriever_cfg.get("hyde_max_output_chars", 180)

# --- Loader Settings ---
LOADER_TYPE = config.get("loader", {}).get("type", "UnstructuredFileLoader")
PDF_LOADER_TYPE = config.get("loader", {}).get("pdf_type", "PyPDFLoader")
TXT_LOADER_TYPE = config.get("loader", {}).get("txt_type", "TextLoader")

# --- Data Persistence ---
DATA_DIR = data_cfg.get("dir", "./data")
MAX_UPLOAD_MB = data_cfg.get("max_upload_mb", 20)
MAX_CHUNKS_PER_FILE = data_cfg.get("max_chunks_per_file", 1000)
ASYNC_UPLOAD_THRESHOLD_MB = data_cfg.get("async_upload_threshold_mb", 5)
MAX_SESSION_STORAGE_MB = data_cfg.get("max_session_storage_mb", 200)
MAX_WORKSPACE_IMPORT_MB = data_cfg.get("max_workspace_import_mb", 100)
MAX_WORKSPACE_OUTPUT_FILES = data_cfg.get("max_workspace_output_files", 100)
MAX_WORKSPACE_OUTPUT_MB = data_cfg.get("max_workspace_output_mb", 50)
MAX_ARTIFACT_MB = data_cfg.get("max_artifact_mb", 20)
UPLOAD_SCAN_MODE = str(data_cfg.get("upload_scan_mode", "off")).lower()
CLAMAV_COMMAND = str(data_cfg.get("clamav_command", "clamscan"))

# --- Logging ---
LOG_FILE = logging_cfg.get("log_file", "chatbot.log")

# --- Agent Runtime ---
AGENT_ENABLE_CODE_EXECUTION = agent_cfg.get("enable_code_execution", True)
AGENT_MAX_ITERATIONS = agent_cfg.get("max_iterations", 8)
AGENT_MAX_EXECUTION_TIME_SECONDS = agent_cfg.get("max_execution_time_seconds", 60)
AGENT_CODE_TIMEOUT_SECONDS = agent_cfg.get("code_timeout_seconds", 5)
AGENT_CODE_OUTPUT_LIMIT = agent_cfg.get("code_output_limit", 8000)
AGENT_CODE_MAX_CHARS = agent_cfg.get("code_max_chars", 12000)

# --- Isolated workspace runtime ---
SANDBOX_ENABLED = bool(sandbox_cfg.get("enabled", False))
SANDBOX_PROVIDER = str(sandbox_cfg.get("provider", "opensandbox"))
SANDBOX_DOMAIN = str(sandbox_cfg.get("domain", "127.0.0.1:8080"))
SANDBOX_PROTOCOL = str(sandbox_cfg.get("protocol", "http"))
SANDBOX_API_KEY_ENV = str(sandbox_cfg.get("api_key_env", "OPEN_SANDBOX_API_KEY"))
SANDBOX_IMAGE = str(sandbox_cfg.get("image", "chatbot-workspace:py311-pandas-duckdb"))
SANDBOX_TIMEOUT_SECONDS = int(sandbox_cfg.get("sandbox_timeout_seconds", 1200))
SANDBOX_IDLE_TIMEOUT_SECONDS = int(sandbox_cfg.get("sandbox_idle_timeout_seconds", 600))
SANDBOX_REAPER_INTERVAL_SECONDS = int(sandbox_cfg.get("sandbox_reaper_interval_seconds", 60))
SANDBOX_REQUEST_TIMEOUT_SECONDS = int(sandbox_cfg.get("request_timeout_seconds", 30))
SANDBOX_COMMAND_TIMEOUT_SECONDS = int(sandbox_cfg.get("command_timeout_seconds", 120))
SANDBOX_MAX_COMMAND_CHARS = int(sandbox_cfg.get("max_command_chars", 16000))
SANDBOX_MAX_OUTPUT_CHARS = int(sandbox_cfg.get("max_output_chars", 50000))
SANDBOX_CPU = str(sandbox_cfg.get("cpu", "1"))
SANDBOX_MEMORY = str(sandbox_cfg.get("memory", "2Gi"))
SANDBOX_NETWORK_ENABLED = bool(sandbox_cfg.get("network_enabled", False))
SANDBOX_ALLOWED_EGRESS = list(sandbox_cfg.get("allowed_egress", []))
SANDBOX_USE_SERVER_PROXY = bool(sandbox_cfg.get("use_server_proxy", True))
