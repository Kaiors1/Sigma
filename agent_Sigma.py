import asyncio
import os
import uuid
import warnings

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.db.sqlite import SqliteDb
from agno.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.vectordb.lancedb import LanceDb
from agno.memory import MemoryManager
from agno.db.schemas.memory import UserMemory
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.calculator import CalculatorTools
from agno.tools.memory import MemoryTools
from agno.tools.shell import ShellTools
from agno.tools.file import FileTools
from agno.tools.telegram import TelegramTools
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

# Structured output schema
class Response(BaseModel):
    answer: str = Field(description="Main answer to the question")
    confidence: float = Field(description="Confidence level 0-1")
    sources: List[str] = Field(description="Information sources used")

DB_FILE = os.getenv("AGNO_DB_FILE", "agent_chat.db")
LANCEDB_URI = os.getenv("AGNO_LANCEDB_URI", "tmp/lancedb")
LANCEDB_TABLE = os.getenv("AGNO_LANCEDB_TABLE", "knowledge")
EMBEDDER_ID = os.getenv("AGNO_EMBEDDER_ID", "qwen3-embedding:8b")
MODEL_ID = os.getenv("AGNO_MODEL_ID", "deepseek-reasoner")
SESSION_ID = os.getenv("AGNO_SESSION_ID", "sigma_chat_main")
HISTORY_RUNS = getenv_int("AGNO_HISTORY_RUNS", 5)
LOAD_KNOWLEDGE = getenv_bool("AGNO_LOAD_KNOWLEDGE", True)
KNOWLEDGE_URL = os.getenv("AGNO_KNOWLEDGE_URL", "https://docs.agno.com/llms-full.txt")
ENABLE_USER_MEMORIES = getenv_bool("AGNO_ENABLE_USER_MEMORIES", True)
ENABLE_AGENTIC_MEMORY = getenv_bool("AGNO_ENABLE_AGENTIC_MEMORY", True)
ADD_HISTORY_TO_CONTEXT = getenv_bool("AGNO_ADD_HISTORY_TO_CONTEXT", True)
SEARCH_KNOWLEDGE = getenv_bool("AGNO_SEARCH_KNOWLEDGE", True)
ENABLE_REASONING = getenv_bool("AGNO_ENABLE_REASONING", False)
REASONING_MIN_STEPS = getenv_int("AGNO_REASONING_MIN_STEPS", 1)
REASONING_MAX_STEPS = getenv_int("AGNO_REASONING_MAX_STEPS", 10)
REASONING_MODEL_ID = os.getenv("AGNO_REASONING_MODEL_ID")
STREAM_INTERMEDIATE_STEPS = getenv_bool("AGNO_STREAM_INTERMEDIATE_STEPS", False)
ENABLE_KNOWLEDGE_UPDATE = getenv_bool("AGNO_UPDATE_KNOWLEDGE", False)
ENABLE_RESPONSE_CAPTURE = getenv_bool("AGNO_CAPTURE_RESPONSES", False)
RESPONSE_CAPTURE_TOPIC = os.getenv("AGNO_CAPTURE_TOPIC", "sigma_responses")
ENABLE_MARKDOWN = getenv_bool("AGNO_ENABLE_MARKDOWN", True)
ENABLE_DEBUG_MODE = getenv_bool("AGNO_ENABLE_DEBUG_MODE", False)
ENABLE_DUCKDUCKGO = getenv_bool("AGNO_ENABLE_DUCKDUCKGO", True)
ENABLE_CALCULATOR = getenv_bool("AGNO_ENABLE_CALCULATOR", True)
ENABLE_MEMORY_TOOL = getenv_bool("AGNO_ENABLE_MEMORY_TOOL", True)
DESCRIPTION = os.getenv(
    "AGNO_DESCRIPTION",
    "Sigma Agent — terminal-native researcher with web, math, file, shell, memory, and optional Telegram push tooling",
)
PROMPT_SESSION_ID = getenv_bool("AGNO_PROMPT_SESSION_ID", False)
ENABLE_SHELL_TOOL = getenv_bool("AGNO_ENABLE_SHELL_TOOL", True)
SHELL_BASE_DIR = os.getenv("AGNO_SHELL_BASE_DIR")
SHELL_ENABLE_RUN = getenv_bool("AGNO_SHELL_ENABLE_RUN", True)
SHELL_ENABLE_ALL = getenv_bool("AGNO_SHELL_ENABLE_ALL", False)
ENABLE_FILE_TOOL = getenv_bool("AGNO_ENABLE_FILE_TOOL", True)
FILE_BASE_DIR = os.getenv("AGNO_FILE_BASE_DIR")
FILE_ENABLE_SAVE = getenv_bool("AGNO_FILE_ENABLE_SAVE", True)
FILE_ENABLE_READ = getenv_bool("AGNO_FILE_ENABLE_READ", True)
FILE_ENABLE_LIST = getenv_bool("AGNO_FILE_ENABLE_LIST", True)
FILE_ENABLE_SEARCH = getenv_bool("AGNO_FILE_ENABLE_SEARCH", True)
FILE_ENABLE_ALL = getenv_bool("AGNO_FILE_ENABLE_ALL", False)
ENABLE_SESSION_SUMMARIES = getenv_bool("AGNO_ENABLE_SESSION_SUMMARIES", True)
AGENT_ID = os.getenv("AGNO_AGENT_ID", "sigma-personal")
AGENT_NAME = os.getenv("AGNO_AGENT_NAME", "Sigma Personal Agent")
AGENT_ROLE = os.getenv("AGNO_AGENT_ROLE")
ENABLE_TELEGRAM_TOOL = getenv_bool("AGNO_ENABLE_TELEGRAM_TOOL", True)
TELEGRAM_CHAT_ID = os.getenv("AGNO_TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("AGNO_TELEGRAM_TOKEN")
DEFAULT_USER_ID = os.getenv("AGNO_USER_ID")
SYSTEM_MEMORY_USER_ID = os.getenv("AGNO_SYSTEM_USER_ID", "agent_system")
SEED_SYSTEM_MEMORIES = getenv_bool("AGNO_SEED_SYSTEM_MEMORIES", True)

instructions_env = os.getenv("AGNO_INSTRUCTIONS")
if instructions_env:
    INSTRUCTIONS = [item.strip() for item in instructions_env.split("||") if item.strip()]
else:
    INSTRUCTIONS = [
        "You are Sigma, a terminal-first Agno agent. Be transparent about your toolkit: DuckDuckGo for research, Calculator for math, File/Shell tools for local changes (always confirm before mutating state), and Telegram for outbound alerts when enabled.",
        "Lead with a concise outcome summary, then provide supporting detail and cite sources for factual claims.",
        "Check LanceDB knowledge and MemoryTools before answering questions that may rely on stored context or documentation.",
        "Maintain two memory lanes: personal user context stays under the provided user_id; agent/system knowledge belongs under AGNO_SYSTEM_USER_ID.",
        "Call out when you store or retrieve memories so the user knows what Sigma already remembers.",
        "Ask clarifying questions whenever requirements or intent are uncertain, and offer safe alternatives when a request is risky or unsupported.",
        "State clearly when a request cannot be fulfilled or would violate guardrails.",
    ]

# Database setup
db = SqliteDb(db_file=DB_FILE)

# Memory manager setup
memory_manager = MemoryManager(db=db)

# Knowledge base setup
knowledge = Knowledge(
    contents_db=db,
    vector_db=LanceDb(
        uri=LANCEDB_URI,
        table_name=LANCEDB_TABLE,
        embedder=OllamaEmbedder(id=EMBEDDER_ID),
    ),
)

_knowledge_initialized = False
_system_memories_seeded = False
SYSTEM_MEMORY_CONTEXT: Optional[str] = None


def add_system_memory(memory: str, topics: Optional[List[str]] = None) -> Optional[str]:
    """Persist agent-centric knowledge under a dedicated system user."""
    if not memory:
        warnings.warn("Empty memory text provided; skipping system memory creation.")
        return None
    if memory_manager.db is None:
        warnings.warn("Memory manager has no database configured; cannot store system memory.")
        return None
    if not SYSTEM_MEMORY_USER_ID:
        warnings.warn("AGNO_SYSTEM_USER_ID is not set; skipping system memory creation.")
        return None
    user_memory = UserMemory(memory=memory, topics=topics)
    return memory_manager.add_user_memory(memory=user_memory, user_id=SYSTEM_MEMORY_USER_ID)


def load_system_memory_context() -> Optional[str]:
    """Return concatenated system memory snapshots for prompt injection."""
    if memory_manager.db is None or not SYSTEM_MEMORY_USER_ID:
        return None
    memories = memory_manager.get_user_memories(user_id=SYSTEM_MEMORY_USER_ID) or []
    snippets = [mem.memory for mem in memories if getattr(mem, "memory", None)]
    if not snippets:
        return None
    return "\n".join(snippets)


def seed_system_memories() -> None:
    """Ensure baseline system memories about Sigma are persisted."""
    global _system_memories_seeded
    if _system_memories_seeded or not SEED_SYSTEM_MEMORIES:
        return
    if memory_manager.db is None:
        warnings.warn("Memory manager unavailable; cannot seed system memories.")
        return
    if not SYSTEM_MEMORY_USER_ID:
        warnings.warn("AGNO_SYSTEM_USER_ID is empty; cannot seed system memories.")
        return
    existing = memory_manager.get_user_memories(user_id=SYSTEM_MEMORY_USER_ID) or []
    existing_memories = {mem.memory for mem in existing if getattr(mem, "memory", None)}
    default_user = DEFAULT_USER_ID or "default"
    policy_snapshot = (
        f"Memory policy: personal facts use user_id '{default_user}'; agent/system knowledge is stored under '{SYSTEM_MEMORY_USER_ID}'."
    )
    stack_snapshot = (
        f"Runtime stack: DeepSeek model '{MODEL_ID}', LanceDB knowledge source '{KNOWLEDGE_URL}', SQLite history file '{DB_FILE}'."
    )
    system_snapshots = [
        "Sigma Agent capabilities: DuckDuckGo lookup, calculator, file access, shell execution with confirmation prompts, optional Telegram push notifications.",
        policy_snapshot,
        stack_snapshot,
    ]
    topics = ["agent_profile", "capabilities"]
    for snapshot in system_snapshots:
        if snapshot and snapshot not in existing_memories:
            add_system_memory(snapshot, topics=topics)
    _system_memories_seeded = True


async def ensure_knowledge_loaded() -> None:
    """Load configured knowledge corpus once per process."""
    global _knowledge_initialized
    if LOAD_KNOWLEDGE and KNOWLEDGE_URL and not _knowledge_initialized:
        await knowledge.add_content_async(url=KNOWLEDGE_URL, skip_if_exists=True)
        _knowledge_initialized = True
    seed_system_memories()
    global SYSTEM_MEMORY_CONTEXT
    SYSTEM_MEMORY_CONTEXT = load_system_memory_context()


def _build_reasoning_model():
    if not REASONING_MODEL_ID:
        return None
    try:
        return DeepSeek(id=REASONING_MODEL_ID)
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(
            f"Failed to initialize reasoning model '{REASONING_MODEL_ID}'. Falling back to primary model. Error: {exc}"
        )
        return None


def _capture_response_hook(run_output, agent: Agent, **_kwargs) -> None:
    if not ENABLE_RESPONSE_CAPTURE:
        return
    memory_backend = getattr(agent, "memory_manager", None)
    if memory_backend is None:
        warnings.warn("AGNO_CAPTURE_RESPONSES enabled but no memory manager configured; skipping capture.")
        return
    content = getattr(run_output, "content", None)
    if not isinstance(content, str):
        return
    trimmed = content.strip()
    if not trimmed:
        return
    topics = [RESPONSE_CAPTURE_TOPIC] if RESPONSE_CAPTURE_TOPIC else None
    user_id = agent.user_id or DEFAULT_USER_ID or "default"
    memory_entry = UserMemory(memory=trimmed, topics=topics)
    memory_backend.add_user_memory(memory=memory_entry, user_id=user_id)


def build_tools() -> List[object]:
    tool_instances = []
    if ENABLE_DUCKDUCKGO:
        tool_instances.append(DuckDuckGoTools())
    if ENABLE_CALCULATOR:
        tool_instances.append(CalculatorTools())
    if ENABLE_MEMORY_TOOL:
        tool_instances.append(MemoryTools(db=db))
    if ENABLE_FILE_TOOL:
        file_kwargs = {
            "enable_save_file": FILE_ENABLE_SAVE,
            "enable_read_file": FILE_ENABLE_READ,
            "enable_list_files": FILE_ENABLE_LIST,
            "enable_search_files": FILE_ENABLE_SEARCH,
            "all": FILE_ENABLE_ALL,
        }
        if FILE_BASE_DIR:
            file_kwargs["base_dir"] = FILE_BASE_DIR
        tool_instances.append(FileTools(**file_kwargs))
    if ENABLE_SHELL_TOOL:
        shell_kwargs = {
            "enable_run_shell_command": SHELL_ENABLE_RUN,
            "all": SHELL_ENABLE_ALL,
        }
        if SHELL_BASE_DIR:
            shell_kwargs["base_dir"] = SHELL_BASE_DIR
        tool_instances.append(ShellTools(**shell_kwargs))
    if ENABLE_TELEGRAM_TOOL:
        if TELEGRAM_CHAT_ID:
            telegram_kwargs = {"chat_id": TELEGRAM_CHAT_ID}
            if TELEGRAM_TOKEN:
                telegram_kwargs["token"] = TELEGRAM_TOKEN
            tool_instances.append(TelegramTools(**telegram_kwargs))
        else:
            warnings.warn("AGNO_TELEGRAM_CHAT_ID is not set; skipping TelegramTools.")
    return tool_instances


def default_session_id() -> str:
    return SESSION_ID or f"sigma_chat_{uuid.uuid4().hex}"


def build_agent(session_id: Optional[str] = None) -> Agent:
    resolved_session_id = session_id or default_session_id()
    effective_instructions = list(INSTRUCTIONS)
    if SYSTEM_MEMORY_CONTEXT:
        system_prompt = "System knowledge reminders:\n" + SYSTEM_MEMORY_CONTEXT
        effective_instructions.insert(0, system_prompt)

    agent_kwargs = {
        "model": DeepSeek(id=MODEL_ID),
        "db": db,
        "session_id": resolved_session_id,
        "enable_user_memories": ENABLE_USER_MEMORIES,
        "enable_agentic_memory": ENABLE_AGENTIC_MEMORY,
        "enable_session_summaries": ENABLE_SESSION_SUMMARIES,
        "add_history_to_context": ADD_HISTORY_TO_CONTEXT,
        "num_history_runs": HISTORY_RUNS,
        "tools": build_tools(),
        "knowledge": knowledge,
        "memory_manager": memory_manager,
        "search_knowledge": SEARCH_KNOWLEDGE,
        "reasoning": ENABLE_REASONING,
        "instructions": effective_instructions,
        "description": DESCRIPTION,
        "markdown": ENABLE_MARKDOWN,
        "debug_mode": ENABLE_DEBUG_MODE,
        "reasoning_min_steps": max(1, REASONING_MIN_STEPS),
        "reasoning_max_steps": max(REASONING_MIN_STEPS, REASONING_MAX_STEPS),
        "stream_intermediate_steps": STREAM_INTERMEDIATE_STEPS,
        "update_knowledge": ENABLE_KNOWLEDGE_UPDATE,
    }
    if DEFAULT_USER_ID:
        agent_kwargs["user_id"] = DEFAULT_USER_ID
    if AGENT_ID:
        agent_kwargs["id"] = AGENT_ID
    if AGENT_NAME:
        agent_kwargs["name"] = AGENT_NAME
    if AGENT_ROLE:
        agent_kwargs["role"] = AGENT_ROLE
    reasoning_model = _build_reasoning_model()
    if reasoning_model is not None:
        agent_kwargs["reasoning_model"] = reasoning_model
    if ENABLE_RESPONSE_CAPTURE:
        agent_kwargs["post_hooks"] = [_capture_response_hook]
    return Agent(**agent_kwargs)


async def initialize_agent(session_id: Optional[str] = None) -> Agent:
    await ensure_knowledge_loaded()
    global SYSTEM_MEMORY_CONTEXT
    if SYSTEM_MEMORY_CONTEXT is None:
        SYSTEM_MEMORY_CONTEXT = load_system_memory_context()
    return build_agent(session_id=session_id)


def resolve_session_id_from_user() -> Optional[str]:
    if PROMPT_SESSION_ID:
        try:
            candidate = input("Enter session ID (leave blank to auto-generate): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if candidate:
            return candidate
    if SESSION_ID:
        return SESSION_ID
    return None


def run_repl(agent_instance: Agent) -> None:
    print("🤖 AI Assistant ready! (type 'exit' to quit)")
    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! 👋")
            break

        print("\nAssistant:")
        try:
            agent_instance.print_response(user_input, stream=True)
        except KeyboardInterrupt:
            print("Generation interrupted by user.")
        except Exception as exc:
            if ENABLE_DEBUG_MODE:
                raise
            print(f"Encountered an error while generating a response: {exc}")


def main() -> None:
    session_id_choice = resolve_session_id_from_user()
    resolved_session = session_id_choice or default_session_id()
    agent_instance = asyncio.run(initialize_agent(session_id=resolved_session))
    run_repl(agent_instance)


if __name__ == "__main__":
    main()
