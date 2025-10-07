"""AgentOS wrapper for the Sigma assistant."""

import asyncio
import os
from typing import Optional

from agno.agent import Agent
from agno.os import AgentOS

from agent_Sigma import (
    build_agent,
    default_session_id,
    ensure_knowledge_loaded,
)


def _resolve_session_id() -> Optional[str]:
    override = os.getenv("AGNO_OS_SESSION_ID")
    if override:
        return override
    return default_session_id()


def create_sigma_agent() -> Agent:
    """Build a Sigma agent instance with the AgentOS session preferences."""
    session_id = _resolve_session_id()
    return build_agent(session_id=session_id)


def _build_agent_os() -> AgentOS:
    sigma_agent = create_sigma_agent()
    return AgentOS(agents=[sigma_agent])


agent_os = _build_agent_os()
app = agent_os.get_app()


def serve_agentos_app() -> None:
    """Run AgentOS with the Sigma agent using configuration from environment variables."""
    host = os.getenv("AGNO_OS_HOST", "0.0.0.0")
    port = int(os.getenv("AGNO_OS_PORT", "7777"))
    reload_flag = os.getenv("AGNO_OS_RELOAD")
    reload_enabled = (
        reload_flag.strip().lower() in {"1", "true", "yes", "on"}
        if reload_flag
        else False
    )
    asyncio.run(ensure_knowledge_loaded())
    agent_os.serve(app="sigma_os:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    serve_agentos_app()
