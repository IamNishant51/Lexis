"""Lexis: ultra-fast, local-first, zero-hallucination structured decisions."""

from .agents import (
    LexisAgentProxy,
    aider_config_yml,
    continue_config_yaml,
    detect_structural_intent,
    opencode_json,
    register_lexis_shield,
    shield_completion,
    strip_conversational_text,
    write_agent_configs,
)
from .engine import StructuredEngine, compile_schema_to_regex, get_engine
from .wrapper import GuardedClient, guard, zero_hallucination

__all__ = [
    "StructuredEngine",
    "compile_schema_to_regex",
    "get_engine",
    "zero_hallucination",
    "guard",
    "GuardedClient",
    "shield_completion",
    "strip_conversational_text",
    "detect_structural_intent",
    "LexisAgentProxy",
    "register_lexis_shield",
    "aider_config_yml",
    "continue_config_yaml",
    "opencode_json",
    "write_agent_configs",
]

__version__ = "0.2.0"
