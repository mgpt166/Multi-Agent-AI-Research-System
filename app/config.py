"""
app/config.py
=============
Central configuration for the Multi-Agent AI Research System.

This is the ONLY place where parameters and environment variables are read.
No other module should call os.getenv() directly — import from here instead.

To change any setting:
    1. Set the environment variable in .env
    2. The new value is picked up automatically on next server start

Groups:
    LLM             Model IDs and API keys for LLM providers
    Search          Search provider selection and settings
    Agent Limits    Caps on sub-agents, iterations, tool rounds, token limits
    HITL            Human-in-the-loop timeout and refine round limits
    Concurrency     Thread pool sizes and timeouts
    Output          Report output directory
    UI              Gradio interface settings
    Server          Host / port for uvicorn
"""

import os

# ── LLM ───────────────────────────────────────────────────────────────────────
# Two separate model configs so lead/synthesis quality stays high while
# sub-agents (which run many times per job) use a cheaper/faster model.
#
# GROQ_MODEL         — lead researcher, synthesizer, citation agent, sufficiency eval
#   llama-3.3-70b-versatile   best quality  (100k TPD free tier)
#
# GROQ_SUB_AGENT_MODEL — sub-agents only (run N times per job)
#   llama-3.1-8b-instant      500k TPD free tier — much higher daily allowance
#   llama-3.3-70b-versatile   same as lead (set both the same for uniform quality)

GROQ_API_KEY: str            = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str              = os.getenv("GROQ_MODEL",             "llama-3.3-70b-versatile")
GROQ_SUB_AGENT_MODEL: str    = os.getenv("GROQ_SUB_AGENT_MODEL",   "llama-3.1-8b-instant")
ANTHROPIC_API_KEY: str       = os.getenv("ANTHROPIC_API_KEY", "")

# Per-call max_tokens caps — controls how long each LLM response can be
LLM_MAX_TOKENS_PLAN: int        = int(os.getenv("LLM_MAX_TOKENS_PLAN",        "4000"))
LLM_MAX_TOKENS_EVALUATE: int    = int(os.getenv("LLM_MAX_TOKENS_EVALUATE",    "50"))
LLM_MAX_TOKENS_SYNTHESIZE: int  = int(os.getenv("LLM_MAX_TOKENS_SYNTHESIZE",  "8000"))
LLM_MAX_TOKENS_REVIEW: int      = int(os.getenv("LLM_MAX_TOKENS_REVIEW",      "1000"))
LLM_MAX_TOKENS_SUBAGENT: int    = int(os.getenv("LLM_MAX_TOKENS_SUBAGENT",    "8000"))
LLM_MAX_TOKENS_CITATION: int    = int(os.getenv("LLM_MAX_TOKENS_CITATION",    "8000"))

# ── Search ────────────────────────────────────────────────────────────────────
# Swap backend by setting SEARCH_PROVIDER=bing (or brave, serpapi) in .env

SEARCH_PROVIDER: str        = os.getenv("SEARCH_PROVIDER", "tavily")
TAVILY_API_KEY: str         = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_DEPTH: str    = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
SEARCH_MAX_RESULTS: int     = int(os.getenv("SEARCH_MAX_RESULTS", "10"))

# ── Agent Limits ──────────────────────────────────────────────────────────────
# Control how much work each agent does per run

MAX_SUBAGENTS: int          = int(os.getenv("MAX_SUBAGENTS", "2"))
MAX_ITERATIONS: int         = int(os.getenv("MAX_ITERATIONS", "3"))
MAX_TOOL_ROUNDS: int        = int(os.getenv("MAX_TOOL_ROUNDS", "10"))   # per sub-agent

# Content truncation limits sent to LLM (chars, not tokens)
FINDINGS_TRUNCATE_CHARS: int    = int(os.getenv("FINDINGS_TRUNCATE_CHARS",  "8000"))
NARRATIVE_TRUNCATE_CHARS: int   = int(os.getenv("NARRATIVE_TRUNCATE_CHARS", "6000"))
REVIEW_TRUNCATE_CHARS: int      = int(os.getenv("REVIEW_TRUNCATE_CHARS",    "3000"))
EVALUATE_TRUNCATE_CHARS: int    = int(os.getenv("EVALUATE_TRUNCATE_CHARS",  "2000"))

# ── HITL ──────────────────────────────────────────────────────────────────────
# Human-in-the-loop checkpoint settings

HITL_TIMEOUT_SECONDS: int       = int(os.getenv("HITL_TIMEOUT_SECONDS",       "180"))
MAX_HITL_REFINE_ROUNDS: int     = int(os.getenv("MAX_HITL_REFINE_ROUNDS",     "3"))
MAX_SYNTHESIS_REVIEW_ROUNDS: int= int(os.getenv("MAX_SYNTHESIS_REVIEW_ROUNDS","1"))

# ── Concurrency ───────────────────────────────────────────────────────────────
# Thread pool and timeout settings for background job execution

MAX_CONCURRENT_JOBS: int        = int(os.getenv("MAX_CONCURRENT_JOBS",        "3"))
SUBAGENT_TIMEOUT_SECONDS: int   = int(os.getenv("SUBAGENT_TIMEOUT_SECONDS",   "600"))
API_REQUEST_TIMEOUT: int        = int(os.getenv("API_REQUEST_TIMEOUT",        "15"))

# ── Output ────────────────────────────────────────────────────────────────────

OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")

# ── UI ────────────────────────────────────────────────────────────────────────

API_BASE_URL: str             = os.getenv("API_BASE_URL", "http://localhost:8000")
UI_PATH: str                  = os.getenv("UI_PATH", "/ui")
UI_URL: str                   = f"{API_BASE_URL}{UI_PATH}"   # e.g. http://localhost:8000/ui
UI_POLL_INTERVAL_SECONDS: int = int(os.getenv("UI_POLL_INTERVAL_SECONDS", "5"))

# ── Server ────────────────────────────────────────────────────────────────────

SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

# ── Evaluation ────────────────────────────────────────────────────────────────
# Controls the eval framework (scripts/run_eval.py).
# Distribution: ~27% simple, ~27% medium, ~27% complex, ~19% adversarial.
# Examples: 15 → 4/4/4/3,  8 → 2/2/2/2,  4 → 1/1/1/1

EVAL_TEST_COUNT: int  = int(os.getenv("EVAL_TEST_COUNT",  "15"))
# Judge model — can be a Groq model (e.g. llama-3.3-70b-versatile) or a Claude model
# (e.g. claude-sonnet-4-6). The judge auto-detects the provider from the model name.
EVAL_JUDGE_MODEL: str = os.getenv("EVAL_JUDGE_MODEL", "llama-3.3-70b-versatile")

# ── Cost Controls ──────────────────────────────────────────────────────────────
# Hard budget ceiling per research query. Pipeline stops if exceeded.
# TOKEN_PRICE_* defaults match Groq free-tier blended rate (USD per million tokens).

MAX_COST_PER_QUERY: float       = float(os.getenv("MAX_COST_PER_QUERY",     "0.50"))
COST_WARNING_THRESHOLD: float   = float(os.getenv("COST_WARNING_THRESHOLD", "0.35"))
TOKEN_PRICE_INPUT: float        = float(os.getenv("TOKEN_PRICE_INPUT",      "0.065"))  # per million
TOKEN_PRICE_OUTPUT: float       = float(os.getenv("TOKEN_PRICE_OUTPUT",     "0.065"))  # per million

# ── Logging ────────────────────────────────────────────────────────────────────
# LOG_FORMAT: "pretty" for colourised dev output, "json" for production aggregators
# LOG_LEVEL:  standard Python level name (DEBUG | INFO | WARNING | ERROR)

LOG_FORMAT: str = os.getenv("LOG_FORMAT", "pretty")
LOG_LEVEL: str  = os.getenv("LOG_LEVEL",  "INFO")
