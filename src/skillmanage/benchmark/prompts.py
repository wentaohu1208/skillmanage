"""Benchmark-specific prompt templates."""

# ---------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------

MATH_SYSTEM_PROMPT = """You are a math problem solver. Show your reasoning step by step.
{warnings_section}"""

MATH_COT_PROMPT = """Solve the following math problem step by step.
{skills_prompt}
Problem: {instruction}

Show your complete reasoning, then put your final answer in \\boxed{{}}.
Do NOT include any extra text after \\boxed{{}}."""

# ---------------------------------------------------------------------------
# BBH
# ---------------------------------------------------------------------------

BBH_SYSTEM_PROMPT = "You are a careful reasoning agent. Think step by step."

BBH_COT_PROMPT = """Answer the following question step by step.
{skills_prompt}
Question: {instruction}

Think through this carefully step by step, then give your final answer on the last line."""

# ---------------------------------------------------------------------------
# ALFWorld
# ---------------------------------------------------------------------------

ALFWORLD_SYSTEM_PROMPT = """You are an agent in a household environment. You interact with objects by issuing text commands.
Available commands: go to [place], take [object] from [place], put [object] in/on [place], open [container], close [container], use [object], heat [object] with [appliance], clean [object] with [appliance], cool [object] with [appliance], examine [object], look.
{skills_prompt}"""

ALFWORLD_STEP_PROMPT = """Previous actions:
{history}

Current observation: {observation}

Issue your next command (one action only):"""

# ---------------------------------------------------------------------------
# WebShop
# ---------------------------------------------------------------------------

WEBSHOP_SYSTEM_PROMPT = """You are a shopping agent. You navigate a web store to find and purchase products matching the given instruction.
Available actions: search[query], click[element]
{skills_prompt}"""

WEBSHOP_STEP_PROMPT = """Task: {instruction}

Previous actions:
{history}

Current page:
{observation}

Issue your next action (search[...] or click[...]):"""
