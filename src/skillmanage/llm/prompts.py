"""All LLM prompt templates."""

# ---------------------------------------------------------------------------
# Acquisition: Coverage judgment
# ---------------------------------------------------------------------------

COVERAGE_JUDGMENT_PROMPT = """Agent used the following skill to complete a task:

Skill: "{skill_name}"
Skill steps:
{skill_steps}

Agent's actual execution trajectory:
{trajectory}

Please determine:
1. Which trajectory steps correspond to executing this skill?
2. Which steps are NOT covered by the skill (agent did on its own)?

Respond in JSON:
{{
  "covered_steps": [list of step indices that match the skill],
  "uncovered_steps": [list of step indices not covered],
  "coverage_rate": float between 0 and 1
}}"""

# ---------------------------------------------------------------------------
# Acquisition: Segmentation
# ---------------------------------------------------------------------------

SEGMENTATION_INTERACTIVE_PROMPT = """The following is a successful execution trajectory for an interactive task:

{trajectory}

Segment it into 2-4 high-level strategy phases (NOT individual steps).
Group related steps together. For example, "navigate + pick up object" is ONE phase, not two.

For each phase, provide:
1. The steps included (group multiple related steps)
2. A one-sentence strategy description (e.g., "locate and retrieve the target object")

Respond in JSON:
{{
  "segments": [
    {{"steps": ["step1", "step2", "step3"], "subgoal": "strategy description"}},
    ...
  ]
}}"""

SEGMENTATION_REASONING_PROMPT = """The following is a successful reasoning chain (Chain-of-Thought):

{trajectory}

Segment it into 2-4 high-level reasoning strategy phases (NOT individual calculation steps).
Group related steps together. For example, "set up equation + solve equation" is ONE phase, not two.

For each phase, provide:
1. The reasoning steps included (group multiple related steps)
2. A one-sentence strategy description (e.g., "convert fractions to common denominator and add")

Respond in JSON:
{{
  "segments": [
    {{"steps": ["step1", "step2", "step3"], "subgoal": "strategy description"}},
    ...
  ]
}}"""

# ---------------------------------------------------------------------------
# Acquisition: Alignment (pattern matching)
# ---------------------------------------------------------------------------

ALIGNMENT_PROMPT = """Given the following execution records for task type "{task_type}":

{records}

Find recurring common patterns across these records:
1. Pattern description (generalized, no specific values)
2. How many records contain this pattern (out of {total})
3. Variants of this pattern across records (if any)

Only include patterns that appear in at least 2 records.

Respond in JSON:
{{
  "patterns": [
    {{
      "description": "...",
      "count": int,
      "variants": ["variant1", "variant2"]
    }},
    ...
  ]
}}"""

# ---------------------------------------------------------------------------
# Acquisition: Formalization
# ---------------------------------------------------------------------------

FORMALIZATION_PROMPT = """Convert the following pattern into a reusable skill:

Pattern: {pattern_description}
Confidence: {confidence}
Variants: {variants}
Source task type: {task_type}

Create a skill with:
1. name: short English name (snake_case)
2. description: one sentence explaining when to use this skill
3. precondition: when this skill applies
4. steps: concrete steps (generalized, no specific values)
5. parameters: parameterize variant parts if any

Keep the description concise (under 100 tokens).

Respond in JSON:
{{
  "name": "...",
  "description": "...",
  "precondition": "...",
  "parameters": ["param1: option1 | option2"],
  "steps": ["1. ...", "2. ...", "3. ..."]
}}"""

# ---------------------------------------------------------------------------
# Acquisition: Failure analysis
# ---------------------------------------------------------------------------

FAILURE_ANALYSIS_PROMPT = """Agent attempted but failed the following task:

Task: {task}
Execution trajectory: {trajectory}
Failure point: {failure_point}

Please analyze:
1. Root cause of the failure
2. What to avoid next time (one specific, actionable sentence)
3. Which task types does this lesson apply to?
4. Does the warning contain actionable guidance? (yes/no)

Respond in JSON:
{{
  "root_cause": "...",
  "warning": "...",
  "applicable_task_types": ["type1", "type2"],
  "actionable": true/false
}}"""

# ---------------------------------------------------------------------------
# Acquisition: Skill failure diagnosis
# ---------------------------------------------------------------------------

SKILL_FAILURE_DIAGNOSIS_PROMPT = """An agent used a skill but still failed the task.

Task: {task}
Skill used: "{skill_name}"
Skill steps:
{skill_steps}

Agent's execution:
{trajectory}

Ground truth answer: {ground_truth}
Agent's answer: {agent_answer}

Diagnose: Is the failure because:
A) The skill steps are correct but the model made an execution error (calculation mistake, misread, etc.)
B) The skill steps themselves are wrong, incomplete, or misleading

Respond in JSON:
{{
  "diagnosis": "A" or "B",
  "reason": "brief explanation"
}}"""

# ---------------------------------------------------------------------------
# Acquisition: Skill repair
# ---------------------------------------------------------------------------

SKILL_REPAIR_PROMPT = """The following skill caused a task failure. Please repair it.

Skill name: {skill_name}
Current steps:
{skill_steps}

Current warnings:
{skill_warnings}

Failed task: {task}
What went wrong: {failure_reason}
Ground truth answer: {ground_truth}

Rewrite the skill steps to fix the issue. Keep what works, fix what's broken.
Do NOT make the steps overly long — keep them concise and actionable.

Respond in JSON:
{{
  "steps": ["1. ...", "2. ...", "3. ..."],
  "warnings": ["..."]
}}"""

# ---------------------------------------------------------------------------
# Active: Merge
# ---------------------------------------------------------------------------

MERGE_PROMPT = """The following two skills are similar and should be merged into one:

Skill A:
  Name: {skill_a_name}
  Description: {skill_a_description}
  Steps: {skill_a_steps}
  Warnings: {skill_a_warnings}

Skill B:
  Name: {skill_b_name}
  Description: {skill_b_description}
  Steps: {skill_b_steps}
  Warnings: {skill_b_warnings}

Merge them into a single, more general skill:
1. Parameterize differing parts
2. Combine warnings (deduplicate)
3. Keep description concise

Respond in JSON:
{{
  "name": "...",
  "description": "...",
  "precondition": "...",
  "parameters": ["..."],
  "steps": ["1. ...", "2. ..."],
  "warnings": ["..."]
}}"""

# ---------------------------------------------------------------------------
# Active: Distill
# ---------------------------------------------------------------------------

DISTILL_PROMPT = """The following skill description is too long. Please compress it:

Current skill ({current_tokens} tokens):
  Name: {skill_name}
  Description: {skill_description}
  Steps: {skill_steps}
  Warnings: {skill_warnings}

Target: under {target_tokens} tokens.
Requirements:
1. Keep core steps and key information
2. Remove redundant descriptions
3. Do NOT lose any warnings
4. Keep the name and description unchanged if possible

Respond in JSON:
{{
  "steps": ["1. ...", "2. ..."],
  "warnings": ["..."]
}}"""

# ---------------------------------------------------------------------------
# Archive: Compress for storage
# ---------------------------------------------------------------------------

ARCHIVE_COMPRESS_PROMPT = """Compress the following skill into a one-sentence summary:

Name: {skill_name}
Description: {skill_description}
Steps: {skill_steps}

The summary should capture WHAT the skill does and WHEN to use it,
in one concise sentence (under 30 tokens).

Respond with just the summary text, no JSON."""

# ---------------------------------------------------------------------------
# Task classification (for WebShop)
# ---------------------------------------------------------------------------

TASK_CLASSIFICATION_PROMPT = """Classify the following shopping instruction into a product category:

Instruction: "{instruction}"

Respond with just the category name (e.g., "electronics/headphones", "home/decor", "clothing/shoes").
Keep it to two levels: main_category/sub_category."""
