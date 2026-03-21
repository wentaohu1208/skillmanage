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

Please segment it into meaningful parts, where each part completes an independent sub-goal.

For each segment, provide:
1. The steps included
2. A one-sentence sub-goal description

Respond in JSON:
{{
  "segments": [
    {{"steps": ["step1", "step2"], "subgoal": "description"}},
    ...
  ]
}}"""

SEGMENTATION_REASONING_PROMPT = """The following is a successful reasoning chain (Chain-of-Thought):

{trajectory}

Please segment it into meaningful reasoning stages.

For each segment, provide:
1. The reasoning steps included
2. A one-sentence description of the reasoning goal

Respond in JSON:
{{
  "segments": [
    {{"steps": ["step1", "step2"], "subgoal": "description"}},
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
