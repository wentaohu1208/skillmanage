"""ALFWorld benchmark implementation.

Based on patterns from ReAct, Reflexion, and SkillRL repos:
- env.reset() returns (obs_list, info_dict), advance sequentially
- task instruction: '\n'.join(obs[0].split('\n\n')[1:])
- task type: parse from info['extra.gamefile'][0]
- env.step([action]) returns (obs, reward, done, info), batched
- info['won'][0] for task completion check
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .base import InteractiveBenchmark, InteractionMode, TaskInstance, register_benchmark
from .prompts import ALFWORLD_STEP_PROMPT, ALFWORLD_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Task type prefix mapping (from ReAct/Reflexion)
TASK_TYPE_PREFIXES = {
    "pick_and_place": "pick_and_place",
    "pick_clean_then_place": "pick_clean_then_place",
    "pick_heat_then_place": "pick_heat_then_place",
    "pick_cool_then_place": "pick_cool_then_place",
    "look_at_obj": "look_at_obj_in_light",
    "pick_two_obj": "pick_two_obj_and_place",
}

TASK_TYPES = list(set(TASK_TYPE_PREFIXES.values()))


def _parse_task_type(game_file: str) -> str:
    """Parse task type from ALFWorld game file path.

    Game file paths look like:
    .../pick_heat_then_place_in_recep-Egg-None-Microwave-18/trial_.../game.tw-pddl

    Args:
        game_file: Full path to game file.

    Returns:
        Task type string.
    """
    # Extract the game name part: "pick_heat_then_place_in_recep-Egg-..."
    name = "/".join(game_file.split("/")[-3:-1])
    for prefix, task_type in TASK_TYPE_PREFIXES.items():
        if name.startswith(prefix):
            return task_type
    return "unknown"


def _process_observation(obs: str) -> str:
    """Strip navigation prefix from observation (from ReAct/Reflexion).

    Args:
        obs: Raw observation text.

    Returns:
        Cleaned observation.
    """
    if obs.startswith("You arrive at loc "):
        obs = obs[obs.find(". ") + 2 :]
    return obs


@register_benchmark("alfworld")
class ALFWorldBenchmark(InteractiveBenchmark):
    """ALFWorld text-based household environment benchmark.

    6 task types, 134 unseen test tasks. Multi-step interactive.

    Unlike MATH, ALFWorld tasks are NOT pre-loaded. The environment is
    stateful and games are accessed sequentially via env.reset().
    Each call to run_games() creates a fresh environment for the requested split.

    Args:
        data_path: Path to ALFWorld data (ALFWORLD_DATA env var).
        max_steps: Maximum steps per task.
        config_path: Path to base_config.yaml (optional, auto-detected).
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        max_steps: int = 30,
        config_path: Optional[str] = None,
    ) -> None:
        self._data_path = data_path or os.environ.get("ALFWORLD_DATA", "")
        self.max_steps = max_steps
        self._config_path = config_path

        # Current game state (set during run_games / reset_env)
        self._env = None
        self._current_split: Optional[str] = None
        self._current_task_type = ""
        self._current_instruction = ""
        self._won = False
        self._admissible_actions: List[str] = []
        self._last_action = ""
        self._repeat_count = 0

    @property
    def name(self) -> str:
        return "alfworld"

    @property
    def current_task_type(self) -> str:
        """Task type of the current game (set after reset_env)."""
        return self._current_task_type

    @property
    def current_instruction(self) -> str:
        """Task instruction of the current game (set after reset_env)."""
        return self._current_instruction

    def load_tasks(
        self, split: str = "train", limit: Optional[int] = None
    ) -> List[TaskInstance]:
        """Create placeholder TaskInstances for ALFWorld games.

        ALFWorld tasks are accessed sequentially via env.reset(), not randomly.
        This method creates placeholder TaskInstances with game indices.
        The actual instruction and task_type are populated at reset_env time.

        Args:
            split: 'train' or 'test' (test = eval_out_of_distribution).
            limit: Max number of tasks.

        Returns:
            List of TaskInstance placeholders.
        """
        num_games = self._count_games(split)
        if limit:
            num_games = min(num_games, limit)

        tasks = []
        for i in range(num_games):
            tasks.append(TaskInstance(
                task_id=f"alfworld_{split}_{i}",
                instruction="",  # Populated at reset_env time
                task_type="",    # Populated at reset_env time
                ground_truth="done",
                metadata={"game_index": i, "split": split},
            ))

        logger.info("Created %d ALFWorld %s task placeholders", len(tasks), split)
        return tasks

    def build_prompt(self, task: TaskInstance, skills_prompt: str) -> str:
        """Not used for interactive benchmarks."""
        return ""

    def build_system_prompt(self, skills_prompt: str) -> str:
        """Build system prompt with available commands and skills."""
        task_desc = f"\nYour task is: {self._current_instruction}" if self._current_instruction else ""
        return ALFWORLD_SYSTEM_PROMPT.format(skills_prompt=skills_prompt) + task_desc

    def build_step_prompt(
        self, task: TaskInstance, observation: str, history: List[str]
    ) -> str:
        """Build prompt for one interaction step."""
        history_str = "\n".join(history[-20:]) if history else "(none)"
        admissible_str = ", ".join(self._admissible_actions) if self._admissible_actions else "look"
        return ALFWORLD_STEP_PROMPT.format(
            task_instruction=self._current_instruction,
            history=history_str,
            observation=observation,
            admissible_actions=admissible_str,
        )

    def check_answer(
        self, task: TaskInstance, agent_output: str
    ) -> Tuple[bool, float]:
        """Check if task was completed."""
        return self._won, 1.0 if self._won else 0.0

    def extract_trajectory(self, agent_output: str) -> List[str]:
        """Extract action steps from multi-step output."""
        lines = agent_output.strip().split("\n")
        actions = [
            line.replace("Action: ", "").strip()
            for line in lines
            if line.startswith("Action: ")
        ]
        return actions

    def get_task_types(self) -> List[str]:
        """Return the 6 ALFWorld task types."""
        return list(TASK_TYPES)

    # ------------------------------------------------------------------
    # InteractiveBenchmark interface
    # ------------------------------------------------------------------

    def reset_env(self, task: TaskInstance) -> str:
        """Reset environment to the next game.

        ALFWorld env advances sequentially. Each call returns the next game.
        Populates current_task_type and current_instruction.

        Returns:
            Initial observation text (task goal, without receptacle listing).
        """
        split = task.metadata.get("split", "eval_out_of_distribution")
        env = self._ensure_env(split)

        obs, info = env.reset()

        # Extract instruction: strip first paragraph (receptacle list),
        # keep task goal (from ReAct/Reflexion pattern)
        raw_obs = obs[0] if isinstance(obs, list) else str(obs)
        self._current_instruction = "\n".join(raw_obs.split("\n\n")[1:])

        # Extract task type from game file path
        game_file = info.get("extra.gamefile", [""])[0] if isinstance(info, dict) else ""
        self._current_task_type = _parse_task_type(game_file)

        self._won = False
        self._last_action = ""
        self._repeat_count = 0

        # Capture initial admissible actions
        raw_admissible = info.get("admissible_commands", []) if isinstance(info, dict) else []
        admissible = raw_admissible[0] if raw_admissible else []
        self._admissible_actions = [a for a in admissible if a != "help"] if isinstance(admissible, list) else []

        logger.debug(
            "ALFWorld reset: type=%s, instruction=%s",
            self._current_task_type,
            self._current_instruction[:60],
        )
        return self._current_instruction

    def step(self, action: str) -> Tuple[str, float, bool]:
        """Execute one action in the environment.

        Handles think: interception, Action: prefix stripping, and loop detection.

        Returns:
            (observation, reward, done)
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset_env first.")

        # Strip common LLM prefixes
        action = action.strip()
        if action.lower().startswith("action:"):
            action = action[len("action:"):].strip()

        # Empty action guard
        if not action:
            return "Invalid action.", 0.0, False

        # Intercept think: actions (ReAct pattern) — don't send to env
        if action.lower().startswith("think:"):
            return "OK.", 0.0, False

        # Loop detection: if same action repeated 3+ times, force termination
        if action == self._last_action:
            self._repeat_count += 1
            if self._repeat_count >= 3:
                logger.warning("Loop detected: '%s' repeated %d times. Terminating.", action, self._repeat_count)
                return "Nothing happens. (terminated due to repeated action)", 0.0, True
        else:
            self._last_action = action
            self._repeat_count = 1

        obs, reward, done, info = self._env.step([action])

        observation = obs[0] if isinstance(obs, list) else str(obs)
        observation = _process_observation(observation)

        is_done = done[0] if isinstance(done, list) else bool(done)
        self._won = info["won"][0] if isinstance(info, dict) and "won" in info else False

        # Update admissible actions for next step
        raw_admissible = info.get("admissible_commands", []) if isinstance(info, dict) else []
        admissible = raw_admissible[0] if raw_admissible else []
        self._admissible_actions = [a for a in admissible if a != "help"] if isinstance(admissible, list) else []

        return observation, float(self._won), is_done

    # ------------------------------------------------------------------
    # Environment management
    # ------------------------------------------------------------------

    def _ensure_env(self, split: str):
        """Get or create the ALFWorld environment for a split.

        Creates a NEW environment each time the split changes.
        """
        # Map split names
        if split == "test":
            train_eval = "eval_out_of_distribution"
        elif split == "train":
            train_eval = "train"
        else:
            train_eval = split

        # If env exists for a different split, close and recreate
        if self._env is not None and getattr(self, "_current_split", None) != train_eval:
            self.close()

        if self._env is None:
            self._env = self._create_env(train_eval)
            self._current_split = train_eval

        return self._env

    def _create_env(self, train_eval: str):
        """Create a new ALFWorld environment."""
        try:
            import alfworld
            import alfworld.agents.environment
        except ImportError:
            raise ImportError(
                "alfworld is required. Install with: pip install alfworld[full] && alfworld-download"
            )

        if self._data_path:
            os.environ["ALFWORLD_DATA"] = self._data_path

        if self._config_path:
            config_path = self._config_path
        else:
            # Auto-locate base_config.yaml from alfworld package
            config_path = os.path.join(os.path.dirname(alfworld.__file__), "configs", "base_config.yaml")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        env_type = config["env"]["type"]
        env = getattr(alfworld.agents.environment, env_type)(config, train_eval=train_eval)
        env = env.init_env(batch_size=1)

        logger.info("Created ALFWorld env: split=%s, config=%s", train_eval, config_path)
        return env

    def _count_games(self, split: str) -> int:
        """Count available games for a split without creating the full env."""
        # Known counts from ALFWorld paper
        counts = {
            "train": 3553,
            "test": 134,
            "eval_out_of_distribution": 134,
            "eval_in_distribution": 140,
        }
        return counts.get(split, 134)

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
            self._current_split = None
