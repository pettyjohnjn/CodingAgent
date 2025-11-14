
from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentConfig:
    """Configuration for the coding agent."""

    base_dir: str = "experiments"
    max_retries: int = 2  # number of fix attempts after the initial one

    # Security / safety
    forbidden_modules: List[str] = field(
        default_factory=lambda: [
            "os",
            "subprocess",
            "shutil",
            "socket",
            "requests",
        ]
    )
    forbidden_calls: List[str] = field(
        default_factory=lambda: [
            "os.remove",
            "os.rmdir",
            "shutil.rmtree",
        ]
    )

    # Output contract: enforced on the entrypoint file
    require_answer_print: bool = True
    enable_paper_polish: bool = True
    max_experiments: int = 2

    planner_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    coder_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    critic_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    writer_model: str = "openai/gpt-oss-120b"
    editor_model: str = "openai/gpt-oss-120b"
