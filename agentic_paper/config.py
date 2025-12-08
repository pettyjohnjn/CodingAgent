
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentConfig:
    """Configuration for the coding agent."""

    base_dir: str = "experiments"
    max_retries: int = 2  # number of fix attempts after the initial one
    use_isolated_env: bool = True
    env_tool: str = "conda"

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
    enable_github_publish: bool = True
    github_visibility: str = "public"
    github_token: Optional[str] = None
    github_ignore_patterns: List[str] = field(default_factory=list)

    # planner_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    # coder_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    # critic_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

    planner_model: str = "openai/gpt-oss-120b"
    coder_model: str = "openai/gpt-oss-120b"
    critic_model: str = "openai/gpt-oss-120b"
    writer_model: str = "openai/gpt-oss-120b"
    editor_model: str = "openai/gpt-oss-120b"

    #Reproduce Errors

    #No code/computational artifacts available by the author 
    no_code_saved: bool = False

    #Code repository exists and is accessible, but may not include the necessary instructions or environment details.
    no_env_saved: bool = False

    #Code does not compile
    errors_in_code: bool = False

    #Environment available but cannot build
    error_in_env: bool = False

    #Results don't match the paper 
    inconsistent_results: bool = True

