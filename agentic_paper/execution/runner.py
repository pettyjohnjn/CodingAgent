from __future__ import annotations

import io
import os
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess


def run_generated_code(code: str, 
                       work_dir: str, 
                       entrypoint_name: str = "main.py",
                       env_tool: str = "conda",
                       use_isolated_env: bool = True,
                       env_prefix: Optional[str] = None,
                       ) -> Dict[str, Any]:
    """
    Execute the combined project code in an isolated namespace, capture stdout,
    and (if present) call a top-level `main()`.

    Parameters
    ----------
    code : str
        The full combined Python source to execute (all files concatenated).
    work_dir : str
        Directory to run in. Any plt.savefig(...) calls will write here.

    Returns
    -------
    dict with keys:
        - success: bool
        - stdout: str
        - locals: list[str]
        - error: str | None
        - experiment_dir: str (equal to work_dir)
    """

    work_path = Path(work_dir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    # write the combined code to a file and locate environment.yaml file
    entrypoint_path = work_path / entrypoint_name
    with entrypoint_path.open("w", encoding="utf-8") as f:
        f.write(code)

    env_yaml_path = work_path / "environment.yaml"
    if env_prefix is None:
        env_prefix_path = work_path / "env"
    else:
        env_prefix_path = Path(env_prefix)
    env_prefix_path = env_prefix_path.resolve()
    env_prefix_str = str(env_prefix_path)


    if use_isolated_env and env_yaml_path.exists():
        stdout_text = ""
        stderr_text = ""
        error_text: Optional[str] = None
        success = False

        # create env
        try:
            if env_tool == "conda":
                create_cmd = [
                    "conda",
                    "env",
                    "create",
                    "-p",
                    env_prefix_str,
                    "-f",
                    str(env_yaml_path),
                ]
            else:
                raise ValueError(f"Unsupported env_tool: {env_tool}")

            create_proc = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
            )


            if create_proc.returncode != 0:
                error_text = (
                    f"Environment creation failed with return code "
                    f"{create_proc.returncode}:\n{create_proc.stderr}"
                )
                return {
                    "success": False,
                    "stdout": create_proc.stdout,
                    "stderr": create_proc.stderr,
                    "locals": [],
                    "error": error_text,
                    "experiment_dir": str(work_path),
                }
            

            # determine python executable inside the env
            if os.name == "nt":
                python_exe = env_prefix_path / "python.exe"
            else:
                python_exe = env_prefix_path / "bin" / "python"
            

            # run code inside the env
            run_cmd = [
                str(python_exe),
                str(entrypoint_path),
            ]

            run_proc = subprocess.run(
                run_cmd,
                cwd=str(work_path),
                capture_output=True,
                text=True,
            )

            stdout_text = run_proc.stdout
            stderr_text = run_proc.stderr
            success = run_proc.returncode == 0
            if not success:
                error_text = (
                    f"Script failed with return code {run_proc.returncode}.\n"
                    f"stderr:\n{stderr_text}"
                )
        except Exception:
            error_text = traceback.format_exc()
            success = False

        return {
            "success": success,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "locals": [],
            "error": error_text,
            "experiment_dir": str(work_path),
        }


    # fallback if we don't want an isolated env

    run_globals: Dict[str, Any] = {}
    # Make the executed code think it's a __main__ module.
    run_globals["__name__"] = "__main__"

    stdout_buf = io.StringIO()
    error_text = None
    success = False

    cwd_before = os.getcwd()
    try:
        os.chdir(work_path)

        with redirect_stdout(stdout_buf):
            # Execute all definitions
            exec(code, run_globals)

            # If a main() exists, call it explicitly
            main_fn = run_globals.get("main")
            if callable(main_fn):
                main_fn()

        success = True
    except Exception:
        error_text = traceback.format_exc()
        success = False
    finally:
        os.chdir(cwd_before)

    stdout_text = stdout_buf.getvalue()
    locals_list = sorted(k for k in run_globals.keys() if not k.startswith("__"))

    return {
        "success": success,
        "stdout": stdout_text,
        "locals": locals_list,
        "error": error_text,
        # Where code actually ran and figures were written
        "experiment_dir": str(work_path),
    }