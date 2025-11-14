from __future__ import annotations

import io
import os
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Any


def run_generated_code(code: str, work_dir: str) -> Dict[str, Any]:
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
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

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