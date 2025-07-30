# Experimental Branch Description

This branch is the origin source code evaluation implementation, used for experimental evaluation of the performance of each sub-project on 100 samples.

## Sub-project List
- ChatDev-main: HumanEval evaluation
- llm-debate: Multi-agent debate evaluation (AIME, DROP)
- JARVIS/hugginggpt: HuggingGPT evaluation (math, aime, drop, mmlu, bbh, humaneval)

## Using uv for Environment Management

The following are important notes when using `uv` to manage the virtual environment for this project:

- Installation and Version
  - Global installation: `pip install uv` (or `pipx install uv`), **please ensure Python version is within 3.11 range** (see `pyproject.toml`).

- Mirror Source Configuration
  - Alibaba Cloud PyPI mirror has been configured in `pyproject.toml` under `[tool.uv.index]`, `uv install` will automatically use it; if you need to switch back to the official source:
    ```bash
    uv install --index-url https://pypi.org/simple
    ```

- Dependency Installation and Locking
  - Execute `uv install` to create or update `.venv/` based on dependencies.
  - After first installation, run `uv lock` to generate `uv.lock`, and include it in version control to ensure environment reproducibility.

- Running Scripts
  - Execute in the virtual environment:
    ```bash
    uv run python ChatDev-main/Humaneval_benchmark/run_100_tasks.py
    uv run python llm-debate/llm_multiagent_debate/run_all_tests.py
    uv run python JARVIS/hugginggpt/run_benchmark.py
    ```

- Development Dependency Management
  - If you need to install development tools like `ruff`, `pytest`, `black`, etc., you can add them in `pyproject.toml`:
    ```toml
    [tool.uv.dev-dependencies]
    ruff = ">=0.0.280"
    pytest = ">=7.0"
    black = ">=23.0"
    ```
  - Then execute `uv install --dev` to install all development dependencies at once.

## Commands for Running 100 Tasks
```bash
# ChatDev-main: Execute 100 HumanEval tasks
uv run python ChatDev-main/Humaneval_benchmark/run_100_tasks.py

# llm-debate: Execute AIME and DROP full tests
uv run python llm-debate/llm_multiagent_debate/run_all_tests.py

# JARVIS/hugginggpt: Execute evaluations for each dataset sequentially (script defaults to max_samples=100)
uv run python JARVIS/hugginggpt/run_benchmark.py
```

> **Note**:  
> 1. Please configure the environment variable `OPENAI_API_KEY` before running.  
> 2. If you need to evaluate other datasets, you can uncomment the corresponding functions in the script and execute them.

