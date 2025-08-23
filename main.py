#!/usr/bin/env python3
import os
import re
import sys
import json
import asyncio
import tempfile
import subprocess
import typing as t
from dataclasses import dataclass, field

# pip install groq requests
from groq import Groq
import requests

###############################################################################
# Configuration (2025-08 snapshot) — user-preferred families prioritized
###############################################################################
DEFAULT_MODELS = "auto"

PRODUCTION_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-guard-4-12b",
]

# Includes the models you asked to prioritize: DeepSeek R1 Distill, Qwen 3, Llama 4 Maverick, Kimi K2
PREVIEW_MODELS = [
    "deepseek-r1-distill-llama-70b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "moonshotai/kimi-k2-instruct",
    "mixtral-8x7b-32768",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama2-70b-4096",
    "gemma-7b-it",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "playai-tts",
    "playai-tts-arabic",
    "moonshotai/kimi-k2-instruct",  # (already listed above; kept here in case of set ops)
]

# === Preference order updates ===
# Fan-out now prefers: DeepSeek R1 Distill → Qwen 3 → Llama 4 Maverick → Kimi K2, then others.
FANOUT_PREFERENCE = [
    "qwen/qwen3-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "moonshotai/kimi-k2-instruct",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "llama3-70b-8192",
]

# Synthesis/repair now prefers: DeepSeek R1 Distill → Qwen 3 → Llama 4 Maverick → Kimi K2, then others.
SYNTH_PREFERENCE = [
    "qwen/qwen3-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "moonshotai/kimi-k2-instruct",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
]

# Tool-use models (kept for routing when you pass --prefer-tool-use or your task hints at tools/JSON)
TOOL_USE_PREFERENCE = [
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
]

SYSTEM_PROMPT_BASE = """You are a senior software engineer.
You generate a SINGLE self-contained Python module that solves the user's task.
- Output ONLY code, inside one Python code block. No extra text.
- The module must be importable as 'solution'.
- Expose a clear entrypoint function named `solve(*args, **kwargs)` if appropriate.
- If the task asks for CLI behavior, include a `if __name__ == "__main__":` guard.
- Keep dependencies to the standard library unless explicitly required.
"""

SYSTEM_TEST_PROMPT = """You are a senior software engineer writing pytest tests for a module that solves the user's task.
- Output ONLY code, inside one Python code block. No extra text.
- The tests must import the module under test as `solution`.
- Use pytest style assertions.
"""

SYSTEM_TEST_SYNTH_PROMPT = """You are a synthesis engine. You are given multiple candidate Python test modules for the user's task.
Produce ONE merged test module using pytest.
- Output ONLY code, inside one Python code block.
"""

SYSTEM_SYNTH_PROMPT = """You are a synthesis engine. You are given multiple candidate Python modules that attempt the same task.
Produce ONE merged, best-of-all-worlds module, fixing obvious bugs or omissions.
- Output ONLY code, inside one Python code block, importable as 'solution'.
- Prefer clarity, correctness, and minimal dependencies.
- Keep the interface consistent with the task.
"""

SYSTEM_REPAIR_PROMPT = """You are a code repair agent. You receive:
1) The user's original task.
2) The current failing module.
3) The exact test output / traceback.

Your job: Return a corrected, complete Python module that passes the tests.
- Output ONLY code, inside one Python code block.
- Keep the public interface the same unless a change is strictly necessary.
- Avoid adding new external dependencies.
"""

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_think_tags(text: str) -> str:
    """Remove model reasoning tags like <think>...</think> from text."""
    return THINK_TAG_RE.sub("", text)

# Post-pass feature ideation and enhancement
SYSTEM_FEATURE_SUGGEST_PROMPT = """You are a product-minded senior engineer.
Given the user's original task and the current correct solution module, propose ONE
small, concrete, backward-compatible feature that adds clear user value and can be
validated with simple tests. Keep it focused and incremental.

Return only the feature description as one concise sentence. No lists, no code, no preface.
"""

SYSTEM_ENHANCE_PROMPT = """You are a senior software engineer improving an existing module.
You will be given:
1) The user's original task
2) The current working module
3) A new feature to implement (must remain backward compatible)

Return a COMPLETE Python module implementing the feature while preserving existing behavior.
- Output ONLY code, inside one Python code block, importable as 'solution'.
- Keep public API stable (e.g., keep `solve()` if present). Avoid new external deps.
"""

###############################################################################
# Data structures
###############################################################################

@dataclass
class Generation:
    model: str
    code: str
    token_count: t.Optional[int] = None

@dataclass
class SynthesisResult:
    code: str
    meta: dict = field(default_factory=dict)

@dataclass
class TestResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int

###############################################################################
# Groq client + model discovery
###############################################################################

def make_groq_client(api_key: t.Optional[str] = None) -> Groq:
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)
    return Groq(api_key=api_key)

def list_available_models(client: Groq) -> t.List[str]:
    """
    Try SDK first, then raw HTTP to the /openai/v1/models endpoint.
    Returns a list of model IDs (strings).
    """
    api_key = os.getenv("GROQ_API_KEY")
    # 1) SDK path
    try:
        models_obj = client.models.list()
        ids: t.List[str] = []
        if hasattr(models_obj, "data"):
            for m in models_obj.data:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                if mid:
                    ids.append(mid)
        else:
            for m in models_obj.get("data", []):
                if "id" in m:
                    ids.append(m["id"])
        if ids:
            return sorted(set(ids))
    except Exception as e:
        print(f"[models] SDK listing failed, falling back to HTTP: {e}", file=sys.stderr)

    # 2) Raw HTTP
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = [m["id"] for m in data.get("data", []) if "id" in m]
        return sorted(set(ids))
    except Exception as e:
        print(f"[models] HTTP listing failed: {e}", file=sys.stderr)
        return []

def categorize_models(available: t.Iterable[str]) -> t.Tuple[t.Set[str], t.Set[str], t.Set[str]]:
    """
    Returns (production, preview, tool_use) present in `available`.
    """
    avail = set(available)
    prod = {m for m in avail if m in set(PRODUCTION_MODELS)}
    prev = {m for m in avail if m in set(PREVIEW_MODELS)}
    tool = {m for m in avail if m in set(TOOL_USE_PREFERENCE)}
    return prod, prev, tool

def auto_select_fanout(available: t.Set[str], want: int = 3) -> t.List[str]:
    chosen: t.List[str] = []
    for name in FANOUT_PREFERENCE:
        if name in available:
            chosen.append(name)
        if len(chosen) >= want:
            break
    if len(chosen) < want:
        for m in available:
            if m not in chosen:
                chosen.append(m)
            if len(chosen) >= want:
                break
    return chosen

def auto_select_synth(available: t.Set[str]) -> str:
    for name in SYNTH_PREFERENCE:
        if name in available:
            return name
    return next(iter(available), "llama-3.3-70b-versatile")

def auto_select_tool_use(available: t.Set[str]) -> t.Optional[str]:
    for name in TOOL_USE_PREFERENCE:
        if name in available:
            return name
    return None

###############################################################################
# Chat helpers
###############################################################################

def extract_code_block(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    code = m.group(1) if m else text
    code = strip_think_tags(code)
    return code.strip()

def chat_completion(client: Groq, model: str, system: str, user: str,
                    temperature: float = 0.2, seed: t.Optional[int] = 42,
                    max_output_tokens: t.Optional[int] = None) -> str:
    comp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        seed=seed,
        max_tokens=max_output_tokens,
    )
    return comp.choices[0].message.content

async def generate_from_model(client: Groq, model: str, task_prompt: str,
                              temperature: float, seed: t.Optional[int],
                              system_prompt: str) -> Generation:
    loop = asyncio.get_event_loop()
    def _call():
        text = chat_completion(client, model, system_prompt, task_prompt, temperature, seed)
        return extract_code_block(text)
    code = await loop.run_in_executor(None, _call)
    return Generation(model=model, code=code)

async def parallel_generate(client: Groq, models: t.List[str], task_prompt: str,
                            temperature: float, seed: t.Optional[int],
                            system_prompt: str) -> t.List[Generation]:
    tasks = [generate_from_model(client, m, task_prompt, temperature, seed, system_prompt) for m in models]
    return await asyncio.gather(*tasks)

def synthesize(client: Groq, generations: t.List[Generation], task_prompt: str,
               synth_model: str, temperature: float, seed: t.Optional[int],
               system_prompt: str) -> SynthesisResult:
    bundle = []
    for g in generations:
        bundle.append(f"# === Candidate from {g.model} ===\n{g.code}\n")
    synthesis_input = f"## USER TASK\n{task_prompt}\n\n## CANDIDATE MODULES\n" + "\n\n".join(bundle)
    text = chat_completion(client, synth_model, system_prompt,
                           synthesis_input, temperature=temperature, seed=seed)
    code = extract_code_block(text)
    return SynthesisResult(code=code, meta={"models": [g.model for g in generations], "synth_model": synth_model})

def repair_with_feedback(client: Groq, repair_model: str, failing_code: str, task_prompt: str,
                         test_output: str, temperature: float, seed: t.Optional[int]) -> str:
    user_msg = (
        f"## USER TASK\n{task_prompt}\n\n"
        f"## CURRENT FAILING MODULE\n```python\n{failing_code}\n```\n\n"
        f"## TEST OUTPUT / TRACEBACK\n```\n{test_output}\n```\n"
    )
    text = chat_completion(client, repair_model, SYSTEM_REPAIR_PROMPT,
                           user_msg, temperature=temperature, seed=seed)
    return extract_code_block(text)

def suggest_new_feature(client: Groq, model: str, task_prompt: str, current_code: str,
                        temperature: float, seed: t.Optional[int]) -> str:
    user_msg = (
        f"## USER TASK\n{task_prompt}\n\n"
        f"## CURRENT SOLUTION MODULE\n```python\n{current_code}\n```\n"
    )
    suggestion = chat_completion(
        client, model, SYSTEM_FEATURE_SUGGEST_PROMPT, user_msg,
        temperature=max(0.2, min(0.7, temperature)), seed=seed
    )
    # Strip any accidental formatting
    return suggestion.strip().strip('`').strip()

def enhance_with_feature(client: Groq, model: str, task_prompt: str, current_code: str,
                         feature_text: str, temperature: float, seed: t.Optional[int]) -> str:
    user_msg = (
        f"## USER TASK\n{task_prompt}\n\n"
        f"## CURRENT WORKING MODULE\n```python\n{current_code}\n```\n\n"
        f"## NEW FEATURE TO IMPLEMENT\n{feature_text}\n"
    )
    enhanced = chat_completion(
        client, model, SYSTEM_ENHANCE_PROMPT, user_msg,
        temperature=max(0.2, min(0.6, temperature)), seed=seed
    )
    return extract_code_block(enhanced)

###############################################################################
# Testing sandbox
###############################################################################

def run_tests_on_code(solution_code: str, tests_code: str, timeout_sec: int = 25) -> TestResult:
    tmp = tempfile.TemporaryDirectory()
    td = tmp.name
    sol_path = os.path.join(td, "solution.py")
    tst_path = os.path.join(td, "tests.py")

    with open(sol_path, "w", encoding="utf-8") as f:
        f.write(strip_think_tags(solution_code))
    with open(tst_path, "w", encoding="utf-8") as f:
        f.write(strip_think_tags(tests_code))

    env = os.environ.copy()
    env["PYTHONPATH"] = td + os.pathsep + env.get("PYTHONPATH", "")

    # First try running tests via pytest so that any test functions are actually
    # executed.  If pytest is missing or collects no tests, fall back to running
    # the test file directly as a script.
    cmd_pytest = [sys.executable, "-X", "faulthandler", "-m", "pytest", tst_path]
    cmd_script = [sys.executable, "-X", "faulthandler", tst_path]

    try:
        proc = subprocess.run(
            cmd_pytest,
            cwd=td,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            text=True,
        )

        # pytest exits with code 5 when no tests are collected. Also handle the
        # case where pytest isn't installed.
        if proc.returncode == 5 or "No module named pytest" in proc.stderr:
            proc = subprocess.run(
                cmd_script,
                cwd=td,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                text=True,
            )

        return TestResult(
            passed=(proc.returncode == 0),
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired as e:
        return TestResult(
            passed=False,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + "\nTEST TIMEOUT",
            returncode=124,
        )
    finally:
        tmp.cleanup()

###############################################################################
# Agent Orchestration (prompt-based task; inline tests optional)
###############################################################################

def build_user_prompt(task_text: str, existing_code: t.Optional[str] = None) -> str:
    prompt = "TASK:\n" + task_text.strip() + "\n\nDeliver a single Python module as specified."
    if existing_code:
        prompt += (
            "\n\nExisting solution module:\n```python\n"
            + existing_code.strip()
            + "\n```"
        )
    return prompt

def build_tests_prompt(task_text: str, existing_code: t.Optional[str] = None) -> str:
    prompt = (
        "TASK:\n" + task_text.strip() + "\n\n"
        "Write pytest unit tests for a module solving this task. "
        "Import the solution module as 'solution'."
    )
    if existing_code:
        prompt += (
            "\n\nExisting solution module:\n```python\n"
            + existing_code.strip()
            + "\n```"
        )
    return prompt

def parse_cli_args(argv: t.List[str]) -> dict:
    import argparse
    p = argparse.ArgumentParser(description="Groq multi-model coding agent (prompt-based)")
    p.add_argument("--task", type=str, default=None,
                   help="Plain text TASK prompt (required unless piping from stdin).")
    p.add_argument("--tests", type=str, default=None,
                   help="Inline Python test code (optional). If omitted, tests are generated by the agent.")
    p.add_argument("--models", type=str, default=DEFAULT_MODELS,
                   help='Comma-separated list of Groq models OR "auto" to query and pick.')
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iters", type=int, default=4, help="Max repair iterations")
    p.add_argument("--timeout", type=int, default=25, help="Per-test timeout (sec)")
    p.add_argument("--print-intermediate", action="store_true",
                   help="Print intermediate candidate snippets (truncated)")
    p.add_argument("--out", type=str, default="solution.py", help="Where to write the final solution")
    p.add_argument("--dry-run", action="store_true", help="Skip executing generated code (for CI review)")
    p.add_argument("--fanout", type=int, default=3, help="How many models to fan out (when --models=auto)")
    p.add_argument("--prefer-tool-use", action="store_true",
                   help="Prefer tool-use models when task hints at tools/JSON/function-calling.")
    p.add_argument("--disable-post-feature", action="store_true",
                   help="Disable post-pass feature suggestion and feedback.")
    p.add_argument("--disable-enhance", action="store_true",
                   help="Disable enhancement attempt after suggesting a feature.")
    p.add_argument("--timeout-explanation", type=str, default=None,
                   help="If tests timeout, include this human explanation of the true issue in the repair feedback.")
    p.add_argument("--max-feature-iters", type=int, default=10,
                   help="Maximum number of successive feature suggestion/enhancement cycles after tests first pass.")
    p.add_argument("--enhance-repair-iters", type=int, default=2,
                   help="Max repair iterations to try if an enhanced solution fails tests before giving up.")
    args = p.parse_args(argv)
    return vars(args)

def get_task_from_args_or_stdin(task_arg: t.Optional[str]) -> str:
    if task_arg and task_arg.strip():
        return task_arg
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    print("ERROR: No --task provided and no stdin data. Provide --task \"...\" or pipe a prompt.", file=sys.stderr)
    sys.exit(2)

def get_tests_inline_or_default(tests_arg: t.Optional[str]) -> t.Optional[str]:
    return tests_arg if (tests_arg and tests_arg.strip()) else None

def truncate(s: str, n: int = 600) -> str:
    s = s.replace("\n", " ")[:n]
    return s + ("…" if len(s) == n else "")

def task_wants_tool_use(task_text: str) -> bool:
    hints = [
        "tool-use", "tool use", "function call", "function-call", "tools:",
        "json schema", "json output", "structured output", "openai tools",
        "assistant tools", "function calling", "tool calling"
    ]
    tt = task_text.lower()
    return any(h in tt for h in hints)

def resolve_models(client: Groq, models_arg: str, fanout_n: int,
                   prefer_tool_use: bool, task_text: str) -> t.Tuple[t.List[str], str]:
    if models_arg.strip().lower() != "auto":
        provided = [m.strip() for m in models_arg.split(",") if m.strip()]
        if provided:
            synth = provided[-1]
            return (provided, synth)
        else:
            print("[agent] --models provided but empty; falling back to auto.", file=sys.stderr)

    available = list_available_models(client)
    if not available:
        print("[agent] Could not fetch models; using built-in defaults.", file=sys.stderr)
        fallback_fan = [
            "deepseek-r1-distill-llama-70b",
            "qwen/qwen3-32b",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ][:fanout_n]
        synth = "deepseek-r1-distill-llama-70b"
        return (fallback_fan, synth)

    prod, prev, tool = categorize_models(available)
    all_avail = set(available)

    print(f"[agent] Discovered {len(available)} Groq models.", file=sys.stderr)
    if prod: print("[agent] Production models: " + ", ".join(sorted(prod)), file=sys.stderr)
    if prev: print("[agent] Preview models: " + ", ".join(sorted(prev)), file=sys.stderr)
    if tool: print("[agent] Tool-use models: " + ", ".join(sorted(tool)), file=sys.stderr)

    selected_tool_model: t.Optional[str] = None
    if prefer_tool_use or task_wants_tool_use(task_text):
        selected_tool_model = auto_select_tool_use(all_avail)
        if selected_tool_model:
            print(f"[agent] Routing synth/repair to tool-use model: {selected_tool_model}", file=sys.stderr)

    fanout = auto_select_fanout(all_avail, want=fanout_n)
    synth = selected_tool_model or auto_select_synth(all_avail)

    print(f"[agent] Fan-out models: {', '.join(fanout)}", file=sys.stderr)
    print(f"[agent] Synthesis/repair model: {synth}", file=sys.stderr)
    return (fanout, synth)

def main(argv: t.List[str]) -> int:
    args = parse_cli_args(argv)
    models_arg = args["models"]
    temperature = args["temperature"]
    seed = args["seed"]
    max_iters = args["max_iters"]
    timeout = args["timeout"]
    out_path = args["out"]
    dry_run = args["dry_run"]
    print_intermediate = args["print_intermediate"]
    fanout_n = args["fanout"]
    prefer_tool_use = args["prefer_tool_use"]
    disable_post_feature = args["disable_post_feature"]
    disable_enhance = args["disable_enhance"]
    timeout_explanation: t.Optional[str] = args["timeout_explanation"]
    max_feature_iters: int = max(0, int(args["max_feature_iters"]))
    enhance_repair_iters: int = max(0, int(args["enhance_repair_iters"]))

    task_text = get_task_from_args_or_stdin(args["task"])
    tests_code = get_tests_inline_or_default(args["tests"])

    solution_code: t.Optional[str] = None
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                solution_code = strip_think_tags(f.read())
            print(f"[agent] Starting from existing {out_path}", file=sys.stderr)
        except Exception as exc:
            print(f"[agent] Warning: could not read {out_path}: {exc}", file=sys.stderr)
            solution_code = None

    user_prompt = build_user_prompt(task_text, solution_code)
    tests_prompt = build_tests_prompt(task_text, solution_code)
    client = make_groq_client()

    # Discover models and decide fan-out + synth with the updated preferences
    fanout_models, synth_model = resolve_models(
        client, models_arg, fanout_n, prefer_tool_use, task_text=task_text
    )

    if not tests_code:
        print(f"[agent] Generating tests with {len(fanout_models)} model(s): {', '.join(fanout_models)}", file=sys.stderr)
        test_gens: t.List[Generation] = asyncio.run(
            parallel_generate(client, fanout_models, tests_prompt, temperature, seed, SYSTEM_TEST_PROMPT)
        )
        if print_intermediate:
            for g in test_gens:
                print(f"\n--- Test candidate from {g.model} (trunc) ---\n{truncate(g.code)}\n", file=sys.stderr)
        print("[agent] Synthesizing merged tests...", file=sys.stderr)
        test_synth = synthesize(client, test_gens, tests_prompt, synth_model, temperature, seed, SYSTEM_TEST_SYNTH_PROMPT)
        tests_code = test_synth.code

    if solution_code is None:
        # 1) Fan-out generation in parallel
        print(f"[agent] Generating with {len(fanout_models)} model(s): {', '.join(fanout_models)}", file=sys.stderr)
        gens: t.List[Generation] = asyncio.run(
            parallel_generate(client, fanout_models, user_prompt, temperature, seed, SYSTEM_PROMPT_BASE)
        )
        if print_intermediate:
            for g in gens:
                print(f"\n--- Candidate from {g.model} (trunc) ---\n{truncate(g.code)}\n", file=sys.stderr)

        # 2) Synthesize a single best module
        print("[agent] Synthesizing merged solution...", file=sys.stderr)
        synth = synthesize(client, gens, user_prompt, synth_model, temperature, seed, SYSTEM_SYNTH_PROMPT)
        solution_code = synth.code

    # Attempt + repair loop
    attempt = 0
    while True:
        attempt += 1
        print(f"[agent] Attempt #{attempt}: testing module...", file=sys.stderr)

        if dry_run:
            print("[agent] DRY RUN: Skipping execution of generated code.", file=sys.stderr)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(strip_think_tags(solution_code))
            print(f"[agent] Wrote solution to {out_path}", file=sys.stderr)
            return 0

        test_result = run_tests_on_code(solution_code, tests_code, timeout_sec=timeout)

        if test_result.passed:
            # Save the currently passing solution first
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(strip_think_tags(solution_code))
            print(f"[agent] ✅ Tests passed. Solution saved to {out_path}", file=sys.stderr)

            # Post-pass feature suggestion and optional repeated enhancement cycles
            if not disable_post_feature:
                try:
                    cycles = 0
                    current_code = solution_code
                    while cycles < max_feature_iters:
                        feature = suggest_new_feature(
                            client=client,
                            model=synth_model,
                            task_prompt=user_prompt,
                            current_code=current_code,
                            temperature=temperature,
                            seed=seed,
                        )
                        print(f"[agent] 💡 Suggested new feature (iter {cycles+1}/{max_feature_iters}): {feature}", file=sys.stderr)

                        if disable_enhance:
                            break

                        print("[agent] Enhancing solution with suggested feature...", file=sys.stderr)
                        enhanced_code = enhance_with_feature(
                            client=client,
                            model=synth_model,
                            task_prompt=user_prompt,
                            current_code=current_code,
                            feature_text=feature,
                            temperature=temperature,
                            seed=seed,
                        )
                        enhanced_result = run_tests_on_code(enhanced_code, tests_code, timeout_sec=timeout)
                        if enhanced_result.passed:
                            # Adopt enhanced code and persist
                            current_code = enhanced_code
                            with open(out_path, "w", encoding="utf-8") as f:
                                f.write(strip_think_tags(current_code))
                            print(f"[agent] ✅ Enhanced solution passes. Updated {out_path}", file=sys.stderr)
                            cycles += 1
                            # continue loop for next feature suggestion
                            continue
                        else:
                            # Attempt to repair the enhanced code using the failure output
                            print(f"[agent] ❌ Enhanced attempt failed. Trying up to {enhance_repair_iters} repair iteration(s)...", file=sys.stderr)
                            repaired_code = enhanced_code
                            repair_attempt = 0
                            while repair_attempt < enhance_repair_iters:
                                repair_attempt += 1
                                feedback_blob = (
                                    f"STDOUT:\n{enhanced_result.stdout}\n\nSTDERR:\n{enhanced_result.stderr}\n\n"
                                    f"RETURNCODE: {enhanced_result.returncode}"
                                )
                                if (enhanced_result.returncode == 124 or "TEST TIMEOUT" in enhanced_result.stderr) and timeout_explanation:
                                    feedback_blob += "\n\nUSER TIMEOUT EXPLANATION:\n" + timeout_explanation.strip()

                                repaired_code = repair_with_feedback(
                                    client=client,
                                    repair_model=synth_model,
                                    failing_code=repaired_code,
                                    task_prompt=user_prompt,
                                    test_output=feedback_blob,
                                    temperature=min(0.6, temperature + 0.2),
                                    seed=seed,
                                )
                                enhanced_result = run_tests_on_code(repaired_code, tests_code, timeout_sec=timeout)
                                if enhanced_result.passed:
                                    current_code = repaired_code
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        f.write(strip_think_tags(current_code))
                                    print(f"[agent] ✅ Enhanced repair succeeded on attempt {repair_attempt}. Updated {out_path}", file=sys.stderr)
                                    cycles += 1
                                    break

                            if not enhanced_result.passed:
                                # Keep the last passing version, write failed enhanced attempt for review
                                enh_fail_out = out_path.rsplit('.', 1)[0] + f".enhanced.iter{cycles+1}.failing.py"
                                with open(enh_fail_out, "w", encoding="utf-8") as f:
                                    f.write(strip_think_tags(repaired_code))
                                print(
                                    f"[agent] ⚠️ Enhanced attempt (with repairs) failed on iteration {cycles+1}. Kept previous passing version. Failing enhanced saved to {enh_fail_out}",
                                    file=sys.stderr,
                                )
                                print(enhanced_result.stdout, file=sys.stderr)
                                print(enhanced_result.stderr, file=sys.stderr)
                                break

                    # Set solution_code to the latest passing code (may be unchanged)
                    solution_code = current_code
                except Exception as e:
                    print(f"[agent] Post-pass feature flow encountered an error (skipping): {e}", file=sys.stderr)
                    # Keep the original passing code that is already saved

            # No post-feature flow requested; we are done.
            return 0

        print("[agent] ❌ Tests failed. Feeding back errors for repair.", file=sys.stderr)
        feedback_blob = (
            f"STDOUT:\n{test_result.stdout}\n\nSTDERR:\n{test_result.stderr}\n\n"
            f"RETURNCODE: {test_result.returncode}"
        )
        # If this was a timeout and the user provided extra context, append it
        if (test_result.returncode == 124 or "TEST TIMEOUT" in test_result.stderr) and timeout_explanation:
            feedback_blob += "\n\nUSER TIMEOUT EXPLANATION:\n" + timeout_explanation.strip()

        if attempt >= max_iters:
            fail_out = out_path.rsplit(".", 1)[0] + ".failing.py"
            with open(fail_out, "w", encoding="utf-8") as f:
                f.write(strip_think_tags(solution_code))
            print(f"[agent] Reached max iterations. Last failing solution at {fail_out}", file=sys.stderr)
            print(test_result.stdout, file=sys.stderr)
            print(test_result.stderr, file=sys.stderr)
            return 1

        solution_code = repair_with_feedback(
            client=client,
            repair_model=synth_model,
            failing_code=solution_code,
            task_prompt=user_prompt,
            test_output=feedback_blob,
            temperature=min(0.6, temperature + 0.2),
            seed=seed,
        )

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))