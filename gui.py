import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from main import (
    DEFAULT_MODELS,
    PRODUCTION_MODELS,
    PREVIEW_MODELS,
    list_available_models,
    make_groq_client,
)

# Default values matching command-line interface
DEFAULTS = {
    "models": DEFAULT_MODELS,
    "temperature": "0.2",
    "seed": "42",
    "max_iters": "4",
    "timeout": "25",
    "out": "solution.py",
    "fanout": "3",
    "max_feature_iters": "10",
    "enhance_repair_iters": "2",
}

def run_agent(cmd, output_widget, run_button):
    """Run main.py in a subprocess and stream output to the widget."""
    run_button.config(state=tk.DISABLED)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            output_widget.insert(tk.END, line)
            output_widget.see(tk.END)
        proc.wait()
        output_widget.insert(tk.END, f"\n[process exited with {proc.returncode}]\n")
    except Exception as exc:
        output_widget.insert(tk.END, f"\n[error running agent: {exc}]\n")
    finally:
        run_button.config(state=tk.NORMAL)

def start_run():
    task = task_text.get("1.0", tk.END).strip()
    if not task:
        messagebox.showerror("Error", "Task prompt is required")
        return
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "main.py"), "--task", task]
    tests = tests_text.get("1.0", tk.END).strip()
    if tests:
        cmd += ["--tests", tests]

    selected = [models_listbox.get(i) for i in models_listbox.curselection()]
    models_arg = ",".join(selected) if selected else DEFAULTS["models"]
    if models_arg:
        cmd += ["--models", models_arg]

    for key, var in entry_vars.items():
        val = var.get().strip()
        if val:
            cmd += [f"--{key.replace('_', '-')}", val]
    if print_intermediate_var.get():
        cmd.append("--print-intermediate")
    if dry_run_var.get():
        cmd.append("--dry-run")
    if prefer_tool_use_var.get():
        cmd.append("--prefer-tool-use")
    if disable_post_feature_var.get():
        cmd.append("--disable-post-feature")
    if disable_enhance_var.get():
        cmd.append("--disable-enhance")
    output_text.delete("1.0", tk.END)
    threading.Thread(target=run_agent, args=(cmd, output_text, run_button), daemon=True).start()

root = tk.Tk()
root.title("Angler Fish Agent")

# Task prompt
ttk.Label(root, text="Task Prompt:").grid(row=0, column=0, sticky="nw")
task_text = scrolledtext.ScrolledText(root, width=60, height=5)
task_text.grid(row=0, column=1, columnspan=3, pady=5, padx=5)

# Tests
ttk.Label(root, text="Tests:").grid(row=1, column=0, sticky="nw")
tests_text = scrolledtext.ScrolledText(root, width=60, height=5)
tests_text.grid(row=1, column=1, columnspan=3, pady=5, padx=5)

# Model selection
try:
    if os.getenv("GROQ_API_KEY"):
        _client = make_groq_client()
        AVAILABLE_MODELS = list_available_models(_client)
    else:
        raise RuntimeError("No API key")
except Exception:
    AVAILABLE_MODELS = sorted(set(PRODUCTION_MODELS + PREVIEW_MODELS))

ttk.Label(root, text="Models:").grid(row=2, column=0, sticky="nw")
models_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=6)
for m in AVAILABLE_MODELS:
    models_listbox.insert(tk.END, m)
models_listbox.grid(row=2, column=1, columnspan=3, pady=5, padx=5, sticky="w")

# Option entries
entry_vars = {
    "temperature": tk.StringVar(value=DEFAULTS["temperature"]),
    "seed": tk.StringVar(value=DEFAULTS["seed"]),
    "max_iters": tk.StringVar(value=DEFAULTS["max_iters"]),
    "timeout": tk.StringVar(value=DEFAULTS["timeout"]),
    "out": tk.StringVar(value=DEFAULTS["out"]),
    "fanout": tk.StringVar(value=DEFAULTS["fanout"]),
    "max_feature_iters": tk.StringVar(value=DEFAULTS["max_feature_iters"]),
    "enhance_repair_iters": tk.StringVar(value=DEFAULTS["enhance_repair_iters"]),
}

row = 3
for i, (label, var) in enumerate([
    ("Temperature", entry_vars["temperature"]),
    ("Seed", entry_vars["seed"]),
]):
    ttk.Label(root, text=label + ":").grid(row=row, column=i, sticky="w", padx=5)
    ttk.Entry(root, textvariable=var, width=20).grid(row=row+1, column=i, padx=5)
row += 2
for i, (label, var) in enumerate([
    ("Max Iters", entry_vars["max_iters"]),
    ("Timeout", entry_vars["timeout"]),
    ("Out", entry_vars["out"]),
]):
    ttk.Label(root, text=label + ":").grid(row=row, column=i, sticky="w", padx=5)
    ttk.Entry(root, textvariable=var, width=20).grid(row=row+1, column=i, padx=5)
row += 2
for i, (label, var) in enumerate([
    ("Fanout", entry_vars["fanout"]),
]):
    ttk.Label(root, text=label + ":").grid(row=row, column=i, sticky="w", padx=5)
    ttk.Entry(root, textvariable=var, width=20).grid(row=row+1, column=i, padx=5)
row += 2
for i, (label, var) in enumerate([
    ("Max Feature Iters", entry_vars["max_feature_iters"]),
    ("Enhance Repair Iters", entry_vars["enhance_repair_iters"]),
]):
    ttk.Label(root, text=label + ":").grid(row=row, column=i, sticky="w", padx=5)
    ttk.Entry(root, textvariable=var, width=20).grid(row=row+1, column=i, padx=5)
row += 2

# Checkboxes
print_intermediate_var = tk.BooleanVar()
dry_run_var = tk.BooleanVar()
prefer_tool_use_var = tk.BooleanVar()
disable_post_feature_var = tk.BooleanVar()
disable_enhance_var = tk.BooleanVar()

checks = [
    ("Print Intermediate", print_intermediate_var),
    ("Dry Run", dry_run_var),
    ("Prefer Tool Use", prefer_tool_use_var),
    ("Disable Post Feature", disable_post_feature_var),
    ("Disable Enhance", disable_enhance_var),
]
for i, (text, var) in enumerate(checks):
    ttk.Checkbutton(root, text=text, variable=var).grid(row=row, column=i % 3, sticky="w", padx=5)
    if i % 3 == 2:
        row += 1
row += 1

run_button = ttk.Button(root, text="Run", command=start_run)
run_button.grid(row=row, column=0, pady=5, padx=5, sticky="w")

# Output area
ttk.Label(root, text="Output:").grid(row=row+1, column=0, sticky="nw")
output_text = scrolledtext.ScrolledText(root, width=80, height=15)
output_text.grid(row=row+1, column=1, columnspan=3, pady=5, padx=5)

if __name__ == "__main__":
    root.mainloop()
