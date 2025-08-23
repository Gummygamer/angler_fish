# Angler Fish

Angler Fish is a Python agent that uses Groq's language models to generate, test, and iteratively refine code. It offers a command-line interface and a simple Tkinter-based GUI.

The agent strips any `<think>` reasoning blocks from generated code and tests before executing them.

## Installation

The project targets Python 3. Install required dependencies:

```bash
pip install groq requests
```

## Command-line Usage

Provide a task description and optional tests:

```bash
python main.py --task "write fibonacci" --tests "assert solution.fib(5) == 5"
```

Run `python main.py --help` to see all available options.

If a file named `solution.py` already exists in the working directory, the agent
uses it as the starting point instead of generating a fresh implementation. The
current contents of `solution.py` are included in prompts and the models are
explicitly instructed to update or extend this code, rather than replacing it
with an unrelated program.

### GUI / Game Tasks

When your task describes a GUI application or a game, Angler Fish skips unit
test generation and simply runs the produced module to ensure it executes
without raising an error. The agent determines whether a task is GUI or game
related by asking each selected model and treating it as such only if a strict
majority votes "yes".

## GUI

Launch the graphical interface with:

```bash
python gui.py
```

The GUI displays available models for selection and no longer includes the timeout explanation field.

## Development

See [AGENTS.md](AGENTS.md) for contribution guidelines.

