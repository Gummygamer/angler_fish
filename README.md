# Angler Fish

Angler Fish is a Python agent that uses Groq's language models to generate, test, and iteratively refine code. It offers a command-line interface and a simple Tkinter-based GUI.

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
uses it as the starting point instead of generating a fresh implementation.

## GUI

Launch the graphical interface with:

```bash
python gui.py
```

The GUI displays available models for selection and no longer includes the timeout explanation field.

## Development

See [AGENTS.md](AGENTS.md) for contribution guidelines.

