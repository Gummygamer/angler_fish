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

## GUI

Launch the graphical interface with:

```bash
python gui.py
```

## Development

See [AGENTS.md](AGENTS.md) for contribution guidelines.

