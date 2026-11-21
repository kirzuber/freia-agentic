# FREIA-001

Minimal local CLI for Bob and Sol, each using LangChain and DeepSeek Flash.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
set -a; . ./.env; set +a
```

The Git-ignored `.env` file must contain `DEEPSEEK_API_KEY=...`. Bob's and Sol's supplied professional instructions are in `instructions/bob.md` and `instructions/sol.md`. Place each private mission in `private_missions/`.

## Run

```sh
.venv/bin/python run.py bob private_missions/bob-demo.md
.venv/bin/python run.py sol private_missions/sol-demo.md
```

Each command makes one model call, writes `outputs/bob.md` or `outputs/sol.md`, and prints elapsed time and token usage. `.env`, `private_missions/`, and `outputs/` are ignored by Git.
