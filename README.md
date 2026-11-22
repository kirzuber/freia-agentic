# FREIA-001

Minimal local CLI for Bob and Sol, each using LangChain and DeepSeek Flash. Each agent can read public webpages with `read_webpage(url)`.

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

Each command writes `outputs/bob.md` or `outputs/sol.md` and prints elapsed time, token usage, and web requests. Sessions allow at most four model calls and three webpage requests, with a two-minute deadline. `.env`, `private/`, `private_missions/`, and `outputs/` are ignored by Git.

## First conversation

With `DEEPSEEK_API_KEY` loaded, run:

```sh
.venv/bin/python collaborate.py private/missions/true-north-dpo.md outputs/bob.md outputs/sol.md
```

This uses four alternating Bob/Sol calls, one Bob synthesis call, and three artifact calls, with no web tools. It prints each message and writes `outputs/conversation.md`, `outputs/joint-assessment.md`, `outputs/solution-direction.md`, `outputs/solution-diagram.svg`, and `outputs/deck.md`.

## Visual room

```sh
set -a; . ./.env; set +a
.venv/bin/uvicorn room:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ to enter a request, optional context and up to three text/Markdown files (32 KB each). **Prepare Mission** makes one DeepSeek call using `instructions/freia.md` and `config/freia.yaml`. Review or edit the mission, then **Approve Mission & Enter Room** passes it to the room for this visit only.
**Start session** runs fresh independent Bob and Sol research followed by their existing collaboration and artifact workflow. Outputs are isolated per session and downloaded from the current room run. Set `TAVILY_API_KEY` in `.env` to enable their bounded `search_web` tool alongside `read_webpage`.
During collaboration, Bob and Sol can use the same two research tools to check a material uncertainty or a colleague's question. The conversation shares a limit of two additional searches, two webpage requests and two model follow-ups before the reserved synthesis call.

After the joint assessment, Sol writes `outputs/solution-direction.md` and `outputs/solution-diagram.svg`; Bob then uses Sol's direction to write the six-slide `outputs/deck.md`. The room previews and downloads all four final artifacts.

The same run also renders `outputs/deck.html`, a standalone HTML slide presentation with Sol's diagram embedded. The room can open or download it. Artifact activity labels come from the live execution phases.
