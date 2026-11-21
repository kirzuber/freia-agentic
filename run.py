"""Run Bob or Sol on one local mission and save a Markdown deliverable."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", choices=("bob", "sol"))
    parser.add_argument("mission", type=Path, help="Path to a private Markdown mission")
    args = parser.parse_args()

    if not os.getenv("DEEPSEEK_API_KEY"):
        parser.error("set DEEPSEEK_API_KEY in the environment")

    config = yaml.safe_load((ROOT / "config" / f"{args.agent}.yaml").read_text())
    instructions = (ROOT / config["instructions"]).read_text().strip()
    mission = args.mission.read_text().strip()
    if not instructions or not mission:
        parser.error("instructions and mission must both contain text")

    model = ChatDeepSeek(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    start = perf_counter()
    response = model.invoke([SystemMessage(content=instructions), HumanMessage(content=mission)])
    elapsed = perf_counter() - start

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"{args.agent}.md"
    output.write_text(str(response.content).strip() + "\n")
    usage = response.usage_metadata or {}
    print(f"Output: {output}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(
        "Tokens: "
        f"input={usage.get('input_tokens', 'unknown')} "
        f"output={usage.get('output_tokens', 'unknown')} "
        f"total={usage.get('total_tokens', 'unknown')}"
    )


if __name__ == "__main__":
    main()
