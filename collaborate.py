"""Run one bounded Bob/Sol conversation over existing local assessments."""

import argparse
import json
import os
import signal
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).resolve().parent
DEADLINE_SECONDS = 120


class SessionState(TypedDict):
    conversation: list[dict[str, str]]
    model_calls: int
    tokens: dict[str, int]
    joint_assessment: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mission", type=Path)
    parser.add_argument("bob_assessment", type=Path)
    parser.add_argument("sol_assessment", type=Path)
    parser.add_argument("--events", action="store_true", help="Emit JSON events for the local room")
    args = parser.parse_args()
    if not os.getenv("DEEPSEEK_API_KEY"):
        parser.error("set DEEPSEEK_API_KEY in the environment")

    mission = args.mission.read_text().strip()
    initial = {
        "bob": args.bob_assessment.read_text().strip(),
        "sol": args.sol_assessment.read_text().strip(),
    }
    if not mission or not all(initial.values()):
        parser.error("mission and both initial assessments must contain text")
    configs = {name: yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text()) for name in ("bob", "sol")}
    instructions = {name: (ROOT / configs[name]["instructions"]).read_text().strip() for name in configs}
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    conversation_path = output_dir / "conversation.md"
    joint_path = output_dir / "joint-assessment.md"
    conversation_path.write_text("# Bob and Sol — True North DPO conversation\n\n")
    joint_path.unlink(missing_ok=True)

    def emit(kind: str, **data) -> None:
        if args.events:
            print("FREIA_EVENT " + json.dumps({"kind": kind, **data}), flush=True)

    start = perf_counter()
    deadline = start + DEADLINE_SECONDS
    source_context = (
        f"Original mission:\n{mission}\n\n"
        f"Bob's independent assessment:\n{initial['bob']}\n\n"
        f"Sol's independent assessment:\n{initial['sol']}"
    )

    def check_time() -> float:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError("two-minute collaboration deadline reached")
        return min(20.0, remaining)

    def model_for(name: str, max_tokens: int) -> ChatDeepSeek:
        config = configs[name]
        return ChatDeepSeek(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
            timeout=check_time(),
            max_retries=0,
        )

    def transcript(state: SessionState) -> str:
        return "\n\n".join(f"{item['speaker'].title()}:\n{item['text']}" for item in state["conversation"]) or "(No messages yet.)"

    def count_tokens(state: SessionState, response) -> dict[str, int]:
        usage = response.usage_metadata or {}
        return {key: state["tokens"][key] + usage.get(key, 0) for key in state["tokens"]}

    def take_turn(state: SessionState, name: str) -> dict:
        if state["model_calls"] >= 4:
            raise RuntimeError("conversation call limit reached")
        emit("active", agent=name, turn=len(state["conversation"]) + 1)
        response = model_for(name, configs[name]["max_tokens"]).invoke([
            SystemMessage(content=instructions[name] + "\n\nThis is a local conversation with the other named agent. Use only the supplied assessments and conversation; no external research tools are available. Keep your own professional judgment and identity."),
            HumanMessage(content=source_context + "\n\nConversation so far:\n" + transcript(state) + f"\n\nRespond as {name.title()} to your colleague. Ask or answer focused questions, challenge material assumptions, and state whether your own initial view changes and why. Keep this turn concise."),
        ])
        content = str(response.content).strip()
        finish = response.response_metadata.get("finish_reason")
        if not content or finish != "stop":
            raise RuntimeError(f"{name} turn failed: finish_reason={finish}, content_chars={len(content)}")
        turn = len(state["conversation"]) + 1
        with conversation_path.open("a") as file:
            file.write(f"## Turn {turn} — {name.title()}\n\n{content}\n\n")
        emit("message", agent=name, turn=turn, text=content)
        print(f"\n{name.upper()} (turn {turn}):\n{content}\n", flush=True)
        return {
            "conversation": state["conversation"] + [{"speaker": name, "text": content}],
            "model_calls": state["model_calls"] + 1,
            "tokens": count_tokens(state, response),
        }

    def synthesize(state: SessionState) -> dict:
        if state["model_calls"] != 4:
            raise RuntimeError("synthesis requires four conversation turns")
        emit("active", agent="bob", phase="synthesis")
        response = model_for("bob", 2048).invoke([
            SystemMessage(content=instructions["bob"] + "\n\nSynthesize the actual Bob/Sol conversation faithfully. Do not invent agreement or new research."),
            HumanMessage(content=source_context + "\n\nConversation:\n" + transcript(state) + "\n\nWrite a Markdown joint assessment of at most 450 words. Include a final recommendation, material changes from each initial assessment and why, unresolved disagreements, and evidence links already present in the initial assessments. Do not describe a disagreement as resolved unless both agents explicitly accepted the resolution. Do not call tools."),
        ])
        content = str(response.content).strip()
        finish = response.response_metadata.get("finish_reason")
        if not content or finish != "stop":
            raise RuntimeError(f"synthesis failed: finish_reason={finish}, content_chars={len(content)}")
        joint_path.write_text(content + "\n")
        return {"joint_assessment": content, "model_calls": state["model_calls"] + 1, "tokens": count_tokens(state, response)}

    graph = StateGraph(SessionState)
    graph.add_node("bob_first", lambda state: take_turn(state, "bob"))
    graph.add_node("sol_first", lambda state: take_turn(state, "sol"))
    graph.add_node("bob_second", lambda state: take_turn(state, "bob"))
    graph.add_node("sol_second", lambda state: take_turn(state, "sol"))
    graph.add_node("synthesis", synthesize)
    graph.add_edge(START, "bob_first")
    graph.add_edge("bob_first", "sol_first")
    graph.add_edge("sol_first", "bob_second")
    graph.add_edge("bob_second", "sol_second")
    graph.add_edge("sol_second", "synthesis")
    graph.add_edge("synthesis", END)

    def on_deadline(_signum, _frame):
        raise TimeoutError("two-minute collaboration deadline reached")

    signal.signal(signal.SIGALRM, on_deadline)
    signal.setitimer(signal.ITIMER_REAL, DEADLINE_SECONDS)
    try:
        result = graph.compile().invoke({
            "conversation": [],
            "model_calls": 0,
            "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "joint_assessment": "",
        }, config={"recursion_limit": 10})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    print("Completion: complete")
    print(f"Elapsed: {perf_counter() - start:.2f}s")
    print(f"Model calls: {result['model_calls']} (4 conversation, 1 synthesis)")
    print("Tokens: " + " ".join(f"{key}={value}" for key, value in result["tokens"].items()))
    print(f"Conversation: {conversation_path}")
    print(f"Joint assessment: {joint_path}")
    emit("complete", elapsed=round(perf_counter() - start, 2), model_calls=result["model_calls"], tokens=result["tokens"])


if __name__ == "__main__":
    main()
