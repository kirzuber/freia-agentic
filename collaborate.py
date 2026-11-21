"""Run one bounded Bob/Sol conversation over existing local assessments."""

import argparse
import json
import os
import signal
import xml.etree.ElementTree as ET
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
    solution_direction: str
    diagram_svg: str
    deck: str


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
    direction_path = output_dir / "solution-direction.md"
    diagram_path = output_dir / "solution-diagram.svg"
    deck_path = output_dir / "deck.md"
    conversation_path.write_text("# Bob and Sol — True North DPO conversation\n\n")
    for path in (joint_path, direction_path, diagram_path, deck_path):
        path.unlink(missing_ok=True)

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
            stream_usage=True,
            timeout=check_time(),
            max_retries=0,
        )

    def transcript(state: SessionState) -> str:
        return "\n\n".join(f"{item['speaker'].title()}:\n{item['text']}" for item in state["conversation"]) or "(No messages yet.)"

    def count_tokens(state: SessionState, response) -> dict[str, int]:
        usage = response.usage_metadata or {}
        return {key: state["tokens"][key] + usage.get(key, 0) for key in state["tokens"]}

    def stream_response(model, messages, name: str, phase: str, turn: int | None = None):
        emit("state", agent=name, state="thinking", phase=phase, turn=turn)
        response = None
        speaking = False
        for chunk in model.stream(messages):
            response = chunk if response is None else response + chunk
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                if not speaking:
                    emit("state", agent=name, state="speaking", phase=phase, turn=turn)
                    speaking = True
                emit("delta", agent=name, phase=phase, turn=turn, text=text)
        if response is None:
            raise RuntimeError(f"{name} returned no model response")
        return response

    def take_turn(state: SessionState, name: str) -> dict:
        if state["model_calls"] >= 4:
            raise RuntimeError("conversation call limit reached")
        turn = len(state["conversation"]) + 1
        response = stream_response(model_for(name, configs[name]["max_tokens"]), [
            SystemMessage(content=instructions[name] + "\n\nThis is a local conversation with the other named agent. Use only the supplied assessments and conversation; no external research tools are available. Keep your own professional judgment and identity."),
            HumanMessage(content=source_context + "\n\nConversation so far:\n" + transcript(state) + f"\n\nRespond as {name.title()} to your colleague. Ask or answer focused questions, challenge material assumptions, and state whether your own initial view changes and why. Keep this turn concise."),
        ], name, "conversation", turn)
        content = str(response.content).strip()
        finish = response.response_metadata.get("finish_reason")
        if not content or finish != "stop":
            raise RuntimeError(f"{name} turn failed: finish_reason={finish}, content_chars={len(content)}")
        with conversation_path.open("a") as file:
            file.write(f"## Turn {turn} — {name.title()}\n\n{content}\n\n")
        emit("message_end", agent=name, turn=turn)
        emit("state", agent=name, state="waiting", phase="conversation", turn=turn)
        print(f"\n{name.upper()} (turn {turn}):\n{content}\n", flush=True)
        return {
            "conversation": state["conversation"] + [{"speaker": name, "text": content}],
            "model_calls": state["model_calls"] + 1,
            "tokens": count_tokens(state, response),
        }

    def synthesize(state: SessionState) -> dict:
        if state["model_calls"] != 4:
            raise RuntimeError("synthesis requires four conversation turns")
        response = stream_response(model_for("bob", 2048), [
            SystemMessage(content=instructions["bob"] + "\n\nSynthesize the actual Bob/Sol conversation faithfully. Do not invent agreement or new research."),
            HumanMessage(content=source_context + "\n\nConversation:\n" + transcript(state) + "\n\nWrite a Markdown joint assessment of at most 450 words. Include a final recommendation, material changes from each initial assessment and why, unresolved disagreements, and evidence links already present in the initial assessments. Do not describe a disagreement as resolved unless both agents explicitly accepted the resolution. Do not call tools."),
        ], "bob", "synthesis")
        content = str(response.content).strip()
        finish = response.response_metadata.get("finish_reason")
        if not content or finish != "stop":
            raise RuntimeError(f"synthesis failed: finish_reason={finish}, content_chars={len(content)}")
        joint_path.write_text(content + "\n")
        emit("artifact", name="joint-assessment.md")
        emit("state", agent="bob", state="waiting", phase="synthesis")
        return {"joint_assessment": content, "model_calls": state["model_calls"] + 1, "tokens": count_tokens(state, response)}

    def create_direction(state: SessionState) -> dict:
        if state["model_calls"] != 5:
            raise RuntimeError("solution direction requires the joint assessment")
        response = stream_response(model_for("sol", 1400), [
            SystemMessage(content=instructions["sol"] + "\n\nUse only the supplied assessments and conversation. No external tools or new research are available."),
            HumanMessage(content=source_context + "\n\nConversation:\n" + transcript(state) + "\n\nJoint assessment:\n" + state["joint_assessment"] + "\n\nWrite solution-direction.md in at most 300 words. Propose one small, conditional solution direction with its discovery gate, a no-build alternative, key dependencies and risks. Keep unverified client systems and economics explicitly unknown. Cite only the source URLs already read. Return Markdown only."),
        ], "sol", "solution")
        content = str(response.content).strip()
        finish = response.response_metadata.get("finish_reason")
        if not content or finish != "stop":
            raise RuntimeError(f"solution direction failed: finish_reason={finish}, content_chars={len(content)}")
        direction_path.write_text(content + "\n")
        emit("artifact", name="solution-direction.md")
        emit("state", agent="sol", state="waiting", phase="solution")
        return {"solution_direction": content, "model_calls": state["model_calls"] + 1, "tokens": count_tokens(state, response)}

    def create_diagram(state: SessionState) -> dict:
        if state["model_calls"] != 6:
            raise RuntimeError("diagram requires Sol's solution direction")
        response = stream_response(model_for("sol", 2000), [
            SystemMessage(content=instructions["sol"] + "\n\nCreate a simple, safe SVG diagram of your own solution direction. No external tools are available."),
            HumanMessage(content="Your solution direction:\n" + state["solution_direction"] + "\n\nReturn only a compact standalone SVG, under 2000 characters, beginning with <svg and ending with </svg>. Use a 720x240 viewBox, one dark background rect, exactly three short labeled boxes, and two connecting lines. Choose box labels from your direction and mark conditional steps as conditional. No title, notes, scripts, links, images, CSS, external resources, or invented client systems."),
        ], "sol", "diagram")
        svg = str(response.content).strip()
        finish = response.response_metadata.get("finish_reason")
        if svg.startswith("```"):
            svg = svg.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if not svg or finish != "stop":
            raise RuntimeError(f"diagram failed: finish_reason={finish}, content_chars={len(svg)}")
        root = ET.fromstring(svg)
        allowed = {"svg", "g", "rect", "line", "path", "circle", "ellipse", "polyline", "polygon", "text", "tspan"}
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] not in allowed:
                raise RuntimeError("diagram contains unsupported SVG elements")
            if any(attribute.rsplit("}", 1)[-1].lower().startswith("on") or attribute.rsplit("}", 1)[-1].lower() in {"href", "style"} for attribute in element.attrib):
                raise RuntimeError("diagram contains active or external SVG attributes")
        if root.tag.rsplit("}", 1)[-1] != "svg":
            raise RuntimeError("diagram response is not SVG")
        diagram_path.write_text(svg + "\n")
        emit("artifact", name="solution-diagram.svg")
        emit("state", agent="sol", state="waiting", phase="diagram")
        return {"diagram_svg": svg, "model_calls": state["model_calls"] + 1, "tokens": count_tokens(state, response)}

    def create_deck(state: SessionState) -> dict:
        if state["model_calls"] != 7:
            raise RuntimeError("deck requires Sol's solution direction and diagram")
        response = stream_response(model_for("bob", 2400), [
            SystemMessage(content=instructions["bob"] + "\n\nThe current user request authorizes a concise discussion deck. Use Sol's supplied solution direction as the technical basis; do not recreate it independently. No external tools or new research are available."),
            HumanMessage(content=source_context + "\n\nJoint assessment:\n" + state["joint_assessment"] + "\n\nSol's solution direction:\n" + state["solution_direction"] + "\n\nWrite deck.md as exactly six Markdown slides, at most 450 words total, each headed '## Slide N — Title' with two to four concise bullets. This is an executive discussion deck for a prospective client, not a sales claim. Include source/access caveats, opportunity hypothesis, Sol's conditional direction and no-build alternative, discovery questions, and a decision gate. Cite only source URLs already read. Do not claim client pain, systems, budget, or ROI as facts."),
        ], "bob", "deck")
        content = str(response.content).strip()
        finish = response.response_metadata.get("finish_reason")
        if not content or finish != "stop":
            raise RuntimeError(f"deck failed: finish_reason={finish}, content_chars={len(content)}")
        deck_path.write_text(content + "\n")
        emit("artifact", name="deck.md")
        emit("state", agent="bob", state="waiting", phase="deck")
        return {"deck": content, "model_calls": state["model_calls"] + 1, "tokens": count_tokens(state, response)}

    graph = StateGraph(SessionState)
    graph.add_node("bob_first", lambda state: take_turn(state, "bob"))
    graph.add_node("sol_first", lambda state: take_turn(state, "sol"))
    graph.add_node("bob_second", lambda state: take_turn(state, "bob"))
    graph.add_node("sol_second", lambda state: take_turn(state, "sol"))
    graph.add_node("synthesis", synthesize)
    graph.add_node("solution", create_direction)
    graph.add_node("diagram", create_diagram)
    graph.add_node("deck", create_deck)
    graph.add_edge(START, "bob_first")
    graph.add_edge("bob_first", "sol_first")
    graph.add_edge("sol_first", "bob_second")
    graph.add_edge("bob_second", "sol_second")
    graph.add_edge("sol_second", "synthesis")
    graph.add_edge("synthesis", "solution")
    graph.add_edge("solution", "diagram")
    graph.add_edge("diagram", "deck")
    graph.add_edge("deck", END)

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
            "solution_direction": "",
            "diagram_svg": "",
            "deck": "",
        }, config={"recursion_limit": 12})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    print("Completion: complete")
    print(f"Elapsed: {perf_counter() - start:.2f}s")
    print(f"Model calls: {result['model_calls']} (4 conversation, 1 synthesis, 3 artifact)")
    print("Tokens: " + " ".join(f"{key}={value}" for key, value in result["tokens"].items()))
    print(f"Conversation: {conversation_path}")
    print(f"Joint assessment: {joint_path}")
    print(f"Solution direction: {direction_path}")
    print(f"Solution diagram: {diagram_path}")
    print(f"Deck: {deck_path}")
    emit("complete", elapsed=round(perf_counter() - start, 2), model_calls=result["model_calls"], tokens=result["tokens"])


if __name__ == "__main__":
    main()
