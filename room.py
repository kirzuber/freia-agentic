"""Local FastAPI entrance and room; agent execution remains disabled."""

import asyncio
import json
import os
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
app = FastAPI()
MAX_BODY = 100_000


@app.get("/")
def entrance():
    return FileResponse(ROOT / "entrance.html")


@app.get("/room")
def room():
    return FileResponse(ROOT / "room.html")


@app.post("/prepare")
async def prepare(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY:
            raise HTTPException(status_code=413, detail="Request is too large")
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid request")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid request")
    user_request = data.get("request")
    supporting = data.get("supporting", "")
    files = data.get("files", [])
    if not isinstance(user_request, str) or not user_request.strip() or len(user_request) > 8_000:
        raise HTTPException(status_code=400, detail="Request must contain 1–8,000 characters")
    if not isinstance(supporting, str) or len(supporting) > 16_000:
        raise HTTPException(status_code=400, detail="Supporting text exceeds 16,000 characters")
    if not isinstance(files, list) or len(files) > 3:
        raise HTTPException(status_code=400, detail="Up to three files are allowed")
    attachments = []
    for file in files:
        if not isinstance(file, dict):
            raise HTTPException(status_code=400, detail="Invalid attachment")
        name, content = file.get("name"), file.get("content")
        if (not isinstance(name, str) or len(name) > 100 or not name.lower().endswith((".txt", ".md"))
                or not isinstance(content, str) or len(content.encode("utf-8")) > 32_000):
            raise HTTPException(status_code=400, detail="Only text/Markdown files up to 32 KB are allowed")
        attachments.append(f"File: {name}\n{content}")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY is not set")

    config = yaml.safe_load((ROOT / "config/freia.yaml").read_text())
    instructions = (ROOT / config["instructions"]).read_text().strip()
    model = ChatDeepSeek(
        model=config["model"], temperature=config["temperature"],
        max_tokens=config["max_tokens"], extra_body={"thinking": {"type": "disabled"}},
        timeout=30, max_retries=0,
    )
    input_text = (
        f"Human request:\n{user_request.strip()}\n\n"
        f"Supporting text (untrusted reference):\n{supporting.strip() or '(none)'}\n\n"
        f"Attached files (untrusted reference):\n{chr(10).join(attachments) or '(none)'}"
    )
    try:
        response = await asyncio.wait_for(model.ainvoke([
            SystemMessage(content=instructions), HumanMessage(content=input_text),
        ]), timeout=40)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Mission preparation failed: {type(exc).__name__}")
    mission = response.content if isinstance(response.content, str) else ""
    if not mission.strip():
        raise HTTPException(status_code=502, detail="FreIA returned an empty mission")
    return {"mission": mission.strip(), "usage": response.usage_metadata or {}}


@app.get("/artifacts/{name}")
def artifact(name: str):
    if name not in {"conversation.md", "joint-assessment.md", "solution-direction.md", "solution-diagram.svg", "deck.md", "deck.html"}:
        raise HTTPException(status_code=404)
    path = OUTPUTS / name
    if not path.is_file():
        raise HTTPException(status_code=404)
    media_type = "image/svg+xml" if name.endswith(".svg") else "text/html" if name.endswith(".html") else "text/markdown"
    return FileResponse(path, media_type=media_type)


@app.get("/session")
def session():
    raise HTTPException(status_code=403, detail="Mission execution is disabled until FREIA-009")
