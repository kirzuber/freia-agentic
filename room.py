"""Local FastAPI room for one Bob/Sol collaboration session."""

import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
app = FastAPI()
session_lock = asyncio.Lock()


@app.get("/")
def index():
    return FileResponse(ROOT / "room.html")


@app.get("/artifacts/{name}")
def artifact(name: str):
    if name not in {"conversation.md", "joint-assessment.md", "solution-direction.md", "solution-diagram.svg", "deck.md"}:
        raise HTTPException(status_code=404)
    path = OUTPUTS / name
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/svg+xml" if name.endswith(".svg") else "text/markdown")


@app.get("/session")
async def session():
    if session_lock.locked():
        raise HTTPException(status_code=409, detail="A session is already running")

    async def events():
        async with session_lock:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", str(ROOT / "collaborate.py"),
                str(ROOT / "private/missions/true-north-dpo.md"),
                str(OUTPUTS / "bob.md"), str(OUTPUTS / "sol.md"), "--events",
                cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                async for line in process.stdout:
                    if line.startswith(b"FREIA_EVENT "):
                        payload = line[len(b"FREIA_EVENT "):].decode().strip()
                        yield f"data: {payload}\n\n"
                await process.wait()
                if process.returncode:
                    error = (await process.stderr.read()).decode(errors="replace").strip().splitlines()
                    yield "data: " + json.dumps({"kind": "failed", "message": next((line for line in reversed(error) if line.startswith(("RuntimeError:", "TimeoutError:"))), "Session failed")}) + "\n\n"
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
