"""
server.py — FastAPI wrapper implementing the Nova External API contract.

Endpoint: POST /external/v1/agents/{agent_id}/invoke
Contract : https://nova.sephora.com/docs/external-api-spec.yaml

Deploy this container and register the base URL in Nova (BYOA Registry)
so Nova can route chat turns to this agent.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent import run as agent_run

AGENT_ID = os.environ.get("NOVA_AGENT_ID", "a062f865-03a9-4a4c-95dd-564985079b00")

app = FastAPI(title="happy_test_suhail_agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class InvokeRequest(BaseModel):
    messages: list[Message]
    chatId: str | None = None
    context: dict[str, Any] | None = None


class InvokeResponse(BaseModel):
    sessionId: str
    agentId: str
    status: str
    response: dict[str, Any]
    usage: dict[str, int]
    timestamp: str


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "agentId": AGENT_ID}


@app.post(
    "/external/v1/agents/{agent_id}/invoke",
    response_model=InvokeResponse,
)
async def invoke(agent_id: str, payload: InvokeRequest, request: Request):
    if agent_id != AGENT_ID:
        raise HTTPException(status_code=404, detail="Agent not found")

    messages = [m.model_dump() for m in payload.messages]
    session_id = payload.chatId or str(uuid.uuid4())

    start = time.monotonic()
    output = agent_run(messages)
    latency_ms = int((time.monotonic() - start) * 1000)

    return InvokeResponse(
        sessionId=session_id,
        agentId=AGENT_ID,
        status="success",
        response={"content": output, "attachments": [], "toolCalls": []},
        usage={"promptTokens": 0, "completionTokens": 0, "totalTokens": 0},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
