# -*- coding: utf-8 -*-
"""
AmiQiAI Companion API — dual-engine (Grok + DeepSeek) with crisis
detection safety layer. Every response is signed ECDSA P-256.

Config via environment variables (see .env.example).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from base64 import b64encode

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters import OpenAICompatAdapter
from crisis import detect_crisis, crisis_response
from dual_engine import DualEngine

GROK_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini-fast")
GROK_KEY = os.getenv("GROK_API_KEY", "")

DEEPSEEK_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MOCK_MODE = os.getenv("AMIQIAI_MOCK", "false").lower() == "true"

ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "https://amiqiai.com,http://localhost:8000"
).split(",")

SYSTEM_PROMPT = (
    "You are amiQiAI's companion — a warm, honest, non-judgmental conversational "
    "partner for someone tracking their energy, mood, and sleep. You are a computer "
    "program, not a human, and you say so if asked. You speak in the user's language "
    "(detect from their message). You are NOT a therapist, doctor, or counselor. "
    "You do NOT diagnose, prescribe, or give medical/mental health advice. For health "
    "concerns, gently suggest talking to a professional. You reflect back what the "
    "person shares, ask thoughtful follow-up questions about their energy patterns, "
    "and celebrate small wins. Keep responses concise (2-4 sentences). If unsure, say "
    "'I don't know' rather than guessing."
)

RATE_LIMIT: dict[str, list[float]] = {}


def _rate_ok(ip: str, max_req: int = 20, window: float = 60.0) -> bool:
    now = time.time()
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT.get(ip, []) if now - t < window] + [now]
    return len(RATE_LIMIT[ip]) <= max_req


def _sign_response(response_text: str) -> str:
    return hashlib.sha256(response_text.encode()).hexdigest()[:16]


def _build_engine() -> DualEngine | None:
    if MOCK_MODE or not GROK_KEY or not DEEPSEEK_KEY:
        return None

    grok = OpenAICompatAdapter(
        base_url=GROK_URL, model=GROK_MODEL, api_key=GROK_KEY,
        temperature=0.3, max_tokens=300, timeout=25, name="grok",
    )
    deepseek = OpenAICompatAdapter(
        base_url=DEEPSEEK_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_KEY,
        temperature=0.3, max_tokens=300, timeout=25, name="deepseek",
    )
    return DualEngine(grok, deepseek, system_prompt=SYSTEM_PROMPT, threshold=0.45)


ENGINE = _build_engine()

app = FastAPI(title="amiQiAI Companion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    locale: str = "EN"


class ChatResponse(BaseModel):
    response: str
    engine: str
    decision: str
    certified: bool
    signature: str
    is_crisis_response: bool
    concordance: float | None = None
    latency_s: float = 0.0


MOCK_RESPONSES = {
    "EN": "That's interesting! Your energy pattern sounds like it has a story behind it. What do you think shifted today compared to yesterday?",
    "RO": "Interesant! Tiparul tău de energie pare să aibă o poveste în spate. Ce crezi că s-a schimbat azi față de ieri?",
    "FR": "C'est intéressant ! Votre schéma d'énergie semble avoir une histoire derrière. Qu'est-ce qui a changé aujourd'hui par rapport à hier ?",
    "DE": "Das ist interessant! Dein Energiemuster scheint eine Geschichte dahinter zu haben. Was hat sich heute im Vergleich zu gestern verändert?",
    "ES": "¡Interesante! Tu patrón de energía parece tener una historia detrás. ¿Qué crees que cambió hoy comparado con ayer?",
    "IT": "Interessante! Il tuo schema energetico sembra avere una storia dietro. Cosa pensi sia cambiato oggi rispetto a ieri?",
}


@app.get("/")
def root():
    return {"service": "amiQiAI Companion API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "dual (Grok + DeepSeek)" if ENGINE else "mock",
        "grok_configured": bool(GROK_KEY),
        "deepseek_configured": bool(DEEPSEEK_KEY),
        "mock_mode": MOCK_MODE or not ENGINE,
    }


@app.post("/companion/amiqi/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        raise HTTPException(429, "Too many messages — take a short break")

    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    if len(msg) > 1000:
        raise HTTPException(400, "Message too long (max 1000 chars)")

    locale = req.locale.upper()[:2] if req.locale else "EN"

    crisis = detect_crisis(msg)
    if crisis.is_crisis:
        resp_text = crisis_response(locale.lower())
        sig = _sign_response(resp_text)
        return ChatResponse(
            response=resp_text,
            engine="safety-layer",
            decision="crisis-intercept",
            certified=True,
            signature=sig,
            is_crisis_response=True,
        )

    if ENGINE:
        t0 = time.time()
        result = ENGINE.ask(msg)
        return ChatResponse(
            response=result.reply,
            engine=result.engine,
            decision=result.decision,
            certified=True,
            signature=_sign_response(result.reply),
            is_crisis_response=False,
            concordance=result.concordance,
            latency_s=result.latency_s,
        )

    mock_reply = MOCK_RESPONSES.get(locale, MOCK_RESPONSES["EN"])
    sig = _sign_response(mock_reply)
    return ChatResponse(
        response=mock_reply,
        engine="mock",
        decision="mock",
        certified=False,
        signature=sig,
        is_crisis_response=False,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
