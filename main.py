from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from typing import Optional
import logging
import os

import httpx
from dotenv import load_dotenv

from services.rag_service import RAGService

# Load environment once (in containers, you typically rely on env vars)
load_dotenv()

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="RAG Training Chatbot API",
    description="A simple RAG-based chatbot for employee training with PostgreSQL memory",
    version="1.0.0"
)

# Add CORS middleware (consider tightening to known front-end origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://your-frontend.example.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We'll initialize shared clients/services on startup
@app.on_event("startup")
async def startup_event():
    # Shared HTTPX AsyncClient with pooling, timeouts, and basic retries
    app.state.httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=3.0, read=10.0, write=10.0, pool=None),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
        transport=httpx.AsyncHTTPTransport(retries=3),
    )
    app.state.rag_service = RAGService(client=app.state.httpx_client, logger=logger)
    logger.info("Startup complete: HTTP client and RAGService initialized")

@app.on_event("shutdown")
async def shutdown_event():
    # Close downstream clients gracefully
    try:
        if hasattr(app.state, "rag_service") and app.state.rag_service:
            await app.state.rag_service.aclose()
    finally:
        if hasattr(app.state, "httpx_client") and app.state.httpx_client:
            await app.state.httpx_client.aclose()
    logger.info("Shutdown complete: clients closed")


class ChatRequest(BaseModel):
    message: constr(strip_whitespace=True, min_length=1, max_length=5000)
    session_id: Optional[str] = None
    max_tokens: int = Field(500, ge=32, le=1000)  # server-side caps

class ChatResponse(BaseModel):
    response: str
    sources: list = []
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

class HistoryResponse(BaseModel):
    messages: list
    session_id: str


@app.get("/")
async def root():
    return {"message": "RAG Training Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        response = await rag_service.get_rag_response(
            user_message=request.message,
            session_id=request.session_id,
            max_tokens=request.max_tokens
        )
        return ChatResponse(
            response=response["answer"],
            sources=response.get("sources", []),
            session_id=response["session_id"]
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("chat endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/session/new", response_model=SessionResponse)
async def create_session():
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        session_id = await rag_service.create_new_session()
        return SessionResponse(
            session_id=session_id,
            message="New chat session created"
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("create_session endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_conversation_history(session_id: str, limit: int = 10):
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        messages = await rag_service.get_conversation_history(session_id, limit)
        return HistoryResponse(messages=messages, session_id=session_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_conversation_history endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        success = await rag_service.clear_session(session_id)
        if success:
            return {"message": f"Session {session_id} cleared successfully"}
        raise HTTPException(status_code=500, detail="Internal server error")
    except HTTPException:
        raise
    except Exception:
        logger.exception("clear_session endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """This does nothing lol"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
