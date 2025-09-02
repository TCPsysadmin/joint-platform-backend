# main.py
import os
import logging
import uuid
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from starlette.requests import Request

from services.rag_service import RAGService

# Load environment once (entrypoint). Keep this here for local development.
load_dotenv()

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="RAG Training Chatbot API",
    description="A simple RAG-based chatbot for employee training with PostgreSQL memory",
    version="1.0.0",
)

# Respect an env var for allowed origins; fallback to wildcard if not provided.
cors_env = os.getenv("CORS_ALLOWED_ORIGINS")
if cors_env:
    allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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
    sources: list = Field(default_factory=list)
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    message: str


class HistoryResponse(BaseModel):
    messages: list = Field(default_factory=list)
    session_id: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: dict = Field(default_factory=dict)
    overall_healthy: bool


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
            max_tokens=request.max_tokens,
        )
        return ChatResponse(
            response=response["answer"],
            sources=response.get("sources", []),
            session_id=response["session_id"],
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
        return SessionResponse(session_id=session_id, message="New chat session created")
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check that tests OpenAI and Supabase connectivity."""
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                services={"error": "RAG service not initialized"},
                overall_healthy=False
            )

        # Run all health checks concurrently
        import asyncio
        openai_task = asyncio.create_task(rag_service.health_check_openai())
        supabase_task = asyncio.create_task(rag_service.health_check_supabase())
        vector_task = asyncio.create_task(rag_service.health_check_vector_search())
        
        openai_health, supabase_health, vector_health = await asyncio.gather(
            openai_task, supabase_task, vector_task, return_exceptions=True
        )

        # Handle any exceptions from the health checks
        services = {}
        
        if isinstance(openai_health, Exception):
            services["openai"] = {"status": "unhealthy", "error": str(openai_health)}
        else:
            services["openai"] = openai_health
            
        if isinstance(supabase_health, Exception):
            services["supabase"] = {"status": "unhealthy", "error": str(supabase_health)}
        else:
            services["supabase"] = supabase_health
            
        if isinstance(vector_health, Exception):
            services["vector_search"] = {"status": "unhealthy", "error": str(vector_health)}
        else:
            services["vector_search"] = vector_health

        # Determine overall health
        all_healthy = all(
            service.get("status") == "healthy" 
            for service in services.values()
        )

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            services=services,
            overall_healthy=all_healthy
        )

    except Exception as e:
        logger.exception("Health check failed")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            services={"error": f"Health check failed: {str(e)}"},
            overall_healthy=False
        )


@app.get("/health/simple")
async def simple_health_check():
    """Simple health check for load balancers - just checks if the service is running."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
