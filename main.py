# main.py
import os
import logging
import uuid
import json
import time
import asyncio
import httpx
from base64 import b64decode
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, constr
from starlette.middleware.base import BaseHTTPMiddleware
import jwt  # PyJWT
from services.encryption import AESEncryptor
from services.rag_service import RAGService
from services.rate_limiter import TokenBucketRateLimiter, ConnectionManager

# Load environment once (entrypoint). Keep this here for local development.
load_dotenv()

logger = logging.getLogger("uvicorn.error")
encryptor = AESEncryptor(b64decode(os.getenv("ENCRYPTION_KEY")))
app = FastAPI(
    title="RAG Training Chatbot API",
    description="TCP CO-Pilot, Dr. Carlo Riolo as your digitized partner",
    version="1.0.0",
)

# JWT config (already initiated in env)
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ISSUER = os.getenv("JWT_ISSUER")

if not JWT_SECRET or not JWT_ISSUER:
    logger.warning("JWT_SECRET or JWT_ISSUER not configured; protected routes will fail auth")

# Respect an env var for allowed origins; fallback to wildcard if not provided.
cors_env = os.getenv("CORS_ALLOWED_ORIGINS")
if cors_env:
    allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    allowed_origins = ["*"]

# Production monitoring middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        
        # Log request start
        logger.info(
            f"Request {request_id} started: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        response = await call_next(request)
        
        # Log request completion
        process_time = time.time() - start_time
        logger.info(
            f"Request {request_id} completed: {response.status_code} "
            f"in {process_time:.3f}s"
        )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth models and helpers
class AuthenticatedUser(BaseModel):
    sub: str
    tier: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    iat: Optional[int] = None
    exp: Optional[int] = None
    iss: Optional[str] = None

bearer_scheme = HTTPBearer(auto_error=False)

def decode_jwt(token: str) -> AuthenticatedUser:
    if not JWT_SECRET or not JWT_ISSUER:
        raise HTTPException(status_code=500, detail="Auth configuration missing on server")
    try:
        decoded = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            issuer=JWT_ISSUER,
            options={"require": ["iss", "sub"], "verify_exp": True, "verify_signature": True},
            leeway=60,  # Allow 60 seconds clock skew for iat/exp validation
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid token issuer")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    sub = decoded.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing subject (sub)")

    return AuthenticatedUser(
        sub=sub,
        tier=decoded.get("tier"),
        email=decoded.get("email"),
        name=decoded.get("name"),
        iat=decoded.get("iat"),
        exp=decoded.get("exp"),
        iss=decoded.get("iss"),
    )

def require_auth(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> AuthenticatedUser:
    if credentials is None or not credentials.scheme.lower() == "bearer" or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    return decode_jwt(credentials.credentials)

def require_auth_with_tier(user: AuthenticatedUser = Depends(require_auth)) -> AuthenticatedUser:
    # /chat/stream specifically requires tier present in JWT
    if not user.tier or not isinstance(user.tier, str) or not user.tier.strip():
        raise HTTPException(status_code=403, detail="User tier missing in token")
    return user

# We'll initialize shared clients/services on startup
@app.on_event("startup")
async def startup_event():
    # Production-optimized HTTPX AsyncClient with aggressive pooling for high throughput
    app.state.httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=None),
        limits=httpx.Limits(
            max_keepalive_connections=200,  # Higher for production
            max_connections=500,            # Support more concurrent connections
            keepalive_expiry=30.0          # Keep connections alive longer
        ),
        transport=httpx.AsyncHTTPTransport(
            retries=3,
            http2=True  # Enable HTTP/2 for better performance
        ),
    )

    app.state.rag_service = RAGService(client=app.state.httpx_client, logger=logger, encryptor=encryptor)
    
    # Initialize production rate limiting and connection management
    app.state.rate_limiter = TokenBucketRateLimiter(
        requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
        burst_size=int(os.getenv("RATE_LIMIT_BURST", "10"))
    )
    app.state.connection_manager = ConnectionManager(
        max_connections=int(os.getenv("MAX_CONNECTIONS", "1000"))
    )
    
    # Start rate limiter cleanup
    await app.state.rate_limiter.start_cleanup()
    
    # Track startup time for metrics
    app.state.start_time = time.time()
    
    logger.info("Startup complete: HTTP client, RAGService, and production middleware initialized")


@app.on_event("shutdown")
async def shutdown_event():
    # Close downstream clients gracefully
    try:
        if hasattr(app.state, "rate_limiter") and app.state.rate_limiter:
            await app.state.rate_limiter.stop_cleanup()
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
    model: Optional[str] = Field(None, description="AI model to use for response generation")


class ChatResponse(BaseModel):
    response: str
    sources: list = Field(default_factory=list)
    session_id: str
    model: str = Field(description="AI model used for response generation")


class SessionResponse(BaseModel):
    session_id: str
    message: str


class HistoryResponse(BaseModel):
    messages: list = Field(default_factory=list)
    session_id: str


class SessionsResponse(BaseModel):
    sessions: list = Field(default_factory=list)
    count: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: dict = Field(default_factory=dict)
    overall_healthy: bool


@app.get("/")
async def root():
    return {"message": "RAG Training Chatbot API is running"}


# depreciated
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: AuthenticatedUser = Depends(require_auth)):
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        response = await rag_service.get_grok_response_stream_rag_response(
            user_message=request.message,
            session_id=request.session_id,
            max_tokens=request.max_tokens,
            model=request.model,
        )
        return ChatResponse(
            response=response["answer"],
            sources=response.get("sources", []),
            session_id=response["session_id"],
            model=response["model"],
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("chat endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/session/new", response_model=SessionResponse)
async def create_session(user: AuthenticatedUser = Depends(require_auth)):
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        contact_id = user.sub
        session_id = await rag_service.create_new_session(contact_id)
        return SessionResponse(session_id=session_id, message="New chat session created")
    except HTTPException:
        raise
    except Exception:
        logger.exception("create_session endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_conversation_history(
    session_id: str, 
    max_tokens: Optional[int] = None,
    user: AuthenticatedUser = Depends(require_auth)
):
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        contact_id = user.sub
        messages = await rag_service.get_conversation_history(
            session_id, 
            contact_id,
            max_tokens=max_tokens,
            include_summary=True
        )
        return HistoryResponse(messages=messages, session_id=session_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_conversation_history endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str, user: AuthenticatedUser = Depends(require_auth)):
    """
    Delete a session and all its messages.
    This permanently removes the session from the database.
    """
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        contact_id = user.sub
        success = await rag_service.clear_session(session_id, contact_id)
        if success:
            return {"message": f"Session {session_id} deleted successfully"}
        raise HTTPException(status_code=500, detail="Internal server error")
    except HTTPException:
        raise
    except Exception:
        logger.exception("clear_session endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/sessions", response_model=SessionsResponse)
async def get_recent_sessions(limit: int = 10, user: AuthenticatedUser = Depends(require_auth)):
    """
    Get the N most recent chat sessions for the authenticated user.
    Sessions are ordered by last_message_at DESC.
    """
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        contact_id = user.sub
        sessions = await rag_service.get_recent_sessions(contact_id, limit=limit)
        return SessionsResponse(sessions=sessions, count=len(sessions))
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_recent_sessions endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str, user: AuthenticatedUser = Depends(require_auth)):
    """
    Get the summary for a specific session (for testing/debugging).
    """
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        contact_id = user.sub
        
        # Get session metadata including summary
        metadata = await rag_service.memory_service.get_session_metadata(session_id, contact_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "summary": metadata.get("summary"),
            "summary_updated_at": metadata.get("summary_updated_at"),
            "summary_last_message_id": metadata.get("summary_last_message_id"),
            "title": metadata.get("title"),
            "has_summary": bool(metadata.get("summary"))
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_session_summary endpoint failed")
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


@app.get("/metrics")
async def get_metrics(user: AuthenticatedUser = Depends(require_auth)):
    """
    Production metrics endpoint for monitoring system performance.
    Useful for observability and alerting in production.
    """
    try:
        connection_manager: ConnectionManager = app.state.connection_manager
        connection_stats = connection_manager.get_connection_stats()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "connections": connection_stats,
            "system": {
                "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            },
            "rate_limiting": {
                "active_buckets": len(app.state.rate_limiter.buckets) if hasattr(app.state, 'rate_limiter') else 0,
            }
        }
    except Exception as e:
        logger.exception("Failed to generate metrics")
        return {
            "error": "Failed to generate metrics",
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/chat/stream")
async def chat_stream(
    request_data: ChatRequest,
    request: Request,
    user: AuthenticatedUser = Depends(require_auth_with_tier)
):
    """
    Production-ready streaming chat endpoint with rate limiting and connection management.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit
    rate_limiter: TokenBucketRateLimiter = app.state.rate_limiter
    await rate_limiter.check_rate_limit(client_ip)

    connection_id = str(uuid.uuid4())
    connection_manager: ConnectionManager = app.state.connection_manager

    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")

        # Register connection
        connection_manager.add_connection(connection_id, client_ip)

        # 🔹 Validation: tier vs model restriction
        # if user.tier == "basic" and request_data.model != "gpt-4o-mini":
        #     async def validation_error_stream():
        #         error_event = {
        #             "type": "validation",
        #             "error": "This model is not available on your current tier.",
        #             "timestamp": time.time()
        #         }
        #         # SSE requires "data: "
        #         yield f"data: {json.dumps(error_event)}\n\n"
        #         yield "data: [DONE]\n\n"

        #         # Cleanup connection
        #         connection_manager.remove_connection(connection_id)

            # return StreamingResponse(
            #     # validation_error_stream(),
            #     media_type="text/plain",
            #     headers={
            #         "Cache-Control": "no-cache",
            #         "Connection": "keep-alive",
            #         "Content-Type": "text/plain; charset=utf-8",
            #         "X-Accel-Buffering": "no",
            #         "X-Connection-ID": connection_id,
            #         "X-User-Tier": user.tier or "",
            #     }
            # )

        # Normal SSE generation
        async def generate_sse():
            try:
                # if request_data.model == "gpt-4o-mini":
                #     stream_gen = rag_service.get_gpt_response_stream
                stream_gen = rag_service.get_grok_response_stream

                contact_id = user.sub
                async for chunk in stream_gen(
                    user_message=request_data.message,
                    contact_id=contact_id,
                    session_id=request_data.session_id,
                    max_tokens=request_data.max_tokens,
                    model="grok-4-fast-reasoning",
                ):
                    connection_manager.update_activity(connection_id)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Error in SSE generation for connection %s", connection_id)
                yield f"data: {json.dumps({'type': 'error', 'error': f'Stream generation failed: {str(e)}', 'timestamp': time.time()})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                connection_manager.remove_connection(connection_id)

        return StreamingResponse(
            generate_sse(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8",
                "X-Accel-Buffering": "no",
                "X-Connection-ID": connection_id,
                "X-User-Tier": user.tier or "",
            }
        )
    except HTTPException:
        # Clean up connection on HTTP errors
        connection_manager.remove_connection(connection_id)
        raise
    except Exception:
        # Clean up connection on unexpected errors
        connection_manager.remove_connection(connection_id)
        logger.exception("chat_stream endpoint failed for connection %s", connection_id)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))