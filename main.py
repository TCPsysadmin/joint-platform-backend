# main.py
import os
import logging
import uuid
import json
import time
import asyncio
import httpx
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, constr
from starlette.middleware.base import BaseHTTPMiddleware
from services.rag_service import RAGService
from services.rate_limiter import TokenBucketRateLimiter, ConnectionManager

# Load environment once (entrypoint). Keep this here for local development.
load_dotenv()

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="RAG Training Chatbot API",
    description="A simple RAG-based chatbot for employee training with PostgreSQL memory",
    version="1.0.0",
)

# Lifespan manager for more reliable startup/shutdown
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup with timeout protection
    logger.info("Starting up RAG service...")
    try:
        # Set startup timeout to prevent hanging
        startup_timeout = 30.0  # 30 seconds max for startup
        
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

        # Initialize RAG service with timeout
        try:
            app.state.rag_service = RAGService(client=app.state.httpx_client, logger=logger)
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
        
        # Initialize production rate limiting and connection management
        app.state.rate_limiter = TokenBucketRateLimiter(
            requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
            burst_size=int(os.getenv("RATE_LIMIT_BURST", "10"))
        )
        app.state.connection_manager = ConnectionManager(
            max_connections=int(os.getenv("MAX_CONNECTIONS", "1000"))
        )
        
        # Start rate limiter cleanup with timeout
        try:
            await asyncio.wait_for(
                app.state.rate_limiter.start_cleanup(),
                timeout=startup_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Rate limiter cleanup startup timed out, continuing...")
        
        # Track startup time for metrics
        app.state.start_time = time.time()
        
        logger.info("Startup complete: HTTP client, RAGService, and production middleware initialized")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown with timeout protection
        logger.info("Shutting down...")
        try:
            if hasattr(app.state, "rate_limiter") and app.state.rate_limiter:
                await asyncio.wait_for(
                    app.state.rate_limiter.stop_cleanup(),
                    timeout=10.0
                )
            if hasattr(app.state, "rag_service") and app.state.rag_service:
                await asyncio.wait_for(
                    app.state.rag_service.aclose(),
                    timeout=10.0
                )
        except asyncio.TimeoutError:
            logger.warning("Some shutdown operations timed out")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            if hasattr(app.state, "httpx_client") and app.state.httpx_client:
                await asyncio.wait_for(
                    app.state.httpx_client.aclose(),
                    timeout=10.0
                )
        logger.info("Shutdown complete: clients closed")

app = FastAPI(
    title="RAG Training Chatbot API",
    description="A simple RAG-based chatbot for employee training with PostgreSQL memory",
    version="1.0.0",
    lifespan=lifespan
)

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

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Training Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render deployment."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "RAG Training Chatbot API"
    }

@app.get("/health/simple")
async def simple_health_check():
    """Simple health check for load balancers - just checks if the service is running."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "RAG Training Chatbot API"
    }

# Startup and shutdown are now handled by the lifespan manager above


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


# This duplicate route was removed - keeping the comprehensive one above


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

        # Run all health checks concurrently with fallbacks for missing methods
        
        health_tasks = []
        
        # Check if methods exist before calling them
        if hasattr(rag_service, 'health_check_openai'):
            openai_task = asyncio.create_task(rag_service.health_check_openai())
            health_tasks.append(('openai', openai_task))
        else:
            openai_health = {"status": "unhealthy", "error": "Method not implemented"}
            
        if hasattr(rag_service, 'health_check_supabase'):
            supabase_task = asyncio.create_task(rag_service.health_check_supabase())
            health_tasks.append(('supabase', supabase_task))
        else:
            supabase_health = {"status": "unhealthy", "error": "Method not implemented"}
            
        if hasattr(rag_service, 'health_check_vector_search'):
            vector_task = asyncio.create_task(rag_service.health_check_vector_search())
            health_tasks.append(('vector_search', vector_task))
        else:
            vector_health = {"status": "unhealthy", "error": "Method not implemented"}
        
        # Wait for existing tasks to complete
        if health_tasks:
            results = await asyncio.gather(*[task for _, task in health_tasks], return_exceptions=True)
            for i, (name, _) in enumerate(health_tasks):
                if name == 'openai':
                    openai_health = results[i]
                elif name == 'supabase':
                    supabase_health = results[i]
                elif name == 'vector_search':
                    vector_health = results[i]

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
async def get_metrics():
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
async def chat_stream(request_data: ChatRequest, request: Request):
    """
    Production-ready streaming chat endpoint with rate limiting and connection management.
    
    Features:
    - Rate limiting per IP to prevent abuse
    - Connection management for monitoring active streams
    - Server-Sent Events (SSE) for real-time responses
    - Optimized for high concurrent throughput on Render
    - Compatible with Next.js/React EventSource API
    """
    # Get client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"
    
    # Apply rate limiting
    rate_limiter: TokenBucketRateLimiter = app.state.rate_limiter
    await rate_limiter.check_rate_limit(client_ip)
    
    # Generate connection ID for tracking
    connection_id = str(uuid.uuid4())
    connection_manager: ConnectionManager = app.state.connection_manager
    
    try:
        rag_service: RAGService = app.state.rag_service
        if not rag_service:
            raise HTTPException(status_code=503, detail="Service unavailable")

        # Register connection
        connection_manager.add_connection(connection_id, client_ip)

        async def generate_sse():
            """Generate Server-Sent Events for streaming response."""
            try:
                async for chunk in rag_service.get_rag_response_stream(
                    user_message=request_data.message,
                    session_id=request_data.session_id,
                    max_tokens=request_data.max_tokens,
                ):
                    # Update connection activity
                    connection_manager.update_activity(connection_id)
                    
                    # Format as SSE
                    data = json.dumps(chunk, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                
                # Send final SSE close event
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.exception("Error in SSE generation for connection %s", connection_id)
                error_data = json.dumps({
                    "type": "error",
                    "error": f"Stream generation failed: {str(e)}",
                    "timestamp": datetime.utcnow().timestamp()
                })
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # Always clean up connection
                connection_manager.remove_connection(connection_id)

        return StreamingResponse(
            generate_sse(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8",
                "X-Accel-Buffering": "no",  # Disable nginx buffering for real-time streaming
                "X-Connection-ID": connection_id,
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
