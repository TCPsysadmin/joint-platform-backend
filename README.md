# Production RAG Chatbot Backend

A production-ready RAG (Retrieval-Augmented Generation) chatbot backend built for The Collaborative Process (TCP) training. Optimized for high throughput and concurrent users on Render.

## 🚀 Features

### Core Functionality
- **RAG Pipeline**: OpenAI embeddings + Supabase vector search + GPT-4o-mini chat completions
- **Streaming Responses**: Real-time Server-Sent Events (SSE) for responsive UX
- **Conversation Memory**: Persistent chat history in Supabase PostgreSQL
- **Session Management**: Multi-user session handling with UUID-based sessions

### Production Features
- **High Concurrency**: Async FastAPI with optimized connection pooling (500+ concurrent connections)
- **Rate Limiting**: Token bucket algorithm prevents abuse (60 RPM per IP, 10 burst)
- **Connection Management**: Active connection tracking and graceful shutdown
- **Comprehensive Monitoring**: Health checks, metrics endpoint, and structured logging
- **Error Handling**: Retry logic, circuit breakers, and graceful degradation
- **Security**: CORS configuration, input validation, and non-root Docker user

## 📡 API Endpoints

### Chat Endpoints
- `POST /chat` - Standard chat (JSON response)
- `POST /chat/stream` - **Streaming chat (SSE)** - Recommended for production
- `POST /session/new` - Create new chat session
- `GET /session/{session_id}/history` - Get conversation history
- `DELETE /session/{session_id}` - Clear session

### Monitoring & Health
- `GET /health` - Comprehensive health check (OpenAI + Supabase + Vector Search)
- `GET /health/simple` - Simple health check for load balancers
- `GET /metrics` - Production metrics (connections, uptime, rate limiting)

## 🛠️ Quick Start

### Local Development

1. **Clone and setup**:
```bash
git clone <repository>
cd rag-chatbot-backend
cp .env.sample .env
# Edit .env with your API keys
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run locally**:
```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Development

```bash
docker-compose up --build
```

## 🚀 Production Deployment

### Render Deployment (Recommended)

1. **Connect your repository** to Render
2. **Use the included `render.yaml`** for automatic configuration
3. **Set environment variables** in Render dashboard:
   - `OPENAI_API_KEY`
   - `SUPABASE_URL` 
   - `SUPABASE_KEY`
   - `CORS_ALLOWED_ORIGINS` (your frontend domains)

4. **Deploy**: Render will automatically build and deploy

### Manual Docker Deployment

```bash
# Build production image
docker build -t rag-chatbot-prod .

# Run with production settings
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name rag-chatbot \
  rag-chatbot-prod
```

## 🔧 Configuration

### Environment Variables

#### Required
- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL  
- `SUPABASE_KEY`: Your Supabase anon key

#### Optional (with defaults)
- `OPENAI_MODEL_NAME`: Model to use (default: `gpt-4o-mini`)
- `OPENAI_TEMPERATURE`: Response creativity (default: `0.5`)
- `SUPABASE_MATCH_FN`: Vector search function (default: `match_documents_justin`)
- `SUPABASE_MATCH_THRESHOLD`: Similarity threshold (default: `0.4`)
- `CORS_ALLOWED_ORIGINS`: Comma-separated frontend URLs (default: `*`)

#### Production Tuning
- `RATE_LIMIT_RPM`: Requests per minute per IP (default: `60`)
- `RATE_LIMIT_BURST`: Burst capacity (default: `10`)
- `MAX_CONNECTIONS`: Max concurrent streams (default: `1000`)

## 💻 Frontend Integration

### React/Next.js Streaming Client

```javascript
// Use the included client_example.js for complete implementation
const response = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "What is The Collaborative Process?",
    session_id: sessionId,
    max_tokens: 500
  })
});

const reader = response.body.getReader();
// Handle SSE stream - see client_example.js for full code
```

### SSE Event Types
- `session`: Session ID and initialization
- `status`: Processing updates ("Processing query...", "Searching knowledge base...")
- `sources`: Retrieved document sources
- `content`: Streaming response content (word by word)
- `done`: Response completion with timing
- `error`: Error messages

## 📊 Monitoring & Observability

### Health Monitoring
```bash
# Simple health check (for load balancers)
curl https://your-app.onrender.com/health/simple

# Comprehensive health check
curl https://your-app.onrender.com/health

# Production metrics
curl https://your-app.onrender.com/metrics
```

### Key Metrics
- Active streaming connections
- Average response times
- Rate limiting statistics
- OpenAI/Supabase connectivity
- Vector search performance

## 🏗️ Architecture

```
Frontend (Next.js/React)
    ↓ SSE Stream
FastAPI Backend
    ├── Rate Limiter (Token Bucket)
    ├── Connection Manager
    ├── RAG Service
    │   ├── OpenAI Embeddings
    │   ├── Supabase Vector Search
    │   └── OpenAI Chat Completions (Streaming)
    └── Memory Service (Supabase PostgreSQL)
```

## 🔒 Security Features

- Input validation and sanitization
- Rate limiting per IP address
- CORS configuration for frontend domains
- Non-root Docker container
- Secure environment variable handling
- Request logging and monitoring

## 📈 Performance Optimizations

- **Async/Await**: Full async pipeline for I/O operations
- **Connection Pooling**: Shared httpx client with 500 max connections
- **HTTP/2**: Enabled for better multiplexing
- **Streaming**: Reduces perceived latency
- **Token Bucket**: Allows burst traffic while preventing abuse
- **Cleanup Tasks**: Automatic memory management

## 🐛 Troubleshooting

### Common Issues

1. **Rate Limited**: Reduce request frequency or increase `RATE_LIMIT_RPM`
2. **Connection Timeout**: Check `SUPABASE_URL` and `OPENAI_API_KEY`
3. **CORS Errors**: Update `CORS_ALLOWED_ORIGINS` with your frontend domain
4. **Memory Issues**: Monitor `/metrics` endpoint for connection leaks

### Logs
```bash
# View application logs
docker logs rag-chatbot

# Follow logs in real-time
docker logs -f rag-chatbot
```

## 📝 License

This project is configured for The Collaborative Process (TCP) training content. Modify prompts and context as needed for your use case.