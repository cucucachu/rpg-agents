# RPG GM Agent

AI Game Master powered by LangGraph and MCP tools.

## Overview

This service provides an AI Game Master that:
- Uses LangGraph for agent orchestration
- Connects to the `rpg-mcp` server for persistent game state
- Supports OpenAI and Anthropic models
- Includes conversation summarization for long sessions
- Exposes a FastAPI server for chat interactions

## Quick Start

### Prerequisites

- Python 3.12+
- Running `rpg-mcp` server (default: http://localhost:8080)
- OpenAI or Anthropic API key

### Setup

1. Copy environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8081
```

### Docker

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f gm-agent
```

## API Endpoints

### Chat

**POST /chat**

Send a message and get a response.

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I look around the tavern", "thread_id": "session-1"}'
```

### Stream Chat

**POST /chat/stream**

Stream responses as server-sent events.

```bash
curl -X POST http://localhost:8081/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "I attack the goblin!"}'
```

Event types:
- `tool_call` - GM is calling a tool
- `tool_result` - Tool execution result
- `token` - Response text
- `done` - Stream complete

### Health

**GET /health**

```bash
curl http://localhost:8081/health
```

### Config

**GET /config**

```bash
curl http://localhost:8081/config
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_URL` | http://localhost:8080 | rpg-mcp server URL |
| `LLM_PROVIDER` | openai | LLM provider (openai/anthropic) |
| `LLM_MODEL` | (provider default) | Model name |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8081 | Server port |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client/UI     │────▶│   GM Agent      │────▶│   rpg-mcp       │
│                 │     │   (LangGraph)   │     │   (MCP Server)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                              ▼                        ▼
                        ┌───────────┐           ┌───────────┐
                        │  OpenAI/  │           │  MongoDB  │
                        │ Anthropic │           │           │
                        └───────────┘           └───────────┘
```

## Development

### Running Tests

```bash
pytest tests/
```

### Local Development

```bash
# With hot reload
DEBUG=true python -m uvicorn src.api.server:app --reload --port 8081
```

## License

MIT
