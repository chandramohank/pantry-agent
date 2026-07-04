# Pantry Agent

A production-ready **LangGraph AI Agent** that acts as a personal kitchen assistant for the **Intelligent Pantry** application.

## Features

| Capability | Description |
|---|---|
| **Vision AI** | Scan groceries, fridges, pantry shelves, and receipts to extract ingredients |
| **Pantry Management** | Add, retrieve, and track pantry inventory |
| **AI Extraction** | Parse free-form text shopping lists into structured pantry items |
| **Recipe Intelligence** | Recommend recipes from current pantry contents |
| **Diet Planning** | Generate personalised weekly meal plans |
| **Cooking Copilot** | Conversational cooking assistance and substitution advice |
| **Waste Reduction** | Identify expiring items and recommend recipes that use them |
| **Sustainability Insights** | Carbon footprint and sustainability recommendations |
| **Human-in-the-Loop** | Approval gate for bulk imports, low-confidence extractions, and deletions |
| **Memory** | Short-term (conversation) and long-term (preferences, pantry history) memory |
| **Observability** | Full LangSmith tracing, structured execution logs |

## Architecture

```
START → load_memory → classify_intent → agent ⟷ tools
                                           ↓ (no more tool calls)
                                      validate_output
                                           ↓
                               ┌── needs_approval? ──┐
                               │YES                  │NO
                         request_approval            │
                               └──────────┬──────────┘
                                     update_memory
                                           ↓
                                          END
```

See [`docs/architecture.md`](docs/architecture.md) for full diagrams and design rationale.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# edit .env and set OPENAI_API_KEY and PANTRY_API_BASE_URL
# defaults already include Azure OpenAI endpoint, Foundry project endpoint, and model=gpt-5.5

# 3. Run the agent (interactive)
python -c "
from pantry_agent.agent import create_agent
app = create_agent()
result = app.invoke({'messages': [('human', 'What can I cook tonight?')], 'user_input': 'What can I cook tonight?'}, config={'configurable': {'thread_id': 'demo'}})
print(result['messages'][-1].content)
"

# 4. Run tests
pytest

# 5. Run API server
python -m uvicorn pantry_agent.api:app --host 0.0.0.0 --port 8080
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health check |
| `/chat/sse` | `POST` | Runs one chat turn and returns the final response as JSON |

### `POST /chat/sse`

Request body:

```json
{
      "user_input": "What can I cook with eggs and spinach?",
      "thread_id": "demo-thread",
      "user_id": "user-123",
      "uploaded_images": []
}
```

Example call:

```bash
curl -X POST http://localhost:8080/chat/sse \
      -H "Content-Type: application/json" \
      -d '{
            "user_input": "What can I cook with eggs and spinach?",
            "thread_id": "demo-thread",
            "user_id": "user-123",
            "uploaded_images": []
      }'
```

Response shape:

```json
{
      "schema_version": "1.0",
  "thread_id": "demo-thread",
      "message": "...final assistant response...",
      "payload": {},
      "artifacts": [],
      "actions": [],
      "context": {
            "intent": "get_recipes",
            "domain": "Recipes"
      },
      "trace": [],
      "approval": null,
      "errors": []
}
```

## Project Structure

```
src/pantry_agent/
├── agent.py               # LangGraph StateGraph assembly
├── api_client.py          # Shared httpx client with retry logic
├── config.py              # Pydantic settings
├── state.py               # PantryAgentState TypedDict
├── models/schemas.py      # Pydantic I/O models for all API endpoints
├── tools/                 # Semantic tool definitions (14 tools)
├── nodes/                 # Graph node implementations
├── memory/                # Short-term & long-term memory
├── prompts/               # System prompt templates
└── observability/         # LangSmith + structured tracing
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `AZURE_OPENAI_API_KEY` | No | Alias for `OPENAI_API_KEY` when using Azure OpenAI credentials |
| `PANTRY_API_BASE_URL` | ✅ | Base URL of the Intelligent Pantry REST API |
| `OPENAI_MODEL` | No | Model name (default: `gpt-5.5`) |
| `AZURE_OPENAI_ENDPOINT` | No | Azure OpenAI endpoint (default: `https://intelligentpantry-ai-foundry.openai.azure.com/openai/v1`) |
| `FOUNDRY_PROJECT_ENDPOINT` | No | Azure AI Foundry project endpoint metadata (default: `https://intelligentpantry-ai-foundry.services.ai.azure.com/api/projects/IntelligentPantry-Scan`) |
| `LANGCHAIN_API_KEY` | No | LangSmith API key for tracing |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing |
| `VISION_CONFIDENCE_THRESHOLD` | No | Min confidence for auto-save (default: 0.80) |
| `HUMAN_APPROVAL_REQUIRED_FOR_BULK` | No | Item count threshold for approval (default: 5) |
| `MEMORY_BACKEND` | No | `sqlite` or `memory` (default: `sqlite`) |

## License

MIT
