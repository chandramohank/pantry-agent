# Pantry Agent – Architecture Reference

## 1. High-Level Architecture

The agent is built on **LangGraph `StateGraph`** with a ReAct (Reasoning + Acting) loop at its core. All state flows through a single `PantryAgentState` TypedDict, ensuring every node has a consistent view of the world.

```
START
  │
  ▼
┌──────────────┐
│  load_memory │   ← reads long-term SQLite store (preferences, pantry history)
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ classify_intent │   ← LLM (temp=0): intent label + domain + image detection
└──────┬──────────┘
       │
       ▼
┌───────────────────────────────────┐
│              agent                │   ← ChatOpenAI + bind_tools(all_tools)
│    (domain-aware system prompt)   │   ← system prompt injects intent/domain/memory
└──────┬──────────┬─────────────────┘
       │ no       │ tool_calls?
       │ tools    ▼
       │   ┌────────────┐
       │   │   tools    │   ← LangGraph ToolNode (executes tool calls)
       │   └─────┬──────┘
       │         │ (loops back)
       │         ▼
       │    (agent again)
       │
       ▼
┌─────────────────┐
│ validate_output │   ← checks confidence, bulk thresholds, API errors
└──────┬──────────┘
       │
       │  approval_required?
       ├──── YES ──► ┌──────────────────┐
       │             │ request_approval │  ← interrupt() – graph PAUSES
       │             │  (human gate)    │  ← resumes via Command(resume=...)
       │             └──────┬───────────┘
       │                    │
       └──── NO ─────────►──┘
                            │
                            ▼
                   ┌───────────────┐
                   │ update_memory │   ← LLM extracts facts → merges to SQLite
                   └───────┬───────┘
                            │
                           END
```

## 2. State Model

```python
class PantryAgentState(TypedDict):
    # Conversation (add_messages reducer – appends, not replaces)
    messages: Annotated[list, add_messages]
    user_input: str

    # Vision
    uploaded_images: list[str]          # base64 or presigned URLs

    # Classification
    intent: str                         # e.g. "scan_groceries"
    domain: str                         # Vision | Pantry | Recipes | ...

    # Tool execution
    selected_tool: str
    tool_outputs: list[dict]

    # Domain data surfaces
    extracted_items: list[dict]
    pantry_items: list[dict]
    recipes: list[dict]
    waste_analysis: list[dict]
    sustainability_data: dict

    # Validation & approval
    validation_errors: list[str]
    human_approval_required: bool
    human_approved: bool | None
    approval_reason: str

    # Metadata
    execution_trace: list[dict]
    retry_count: int
    error: str | None

    # Memory
    memory: dict                        # loaded from long-term store each turn
```

## 3. Node Responsibilities

| Node | Input Keys Read | Output Keys Written | LLM Call? |
|------|----------------|---------------------|-----------|
| `load_memory` | — | `memory`, `execution_trace` | No |
| `classify_intent` | `messages`, `user_input`, `uploaded_images` | `intent`, `domain`, `uploaded_images` | Yes (temp=0) |
| `agent` | `messages`, `domain`, `intent`, `memory` | `messages`, `selected_tool`, `execution_trace` | Yes (tools bound) |
| `tools` (ToolNode) | `messages` (tool_calls) | `messages` (ToolMessages) | No |
| `validate_output` | `messages` | `validation_errors`, `human_approval_required`, `approval_reason`, domain data | No |
| `request_approval` | `extracted_items`, `approval_reason` | `human_approved`, `messages` | No (interrupt) |
| `update_memory` | `messages`, `pantry_items` | `memory` | Yes (temp=0) |

## 4. Tool Architecture

Each tool is a LangChain `@tool` with:
- **Pydantic `args_schema`** — enforces structured input
- **Semantic docstring** — optimised for LLM tool selection
- **httpx + tenacity** — retry on transient errors (3 attempts, exponential backoff)
- **`safe_api_call` wrapper** — normalises API errors to `{"error": true, "message": "..."}`

```
User utterance
      │
      ▼
  [Agent LLM]
  .bind_tools(tools)
      │  generates tool_call JSON
      ▼
  [ToolNode]
      │  calls tool.invoke(args)
      ▼
  [Tool function]
      │  safe_api_call → httpx → Pantry REST API
      ▼
  ToolMessage(content=json)
      │
      ▼
  [Agent LLM] (next iteration)
```

## 5. Memory Architecture

### Short-term (conversation)
- Managed by LangGraph's **MemorySaver checkpointer**
- Scoped to `thread_id` in `config["configurable"]`
- Automatically persisted between `.invoke()` calls with the same thread_id

### Long-term (cross-session)
- SQLite database at `MEMORY_DB_PATH`
- Keyed by `user_id`
- Schema: dietary preferences, allergies, favourite recipes, substitutions, pantry snapshot, waste patterns
- Updated at end of every turn by `update_memory` node using LLM summarisation
- `merge_memory_updates()` unions list fields (deduped) and merges dict fields

## 6. Human-in-the-Loop Protocol

```
Agent detects high-risk operation
        │
        ▼
validate_output sets human_approval_required=True
        │
        ▼
Graph routes to request_approval node
        │
        ▼
interrupt(payload) — graph PAUSES, payload returned to caller
        │
Caller reviews payload
        │
        ▼
app.invoke(Command(resume={"approved": True/False, "feedback": "..."}), config)
        │
        ▼
Graph resumes from request_approval → update_memory → END
```

## 7. Error Handling Strategy

| Error Type | Handling |
|-----------|---------|
| API 4xx (client error) | Surface to user immediately; no retry |
| API 5xx / network error | Retry 3× with exponential backoff (tenacity) |
| Low confidence | Route to human approval gate |
| LLM classification failure | Fall back to `intent=general_query, domain=General` |
| LLM summarisation failure | Log warning; skip memory update silently |
| Tool exception | `ToolException` surfaces to agent; agent decides next action |

## 8. Retry Strategy

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((ConnectError, ReadTimeout, WriteTimeout)),
    reraise=True,
)
def api_get(path, params=None): ...
```

- Maximum 3 attempts
- Wait: 1s → 2s → 4s (capped at 8s)
- Only retries on transient network errors, not HTTP 4xx
- Reraises on final failure

## 9. Observability Strategy

| Layer | Implementation |
|-------|---------------|
| **LangSmith tracing** | Set `LANGCHAIN_TRACING_V2=true` – every LLM call, tool invocation, and node execution is traced |
| **Execution trace** | `state["execution_trace"]` accumulates a structured JSON log of every node and its key outputs |
| **Structured logs** | `logging.basicConfig` with ISO timestamp format; `log_agent_run()` emits a consolidated entry |
| **Timing** | `@traced_node` decorator and `timed_operation()` context manager |
| **Tool metrics** | Tool names logged at INFO; confidence logged at WARNING when below threshold |

## 10. Evaluation Strategy

### Offline evals (CI)
- Unit tests for every tool (mocked API)
- State invariant tests
- Routing logic tests (should_continue, needs_approval)

### Online evals (LangSmith)
- Per-tool success rate
- Intent classification accuracy
- Human approval rate (target < 15%)
- Average tool calls per turn
- Memory update success rate

### Golden dataset
Create representative test cases for each domain:
```python
{"input": "Scan my fridge", "expected_tool": "extract_ingredients_from_image", "expected_domain": "Vision"}
{"input": "What can I cook?", "expected_tool": "recommend_recipes", "expected_domain": "Recipes"}
{"input": "Create a keto meal plan", "expected_tool": "create_diet_plan", "expected_domain": "Diet"}
```

## 11. Security Considerations

- API key stored in `.env`, never hardcoded
- `PANTRY_API_KEY` sent as `X-API-Key` header (not in URL)
- Input validation via Pydantic (type coercion + constraints)
- No `eval()` or shell execution anywhere
- All SQL operations use parameterised queries (SQLite, no string interpolation)
- Image data accepted as base64 or presigned URL – never written to filesystem by the agent
