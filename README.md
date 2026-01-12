# mux-evals

Language-agnostic evaluation suite for the mux agent framework.

This repo contains evals that can be run against both:
- **mux-rs** (Rust implementation)
- **mux** (Go implementation)

## Structure

```
mux-evals/
├── evals/                    # Language-agnostic eval definitions
│   ├── core-tools.jsonl      # Tool registry and execution
│   ├── hooks.jsonl           # Hook system lifecycle
│   ├── agent-loop.jsonl      # Agentic loop behavior
│   ├── subagents.jsonl       # Subagent spawning/resume
│   ├── transcript.jsonl      # Conversation persistence
│   ├── mcp-protocol.jsonl    # MCP server integration
│   └── llm-providers.jsonl   # LLM provider integration
├── runners/
│   ├── rust/                 # Rust runner for mux-rs
│   └── go/                   # Go runner for mux
└── README.md
```

## Eval Format

Each `.jsonl` file contains one eval per line in JSON format:

```json
{
  "id": "tool-001",
  "name": "tool_execution_basic",
  "description": "Tool registry accepts and executes simple tools",
  "category": "tools",
  "given": { ... },
  "when": { ... },
  "then": { ... }
}
```

- **id**: Unique identifier for the eval
- **name**: Human-readable name
- **description**: What this eval validates
- **category**: Category for filtering (tools, hooks, agent, subagent, transcript, mcp, llm)
- **given**: Initial state/setup
- **when**: Action to perform
- **then**: Expected outcomes

### Optional Fields

- **provider**: LLM provider (anthropic, openai, etc.)
- **requires_key**: Environment variable that must be set (e.g., ANTHROPIC_API_KEY)

## Running Evals

### Rust (mux-rs)

```bash
cd runners/rust
cargo run -- --evals ../../evals

# Filter by category
cargo run -- --category tools

# Filter by specific eval
cargo run -- --id tool-001

# Verbose output
cargo run -- --verbose

# Only show failures
cargo run -- --failures-only
```

### Go (mux)

```bash
cd runners/go
go run . -evals ../../evals

# Filter by category
go run . -category tools

# Filter by specific eval
go run . -id tool-001

# Verbose output
go run . -verbose

# Only show failures
go run . -failures-only
```

## Categories

| Category | Description | Count |
|----------|-------------|-------|
| tools | Tool registry, execution, error handling | 5 |
| hooks | Hook lifecycle, blocking, chaining | 6 |
| agent | Agentic loop, iterations, tool calling | 6 |
| subagent | Spawning, inheritance, resume | 5 |
| transcript | Save/load conversation history | 5 |
| mcp | MCP protocol, tool discovery, execution | 5 |
| llm | LLM provider integration (Anthropic, OpenAI) | 6 |

## Adding New Evals

1. Add a new line to the appropriate `.jsonl` file
2. Follow the existing format
3. Run against both implementations to verify behavior

## CI Integration

Both runners exit with code 1 if any evals fail, making them suitable for CI:

```yaml
# GitHub Actions example
- name: Run mux-rs evals
  run: |
    cd mux-evals/runners/rust
    cargo run -- --evals ../../evals

- name: Run mux evals
  run: |
    cd mux-evals/runners/go
    go run . -evals ../../evals
```

## Environment Variables

For LLM provider evals:

```bash
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
```

Evals requiring API keys will be skipped if the key is not set.
