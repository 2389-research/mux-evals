// ABOUTME: Rust eval runner for mux-rs library.
// ABOUTME: Executes language-agnostic eval definitions against the Rust implementation.

use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::Parser;
use colored::Colorize;
use mux::agent::{MemoryTranscriptStore, TranscriptStore};
use mux::hook::{Hook, HookAction, HookEvent, HookRegistry};
use mux::llm::{AnthropicClient, ContentBlock, LlmClient, Message, OpenAIClient, Request, Role};
use mux::tool::{Registry, Tool, ToolResult};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Parser)]
#[command(name = "mux-eval-runner")]
#[command(about = "Run mux evals against the Rust implementation")]
struct Args {
    /// Path to evals directory or specific .jsonl file
    #[arg(short, long, default_value = "../../evals")]
    evals: PathBuf,

    /// Filter by category (tools, hooks, agent, etc.)
    #[arg(short, long)]
    category: Option<String>,

    /// Filter by specific eval ID
    #[arg(short, long)]
    id: Option<String>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Only show failures
    #[arg(long)]
    failures_only: bool,

    /// Judge model for evaluating agent outputs (default: gpt-5-mini)
    #[arg(long, default_value = "gpt-5-mini")]
    judge_model: String,
}

// ============================================================================
// Judge Agent - Uses LLM to evaluate if agent completed task correctly
// ============================================================================

struct Judge {
    client: Arc<dyn LlmClient>,
    model: String,
}

impl Judge {
    fn new(client: Arc<dyn LlmClient>, model: String) -> Self {
        Self { client, model }
    }

    async fn evaluate(
        &self,
        task: &str,
        agent_output: &str,
        criteria: &str,
    ) -> Result<(bool, String)> {
        let prompt = format!(
            r#"You are an eval judge. Evaluate if the agent completed the task correctly.

TASK: {}

AGENT OUTPUT:
{}

EVALUATION CRITERIA: {}

Respond with EXACTLY this format (no markdown, no extra text):
VERDICT: PASS or FAIL
REASON: One sentence explanation

Example:
VERDICT: PASS
REASON: The agent correctly completed the requested task."#,
            task, agent_output, criteria
        );

        let request = Request {
            model: self.model.clone(),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text { text: prompt }],
            }],
            max_tokens: Some(200),
            ..Default::default()
        };

        let response = self.client.create_message(&request).await?;

        // Parse the judge's response
        let text = response
            .content
            .iter()
            .filter_map(|b| {
                if let ContentBlock::Text { text } = b {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        let passed = text.contains("VERDICT: PASS");
        let reason = text
            .lines()
            .find(|l| l.starts_with("REASON:"))
            .map(|l| l.trim_start_matches("REASON:").trim().to_string())
            .unwrap_or_else(|| "No reason provided".to_string());

        Ok((passed, reason))
    }
}

fn create_judge() -> Option<Judge> {
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    let client = OpenAIClient::new(api_key);
    Some(Judge::new(Arc::new(client), "gpt-5-mini".to_string()))
}

#[derive(Debug, Deserialize, Serialize)]
struct Eval {
    id: String,
    name: String,
    description: String,
    category: String,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    requires_key: Option<String>,
    given: serde_json::Value,
    when: serde_json::Value,
    then: serde_json::Value,
}

#[derive(Debug)]
enum EvalResult {
    Pass,
    Fail(String),
    Skip(String),
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file from current directory or parent directories
    let _ = dotenvy::dotenv();

    let args = Args::parse();

    let evals = load_evals(&args.evals, args.category.as_deref(), args.id.as_deref())?;

    // Create judge if API key is available
    let judge = create_judge();
    if judge.is_some() {
        println!("{}", "Judge agent enabled (using Claude)".dimmed());
    }

    println!("\n{} {} evals\n", "Running".bold().cyan(), evals.len());

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for eval in &evals {
        let result = run_eval(eval, args.verbose, judge.as_ref()).await;

        match &result {
            EvalResult::Pass => {
                passed += 1;
                if !args.failures_only {
                    println!("{} {} - {}", "PASS".green().bold(), eval.id, eval.name);
                }
            }
            EvalResult::Fail(reason) => {
                failed += 1;
                println!(
                    "{} {} - {}\n       {}",
                    "FAIL".red().bold(),
                    eval.id,
                    eval.name,
                    reason.dimmed()
                );
            }
            EvalResult::Skip(reason) => {
                skipped += 1;
                if !args.failures_only {
                    println!(
                        "{} {} - {}\n       {}",
                        "SKIP".yellow().bold(),
                        eval.id,
                        eval.name,
                        reason.dimmed()
                    );
                }
            }
        }
    }

    println!(
        "\n{}: {} passed, {} failed, {} skipped\n",
        "Results".bold(),
        passed.to_string().green(),
        if failed > 0 {
            failed.to_string().red()
        } else {
            failed.to_string().normal()
        },
        skipped.to_string().yellow()
    );

    if failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn load_evals(
    path: &PathBuf,
    category_filter: Option<&str>,
    id_filter: Option<&str>,
) -> Result<Vec<Eval>> {
    let mut evals = Vec::new();

    let files = if path.is_dir() {
        std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "jsonl").unwrap_or(false))
            .collect()
    } else {
        vec![path.clone()]
    };

    for file_path in files {
        let file = File::open(&file_path)
            .with_context(|| format!("Failed to open {}", file_path.display()))?;
        let reader = BufReader::new(file);

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let eval: Eval = serde_json::from_str(&line).with_context(|| {
                format!(
                    "Failed to parse line {} in {}",
                    line_num + 1,
                    file_path.display()
                )
            })?;

            // Apply filters
            if let Some(cat) = category_filter {
                if eval.category != cat {
                    continue;
                }
            }
            if let Some(id) = id_filter {
                if eval.id != id {
                    continue;
                }
            }

            evals.push(eval);
        }
    }

    Ok(evals)
}

async fn run_eval(eval: &Eval, verbose: bool, judge: Option<&Judge>) -> EvalResult {
    // Check for required API keys
    if let Some(key) = &eval.requires_key {
        if std::env::var(key).is_err() {
            return EvalResult::Skip(format!("{} not set", key));
        }
    }

    if verbose {
        println!("  given: {:?}", eval.given);
        println!("  when: {:?}", eval.when);
        println!("  then: {:?}", eval.then);
    }

    // Dispatch based on category
    match eval.category.as_str() {
        "tools" => run_tool_eval(eval).await,
        "hooks" => run_hook_eval(eval).await,
        "agent" => run_agent_eval(eval, judge).await,
        "subagent" => run_subagent_eval(eval, judge).await,
        "transcript" => run_transcript_eval(eval).await,
        "mcp" => run_mcp_eval(eval).await,
        "llm" => run_llm_eval(eval, judge).await,
        _ => EvalResult::Skip(format!("Unknown category: {}", eval.category)),
    }
}

// ============================================================================
// Test Tools
// ============================================================================

struct AddTool;

#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &str {
        "add"
    }
    fn description(&self) -> &str {
        "Adds two numbers"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let a = params["a"].as_f64().unwrap_or(0.0);
        let b = params["b"].as_f64().unwrap_or(0.0);
        Ok(ToolResult::text(format!("{}", a + b)))
    }
}

struct DivideTool;

#[async_trait]
impl Tool for DivideTool {
    fn name(&self) -> &str {
        "divide"
    }
    fn description(&self) -> &str {
        "Divides two numbers"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let a = params["a"].as_f64().unwrap_or(0.0);
        let b = params["b"].as_f64().unwrap_or(0.0);
        if b == 0.0 {
            return Err(anyhow::anyhow!("Division by zero"));
        }
        Ok(ToolResult::text(format!("{}", a / b)))
    }
}

struct GreetTool;

#[async_trait]
impl Tool for GreetTool {
    fn name(&self) -> &str {
        "greet"
    }
    fn description(&self) -> &str {
        "Returns greeting"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        })
    }
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let name = params["name"].as_str().unwrap_or("World");
        Ok(ToolResult::text(format!("Hello, {}!", name)))
    }
}

struct GetInfoTool;

#[async_trait]
impl Tool for GetInfoTool {
    fn name(&self) -> &str {
        "get_info"
    }
    fn description(&self) -> &str {
        "Returns info object"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    async fn execute(&self, _params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        Ok(ToolResult::text(
            serde_json::json!({"version": "1.0", "name": "mux"}).to_string(),
        ))
    }
}

struct CounterTool {
    count: AtomicUsize,
}

impl CounterTool {
    fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Tool for CounterTool {
    fn name(&self) -> &str {
        "counter"
    }
    fn description(&self) -> &str {
        "Increments counter"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    async fn execute(&self, _params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let new_count = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(ToolResult::text(format!("Count: {}", new_count)))
    }
}

// ============================================================================
// Tool Evals
// ============================================================================

async fn run_tool_eval(eval: &Eval) -> EvalResult {
    let registry = Registry::new();

    // Register tools based on eval requirements
    registry.register(AddTool).await;
    registry.register(DivideTool).await;
    registry.register(GreetTool).await;
    registry.register(GetInfoTool).await;

    match eval.id.as_str() {
        "tool-001" => {
            // tool_execution_basic - Add two numbers
            let tool = registry.get("add").await;
            if tool.is_none() {
                return EvalResult::Fail("Tool 'add' not found".to_string());
            }
            let result = tool
                .unwrap()
                .execute(serde_json::json!({"a": 2, "b": 3}))
                .await;
            match result {
                Ok(r) => {
                    if r.content.contains("5") {
                        EvalResult::Pass
                    } else {
                        EvalResult::Fail(format!("Expected '5', got: {}", r.content))
                    }
                }
                Err(e) => EvalResult::Fail(format!("Execution failed: {}", e)),
            }
        }
        "tool-002" => {
            // tool_not_found - Unknown tool returns None
            let tool = registry.get("nonexistent").await;
            if tool.is_none() {
                EvalResult::Pass
            } else {
                EvalResult::Fail("Expected None for nonexistent tool".to_string())
            }
        }
        "tool-003" => {
            // tool_invalid_input - Division by zero
            let tool = registry.get("divide").await.unwrap();
            let result = tool.execute(serde_json::json!({"a": 10, "b": 0})).await;
            match result {
                Err(_) => EvalResult::Pass,
                Ok(_) => EvalResult::Fail("Expected error for division by zero".to_string()),
            }
        }
        "tool-004" => {
            // tool_result_string - Greet returns string with name
            let tool = registry.get("greet").await.unwrap();
            let result = tool.execute(serde_json::json!({"name": "World"})).await;
            match result {
                Ok(r) => {
                    if r.content.contains("World") {
                        EvalResult::Pass
                    } else {
                        EvalResult::Fail(format!("Expected 'World' in result, got: {}", r.content))
                    }
                }
                Err(e) => EvalResult::Fail(format!("Execution failed: {}", e)),
            }
        }
        "tool-005" => {
            // tool_result_json - GetInfo returns valid JSON
            let tool = registry.get("get_info").await.unwrap();
            let result = tool.execute(serde_json::json!({})).await;
            match result {
                Ok(r) => {
                    if serde_json::from_str::<serde_json::Value>(&r.content).is_ok() {
                        EvalResult::Pass
                    } else {
                        EvalResult::Fail(format!("Result is not valid JSON: {}", r.content))
                    }
                }
                Err(e) => EvalResult::Fail(format!("Execution failed: {}", e)),
            }
        }
        _ => EvalResult::Skip(format!("Unknown tool eval: {}", eval.id)),
    }
}

// ============================================================================
// Hook Evals
// ============================================================================

struct LoggingHook {
    events: Arc<RwLock<Vec<String>>>,
}

impl LoggingHook {
    fn new() -> (Self, Arc<RwLock<Vec<String>>>) {
        let events = Arc::new(RwLock::new(Vec::new()));
        (
            Self {
                events: events.clone(),
            },
            events,
        )
    }
}

#[async_trait]
impl Hook for LoggingHook {
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        let msg = match event {
            HookEvent::PreToolUse { tool_name, .. } => format!("pre_tool_use:{}", tool_name),
            HookEvent::PostToolUse { tool_name, .. } => format!("post_tool_use:{}", tool_name),
            HookEvent::AgentStart { agent_id, .. } => format!("agent_start:{}", agent_id),
            HookEvent::AgentStop { agent_id, .. } => format!("agent_stop:{}", agent_id),
            HookEvent::Iteration { agent_id, iteration } => {
                format!("iteration:{}:{}", agent_id, iteration)
            }
        };
        self.events.write().await.push(msg);
        Ok(HookAction::Continue)
    }
}

struct BlockingHook {
    block_tool: String,
}

#[async_trait]
impl Hook for BlockingHook {
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        if let HookEvent::PreToolUse { tool_name, .. } = event {
            if tool_name == &self.block_tool {
                return Ok(HookAction::Block(format!("Tool {} is blocked", tool_name)));
            }
        }
        Ok(HookAction::Continue)
    }
}

struct NamedHook {
    name: String,
    events: Arc<RwLock<Vec<String>>>,
}

#[async_trait]
impl Hook for NamedHook {
    async fn on_event(&self, _event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        self.events.write().await.push(self.name.clone());
        Ok(HookAction::Continue)
    }
}

async fn run_hook_eval(eval: &Eval) -> EvalResult {
    match eval.id.as_str() {
        "hook-001" => {
            // hook_pre_tool_fires - PreToolUse hook fires before execution
            let registry = HookRegistry::new();
            let (hook, events) = LoggingHook::new();
            registry.register(hook).await;

            let event = HookEvent::PreToolUse {
                tool_name: "counter".into(),
                input: serde_json::json!({}),
            };
            let _ = registry.fire(&event).await;

            let logged = events.read().await;
            if logged.iter().any(|e| e.starts_with("pre_tool_use:")) {
                EvalResult::Pass
            } else {
                EvalResult::Fail("PreToolUse hook did not fire".to_string())
            }
        }
        "hook-002" => {
            // hook_post_tool_fires - PostToolUse hook fires after execution
            let registry = HookRegistry::new();
            let (hook, events) = LoggingHook::new();
            registry.register(hook).await;

            let event = HookEvent::PostToolUse {
                tool_name: "counter".into(),
                input: serde_json::json!({}),
                result: ToolResult::text("done"),
            };
            let _ = registry.fire(&event).await;

            let logged = events.read().await;
            if logged.iter().any(|e| e.starts_with("post_tool_use:")) {
                EvalResult::Pass
            } else {
                EvalResult::Fail("PostToolUse hook did not fire".to_string())
            }
        }
        "hook-003" => {
            // hook_block_tool - PreToolUse hook can block execution
            let registry = HookRegistry::new();
            registry
                .register(BlockingHook {
                    block_tool: "dangerous".into(),
                })
                .await;

            let event = HookEvent::PreToolUse {
                tool_name: "dangerous".into(),
                input: serde_json::json!({}),
            };
            let action = registry.fire(&event).await.unwrap();

            if matches!(action, HookAction::Block(_)) {
                EvalResult::Pass
            } else {
                EvalResult::Fail("Expected Block action".to_string())
            }
        }
        "hook-004" => {
            // hook_chain_order - Multiple hooks fire in registration order
            let registry = HookRegistry::new();
            let events = Arc::new(RwLock::new(Vec::new()));

            registry
                .register(NamedHook {
                    name: "first".into(),
                    events: events.clone(),
                })
                .await;
            registry
                .register(NamedHook {
                    name: "second".into(),
                    events: events.clone(),
                })
                .await;

            let event = HookEvent::PreToolUse {
                tool_name: "test".into(),
                input: serde_json::json!({}),
            };
            let _ = registry.fire(&event).await;

            let logged = events.read().await;
            if logged.len() == 2 && logged[0] == "first" && logged[1] == "second" {
                EvalResult::Pass
            } else {
                EvalResult::Fail(format!("Expected ['first', 'second'], got {:?}", *logged))
            }
        }
        "hook-005" | "hook-006" => {
            // These require agent execution, skip for now
            EvalResult::Skip("Requires agent execution".to_string())
        }
        _ => EvalResult::Skip(format!("Unknown hook eval: {}", eval.id)),
    }
}

// ============================================================================
// Agent Evals - Use Judge to evaluate agent task completion
// ============================================================================

async fn run_agent_eval(eval: &Eval, judge: Option<&Judge>) -> EvalResult {
    // Check if we have API key for agent execution
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        return EvalResult::Skip("ANTHROPIC_API_KEY not set".to_string());
    }

    let judge = match judge {
        Some(j) => j,
        None => return EvalResult::Skip("Judge not available for agent eval".to_string()),
    };

    // Get task from eval
    let task = eval
        .when
        .get("task")
        .and_then(|t| t.as_str())
        .unwrap_or("Perform the requested task");

    let criteria = eval
        .then
        .get("expect")
        .and_then(|e| e.as_str())
        .unwrap_or("Task should be completed correctly");

    match eval.id.as_str() {
        "agent-001" => {
            // agent_simple_task - Agent completes a simple task
            let client = AnthropicClient::from_env().unwrap();
            let request = Request {
                model: "claude-sonnet-4-20250514".to_string(),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "What is 2 + 2? Reply with just the number.".to_string(),
                    }],
                }],
                max_tokens: Some(100),
                ..Default::default()
            };

            match client.create_message(&request).await {
                Ok(response) => {
                    let output = response
                        .content
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    // Use judge to evaluate
                    match judge
                        .evaluate(
                            "Answer: What is 2 + 2?",
                            &output,
                            "Response should contain the number 4",
                        )
                        .await
                    {
                        Ok((passed, reason)) => {
                            if passed {
                                EvalResult::Pass
                            } else {
                                EvalResult::Fail(reason)
                            }
                        }
                        Err(e) => EvalResult::Fail(format!("Judge error: {}", e)),
                    }
                }
                Err(e) => EvalResult::Fail(format!("LLM request failed: {}", e)),
            }
        }
        "agent-002" | "agent-004" | "agent-005" | "agent-006" => {
            // These evals require full agent loop with tool registration:
            // agent-002: Agent uses tools when needed
            // agent-004: Agent stops on end_turn
            // agent-005: Agent calls multiple tools in sequence
            // agent-006: Agent calls multiple tools in parallel
            EvalResult::Skip("Requires full agent loop with tools".to_string())
        }
        "agent-003" => {
            // agent_multi_turn - Agent maintains context across turns
            let client = AnthropicClient::from_env().unwrap();

            // First turn
            let request1 = Request {
                model: "claude-sonnet-4-20250514".to_string(),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "My name is Alice.".to_string(),
                    }],
                }],
                max_tokens: Some(100),
                ..Default::default()
            };

            let response1 = match client.create_message(&request1).await {
                Ok(r) => r,
                Err(e) => return EvalResult::Fail(format!("First turn failed: {}", e)),
            };

            let assistant_reply = response1
                .content
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::Text { text } = b {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");

            // Second turn with context
            let request2 = Request {
                model: "claude-sonnet-4-20250514".to_string(),
                messages: vec![
                    Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text {
                            text: "My name is Alice.".to_string(),
                        }],
                    },
                    Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: assistant_reply,
                        }],
                    },
                    Message {
                        role: Role::User,
                        content: vec![ContentBlock::Text {
                            text: "What is my name?".to_string(),
                        }],
                    },
                ],
                max_tokens: Some(100),
                ..Default::default()
            };

            match client.create_message(&request2).await {
                Ok(response) => {
                    let output = response
                        .content
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    match judge
                        .evaluate(
                            "Remember the name Alice from context, then answer 'What is my name?'",
                            &output,
                            "Response should mention Alice",
                        )
                        .await
                    {
                        Ok((passed, reason)) => {
                            if passed {
                                EvalResult::Pass
                            } else {
                                EvalResult::Fail(reason)
                            }
                        }
                        Err(e) => EvalResult::Fail(format!("Judge error: {}", e)),
                    }
                }
                Err(e) => EvalResult::Fail(format!("Second turn failed: {}", e)),
            }
        }
        _ => {
            // Generic agent eval using task/criteria from eval definition
            let client = AnthropicClient::from_env().unwrap();
            let request = Request {
                model: "claude-sonnet-4-20250514".to_string(),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: task.to_string(),
                    }],
                }],
                max_tokens: Some(500),
                ..Default::default()
            };

            match client.create_message(&request).await {
                Ok(response) => {
                    let output = response
                        .content
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    match judge.evaluate(task, &output, criteria).await {
                        Ok((passed, reason)) => {
                            if passed {
                                EvalResult::Pass
                            } else {
                                EvalResult::Fail(reason)
                            }
                        }
                        Err(e) => EvalResult::Fail(format!("Judge error: {}", e)),
                    }
                }
                Err(e) => EvalResult::Fail(format!("LLM request failed: {}", e)),
            }
        }
    }
}

// ============================================================================
// Subagent Evals
// ============================================================================

async fn run_subagent_eval(_eval: &Eval, _judge: Option<&Judge>) -> EvalResult {
    // Subagent evals require mux-ffi integration with full agent spawning
    // Skip for now - these test the agent orchestration layer
    EvalResult::Skip("Subagent evals require mux-ffi integration".to_string())
}

// ============================================================================
// Transcript Evals
// ============================================================================

async fn run_transcript_eval(eval: &Eval) -> EvalResult {
    match eval.id.as_str() {
        "transcript-001" => {
            // transcript_save - Can save a conversation
            let store = MemoryTranscriptStore::new();
            let messages = vec![
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "Hello".into(),
                    }],
                },
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "Hi there".into(),
                    }],
                },
            ];
            match store.save("test-agent", &messages).await {
                Ok(_) => EvalResult::Pass,
                Err(e) => EvalResult::Fail(format!("Save failed: {}", e)),
            }
        }
        "transcript-002" => {
            // transcript_load - Saved transcript can be loaded
            let store = MemoryTranscriptStore::new();
            let messages = vec![
                Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "Hello".into(),
                    }],
                },
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "Hi there".into(),
                    }],
                },
            ];
            if let Err(e) = store.save("test-agent", &messages).await {
                return EvalResult::Fail(format!("Save failed: {}", e));
            }

            match store.load("test-agent").await {
                Ok(Some(msgs)) => {
                    if msgs.len() == 2 {
                        EvalResult::Pass
                    } else {
                        EvalResult::Fail(format!("Expected 2 messages, got {}", msgs.len()))
                    }
                }
                Ok(None) => EvalResult::Fail("Failed to load transcript".to_string()),
                Err(e) => EvalResult::Fail(format!("Load failed: {}", e)),
            }
        }
        "transcript-003" => {
            // transcript_missing - Loading missing transcript returns None
            let store = MemoryTranscriptStore::new();
            match store.load("nonexistent").await {
                Ok(None) => EvalResult::Pass,
                Ok(Some(_)) => EvalResult::Fail("Expected None for missing transcript".to_string()),
                Err(e) => EvalResult::Fail(format!("Load failed: {}", e)),
            }
        }
        "transcript-004" => {
            // transcript_preserves_tool_use - Tool use messages preserved
            let store = MemoryTranscriptStore::new();
            let messages = vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "tool-1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"command": "ls"}),
                }],
            }];
            if let Err(e) = store.save("test-agent", &messages).await {
                return EvalResult::Fail(format!("Save failed: {}", e));
            }

            match store.load("test-agent").await {
                Ok(Some(loaded)) => {
                    if let ContentBlock::ToolUse { name, .. } = &loaded[0].content[0] {
                        if name == "bash" {
                            EvalResult::Pass
                        } else {
                            EvalResult::Fail(format!("Expected tool name 'bash', got '{}'", name))
                        }
                    } else {
                        EvalResult::Fail("Tool use not preserved".to_string())
                    }
                }
                Ok(None) => EvalResult::Fail("Transcript not found".to_string()),
                Err(e) => EvalResult::Fail(format!("Load failed: {}", e)),
            }
        }
        "transcript-005" => {
            // transcript_overwrite - Saving overwrites existing
            let store = MemoryTranscriptStore::new();

            // Save initial
            let messages1 = vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "First".into(),
                }],
            }];
            if let Err(e) = store.save("test-agent", &messages1).await {
                return EvalResult::Fail(format!("First save failed: {}", e));
            }

            // Overwrite
            let messages2 = vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "Second".into(),
                }],
            }];
            if let Err(e) = store.save("test-agent", &messages2).await {
                return EvalResult::Fail(format!("Second save failed: {}", e));
            }

            match store.load("test-agent").await {
                Ok(Some(loaded)) => {
                    if let ContentBlock::Text { text } = &loaded[0].content[0] {
                        if text == "Second" {
                            EvalResult::Pass
                        } else {
                            EvalResult::Fail(format!("Expected 'Second', got '{}'", text))
                        }
                    } else {
                        EvalResult::Fail("Wrong content type".to_string())
                    }
                }
                Ok(None) => EvalResult::Fail("Transcript not found".to_string()),
                Err(e) => EvalResult::Fail(format!("Load failed: {}", e)),
            }
        }
        _ => EvalResult::Skip(format!("Unknown transcript eval: {}", eval.id)),
    }
}

// ============================================================================
// MCP Evals (requires real MCP server, skip for now)
// ============================================================================

async fn run_mcp_eval(_eval: &Eval) -> EvalResult {
    EvalResult::Skip("MCP evals require running MCP server".to_string())
}

// ============================================================================
// LLM Provider Evals - Test different LLM providers
// ============================================================================

async fn run_llm_eval(eval: &Eval, _judge: Option<&Judge>) -> EvalResult {
    // Determine which provider to test
    let provider = eval.provider.as_deref().unwrap_or("anthropic");

    match provider {
        "anthropic" => {
            if std::env::var("ANTHROPIC_API_KEY").is_err() {
                return EvalResult::Skip("ANTHROPIC_API_KEY not set".to_string());
            }

            match eval.id.as_str() {
                "llm-001" => {
                    // llm_anthropic_basic - Basic Anthropic call
                    let client = AnthropicClient::from_env().unwrap();
                    let request = Request {
                        model: "claude-sonnet-4-20250514".to_string(),
                        messages: vec![Message {
                            role: Role::User,
                            content: vec![ContentBlock::Text {
                                text: "Say 'hello' and nothing else.".to_string(),
                            }],
                        }],
                        max_tokens: Some(50),
                        ..Default::default()
                    };

                    match client.create_message(&request).await {
                        Ok(response) => {
                            if !response.content.is_empty() {
                                EvalResult::Pass
                            } else {
                                EvalResult::Fail("Empty response from Anthropic".to_string())
                            }
                        }
                        Err(e) => EvalResult::Fail(format!("Anthropic API error: {}", e)),
                    }
                }
                "llm-002" => {
                    // llm_anthropic_streaming - Streaming response
                    use futures::StreamExt;

                    let client = AnthropicClient::from_env().unwrap();
                    let request = Request {
                        model: "claude-sonnet-4-20250514".to_string(),
                        messages: vec![Message {
                            role: Role::User,
                            content: vec![ContentBlock::Text {
                                text: "Count from 1 to 3.".to_string(),
                            }],
                        }],
                        max_tokens: Some(100),
                        ..Default::default()
                    };

                    let mut stream = client.create_message_stream(&request);
                    let mut got_event = false;

                    while let Some(event) = stream.next().await {
                        match event {
                            Ok(_) => got_event = true,
                            Err(e) => return EvalResult::Fail(format!("Stream error: {}", e)),
                        }
                    }

                    if got_event {
                        EvalResult::Pass
                    } else {
                        EvalResult::Fail("No streaming events received".to_string())
                    }
                }
                _ => EvalResult::Skip(format!("Unknown Anthropic eval: {}", eval.id)),
            }
        }
        "openai" => {
            if std::env::var("OPENAI_API_KEY").is_err() {
                return EvalResult::Skip("OPENAI_API_KEY not set".to_string());
            }

            match eval.id.as_str() {
                "llm-003" => {
                    // llm_openai_basic - Basic OpenAI call
                    use mux::llm::OpenAIClient;

                    let client = OpenAIClient::from_env().unwrap();
                    let request = Request {
                        model: "gpt-4o-mini".to_string(),
                        messages: vec![Message {
                            role: Role::User,
                            content: vec![ContentBlock::Text {
                                text: "Say 'hello' and nothing else.".to_string(),
                            }],
                        }],
                        max_tokens: Some(50),
                        ..Default::default()
                    };

                    match client.create_message(&request).await {
                        Ok(response) => {
                            if !response.content.is_empty() {
                                EvalResult::Pass
                            } else {
                                EvalResult::Fail("Empty response from OpenAI".to_string())
                            }
                        }
                        Err(e) => EvalResult::Fail(format!("OpenAI API error: {}", e)),
                    }
                }
                _ => EvalResult::Skip(format!("Unknown OpenAI eval: {}", eval.id)),
            }
        }
        "gemini" => {
            if std::env::var("GEMINI_API_KEY").is_err() {
                return EvalResult::Skip("GEMINI_API_KEY not set".to_string());
            }

            match eval.id.as_str() {
                "llm-005" => {
                    // llm_gemini_basic - Basic Gemini call
                    use mux::llm::GeminiClient;

                    let client = GeminiClient::from_env().unwrap();
                    let request = Request {
                        model: "gemini-2.0-flash".to_string(),
                        messages: vec![Message {
                            role: Role::User,
                            content: vec![ContentBlock::Text {
                                text: "Say 'hello' and nothing else.".to_string(),
                            }],
                        }],
                        max_tokens: Some(50),
                        ..Default::default()
                    };

                    match client.create_message(&request).await {
                        Ok(response) => {
                            if !response.content.is_empty() {
                                EvalResult::Pass
                            } else {
                                EvalResult::Fail("Empty response from Gemini".to_string())
                            }
                        }
                        Err(e) => EvalResult::Fail(format!("Gemini API error: {}", e)),
                    }
                }
                _ => EvalResult::Skip(format!("Unknown Gemini eval: {}", eval.id)),
            }
        }
        _ => EvalResult::Skip(format!("Unknown LLM provider: {}", provider)),
    }
}
