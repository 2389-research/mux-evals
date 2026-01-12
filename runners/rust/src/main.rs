// ABOUTME: Rust eval runner for mux-rs library.
// ABOUTME: Executes language-agnostic eval definitions against the Rust implementation.

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

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

#[allow(unused)]
#[derive(Debug)]
enum EvalResult {
    Pass,
    Fail(String),
    Skip(String),
}

fn main() -> Result<()> {
    let args = Args::parse();

    let evals = load_evals(&args.evals, args.category.as_deref(), args.id.as_deref())?;

    println!(
        "\n{} {} evals\n",
        "Running".bold().cyan(),
        evals.len()
    );

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for eval in &evals {
        let result = run_eval(eval, args.verbose);

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

fn run_eval(eval: &Eval, verbose: bool) -> EvalResult {
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
        "tools" => run_tool_eval(eval),
        "hooks" => run_hook_eval(eval),
        "agent" => run_agent_eval(eval),
        "subagent" => run_subagent_eval(eval),
        "transcript" => run_transcript_eval(eval),
        "mcp" => run_mcp_eval(eval),
        "llm" => run_llm_eval(eval),
        _ => EvalResult::Skip(format!("Unknown category: {}", eval.category)),
    }
}

#[allow(unused)]
fn run_tool_eval(_eval: &Eval) -> EvalResult {
    // TODO: Implement against mux-rs tool registry
    // For now, return skip to show the framework works
    EvalResult::Skip("Tool eval implementation pending".to_string())
}

#[allow(unused)]
fn run_hook_eval(_eval: &Eval) -> EvalResult {
    // TODO: Implement against mux-rs hook system
    EvalResult::Skip("Hook eval implementation pending".to_string())
}

#[allow(unused)]
fn run_agent_eval(_eval: &Eval) -> EvalResult {
    // TODO: Implement against mux-rs agent loop
    EvalResult::Skip("Agent eval implementation pending".to_string())
}

#[allow(unused)]
fn run_subagent_eval(_eval: &Eval) -> EvalResult {
    // TODO: Implement against mux-rs subagent system
    EvalResult::Skip("Subagent eval implementation pending".to_string())
}

#[allow(unused)]
fn run_transcript_eval(_eval: &Eval) -> EvalResult {
    // TODO: Implement against mux-rs transcript persistence
    EvalResult::Skip("Transcript eval implementation pending".to_string())
}

#[allow(unused)]
fn run_mcp_eval(_eval: &Eval) -> EvalResult {
    // TODO: Implement against mux-rs MCP client
    EvalResult::Skip("MCP eval implementation pending".to_string())
}

#[allow(unused)]
fn run_llm_eval(_eval: &Eval) -> EvalResult {
    // TODO: Implement against mux-rs LLM providers
    EvalResult::Skip("LLM eval implementation pending".to_string())
}
