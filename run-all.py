#!/usr/bin/env python3
# ABOUTME: Meta-runner that executes both Rust and Go eval runners.
# ABOUTME: Generates comparison table as HTML and console output.

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class EvalResult:
    id: str
    name: str
    category: str
    status: str
    reason: Optional[str] = None


@dataclass
class RunnerReport:
    runner: str
    results: list[EvalResult]
    passed: int
    failed: int
    skipped: int
    total: int
    error: Optional[str] = None


def run_runner(name: str, cmd: list[str], cwd: Path) -> RunnerReport:
    """Run a single eval runner and parse its JSON output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Parse JSON from stdout
        data = json.loads(result.stdout)

        results = [
            EvalResult(
                id=r["id"],
                name=r["name"],
                category=r["category"],
                status=r["status"],
                reason=r.get("reason"),
            )
            for r in data["results"]
        ]

        return RunnerReport(
            runner=name,
            results=results,
            passed=data["summary"]["passed"],
            failed=data["summary"]["failed"],
            skipped=data["summary"]["skipped"],
            total=data["summary"]["total"],
        )
    except subprocess.TimeoutExpired:
        return RunnerReport(
            runner=name,
            results=[],
            passed=0,
            failed=0,
            skipped=0,
            total=0,
            error="Timeout after 5 minutes",
        )
    except json.JSONDecodeError as e:
        return RunnerReport(
            runner=name,
            results=[],
            passed=0,
            failed=0,
            skipped=0,
            total=0,
            error=f"Invalid JSON output: {e}",
        )
    except FileNotFoundError:
        return RunnerReport(
            runner=name,
            results=[],
            passed=0,
            failed=0,
            skipped=0,
            total=0,
            error="Runner not found",
        )
    except Exception as e:
        return RunnerReport(
            runner=name,
            results=[],
            passed=0,
            failed=0,
            skipped=0,
            total=0,
            error=str(e),
        )


def status_symbol(status: str) -> str:
    """Return emoji/symbol for status."""
    return {"pass": "✅", "fail": "❌", "skip": "⏭️"}.get(status, "❓")


def status_color(status: str) -> str:
    """Return ANSI color code for status."""
    return {
        "pass": "\033[32m",  # Green
        "fail": "\033[31m",  # Red
        "skip": "\033[33m",  # Yellow
    }.get(status, "")


def print_console_table(reports: list[RunnerReport]):
    """Print comparison table to console."""
    reset = "\033[0m"
    bold = "\033[1m"

    # Collect all eval IDs
    all_ids = set()
    for report in reports:
        for result in report.results:
            all_ids.add((result.id, result.name, result.category))

    # Sort by category then ID
    sorted_evals = sorted(all_ids, key=lambda x: (x[2], x[0]))

    # Build result lookup
    results_by_id = {}
    for report in reports:
        for result in report.results:
            results_by_id[(report.runner, result.id)] = result

    # Print header
    runners = [r.runner for r in reports]
    header = f"{'Eval ID':<20} {'Name':<35} "
    for runner in runners:
        header += f"{runner:^8} "
    print(f"\n{bold}{header}{reset}")
    print("=" * len(header))

    # Print rows grouped by category
    current_category = None
    for eval_id, name, category in sorted_evals:
        if category != current_category:
            current_category = category
            print(f"\n{bold}[{category}]{reset}")

        row = f"{eval_id:<20} {name[:33]:<35} "
        for runner in runners:
            result = results_by_id.get((runner, eval_id))
            if result:
                color = status_color(result.status)
                symbol = status_symbol(result.status)
                row += f"{color}{symbol:^8}{reset} "
            else:
                row += f"{'—':^8} "
        print(row)

    # Print summary
    print("\n" + "=" * len(header))
    print(f"{bold}Summary:{reset}")
    for report in reports:
        if report.error:
            print(f"  {report.runner}: ERROR - {report.error}")
        else:
            print(
                f"  {report.runner}: "
                f"\033[32m{report.passed} passed{reset}, "
                f"\033[31m{report.failed} failed{reset}, "
                f"\033[33m{report.skipped} skipped{reset}"
            )
    print()


def generate_html(reports: list[RunnerReport], output_path: Path):
    """Generate HTML comparison table."""
    # Collect all eval IDs
    all_ids = set()
    for report in reports:
        for result in report.results:
            all_ids.add((result.id, result.name, result.category))

    sorted_evals = sorted(all_ids, key=lambda x: (x[2], x[0]))

    results_by_id = {}
    for report in reports:
        for result in report.results:
            results_by_id[(report.runner, result.id)] = result

    runners = [r.runner for r in reports]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mux Eval Results</title>
    <style>
        :root {{
            --pass: #22c55e;
            --fail: #ef4444;
            --skip: #eab308;
            --bg: #0f172a;
            --bg-alt: #1e293b;
            --text: #e2e8f0;
            --border: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            margin-bottom: 10px;
        }}
        .timestamp {{
            color: #64748b;
            font-size: 0.875rem;
            margin-bottom: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.875rem;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--bg-alt);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: var(--bg-alt);
        }}
        .category-header {{
            background: var(--bg-alt);
            font-weight: 600;
            color: #94a3b8;
        }}
        .status {{
            text-align: center;
            font-size: 1.25rem;
        }}
        .pass {{ color: var(--pass); }}
        .fail {{ color: var(--fail); }}
        .skip {{ color: var(--skip); }}
        .summary {{
            margin-top: 20px;
            padding: 15px;
            background: var(--bg-alt);
            border-radius: 8px;
        }}
        .summary h2 {{
            margin-top: 0;
        }}
        .summary-item {{
            margin: 5px 0;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            font-size: 0.875rem;
        }}
        .legend span {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
    </style>
</head>
<body>
    <h1>Mux Eval Comparison</h1>
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

    <div class="legend">
        <span><span class="pass">✅</span> Pass</span>
        <span><span class="fail">❌</span> Fail</span>
        <span><span class="skip">⏭️</span> Skip</span>
    </div>

    <table>
        <thead>
            <tr>
                <th>Eval ID</th>
                <th>Name</th>
"""

    for runner in runners:
        html += f"                <th style='text-align:center'>{runner.title()}</th>\n"

    html += """            </tr>
        </thead>
        <tbody>
"""

    current_category = None
    for eval_id, name, category in sorted_evals:
        if category != current_category:
            current_category = category
            html += f"""            <tr class="category-header">
                <td colspan="{2 + len(runners)}">{category}</td>
            </tr>
"""

        html += f"""            <tr>
                <td>{eval_id}</td>
                <td>{name}</td>
"""
        for runner in runners:
            result = results_by_id.get((runner, eval_id))
            if result:
                symbol = {"pass": "✅", "fail": "❌", "skip": "⏭️"}.get(result.status, "❓")
                css_class = result.status
                title = result.reason or ""
                html += f'                <td class="status {css_class}" title="{title}">{symbol}</td>\n'
            else:
                html += '                <td class="status">—</td>\n'

        html += "            </tr>\n"

    html += """        </tbody>
    </table>

    <div class="summary">
        <h2>Summary</h2>
"""

    for report in reports:
        if report.error:
            html += f'        <div class="summary-item"><strong>{report.runner.title()}:</strong> ERROR - {report.error}</div>\n'
        else:
            html += f"""        <div class="summary-item">
            <strong>{report.runner.title()}:</strong>
            <span class="pass">{report.passed} passed</span>,
            <span class="fail">{report.failed} failed</span>,
            <span class="skip">{report.skipped} skipped</span>
        </div>
"""

    html += """    </div>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"HTML report written to: {output_path}")


def main():
    script_dir = Path(__file__).parent.resolve()

    print("Running eval runners...\n")

    # Build Rust runner first
    print("Building Rust runner...")
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=script_dir / "runners" / "rust",
        capture_output=True,
    )

    # Build Go runner
    print("Building Go runner...")
    subprocess.run(
        ["go", "build", "-o", "mux-eval-runner", "."],
        cwd=script_dir / "runners" / "go",
        capture_output=True,
    )

    reports = []

    # Run Rust
    print("Running Rust evals...")
    rust_report = run_runner(
        "rust",
        ["cargo", "run", "--release", "--", "--json"],
        script_dir / "runners" / "rust",
    )
    reports.append(rust_report)

    # Run Go
    print("Running Go evals...")
    go_report = run_runner(
        "go",
        ["./mux-eval-runner", "-json"],
        script_dir / "runners" / "go",
    )
    reports.append(go_report)

    # Print console table
    print_console_table(reports)

    # Generate HTML
    html_path = script_dir / "results.html"
    generate_html(reports, html_path)


if __name__ == "__main__":
    main()
