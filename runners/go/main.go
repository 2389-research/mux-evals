// ABOUTME: Go eval runner for mux library.
// ABOUTME: Executes language-agnostic eval definitions against the Go implementation.

package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Eval struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Category    string          `json:"category"`
	Provider    string          `json:"provider,omitempty"`
	RequiresKey string          `json:"requires_key,omitempty"`
	Given       json.RawMessage `json:"given"`
	When        json.RawMessage `json:"when"`
	Then        json.RawMessage `json:"then"`
}

type EvalResult int

const (
	Pass EvalResult = iota
	Fail
	Skip
)

type Result struct {
	Status EvalResult
	Reason string
}

// JSON output structures
type JsonEvalResult struct {
	ID       string  `json:"id"`
	Name     string  `json:"name"`
	Category string  `json:"category"`
	Status   string  `json:"status"`
	Reason   *string `json:"reason,omitempty"`
}

type JsonSummary struct {
	Passed  int `json:"passed"`
	Failed  int `json:"failed"`
	Skipped int `json:"skipped"`
	Total   int `json:"total"`
}

type JsonReport struct {
	Runner  string           `json:"runner"`
	Results []JsonEvalResult `json:"results"`
	Summary JsonSummary      `json:"summary"`
}

// ANSI color codes
const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorCyan   = "\033[36m"
	colorBold   = "\033[1m"
	colorDim    = "\033[2m"
)

func main() {
	evalsPath := flag.String("evals", "../../evals", "Path to evals directory or specific .jsonl file")
	category := flag.String("category", "", "Filter by category")
	id := flag.String("id", "", "Filter by specific eval ID")
	verbose := flag.Bool("verbose", false, "Verbose output")
	failuresOnly := flag.Bool("failures-only", false, "Only show failures")
	jsonOutput := flag.Bool("json", false, "Output results as JSON")
	flag.Parse()

	evals, err := loadEvals(*evalsPath, *category, *id)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading evals: %v\n", err)
		os.Exit(1)
	}

	if !*jsonOutput {
		fmt.Fprintf(os.Stderr, "\n%s%sRunning%s %d evals\n\n", colorBold, colorCyan, colorReset, len(evals))
	}

	var passed, failed, skipped int
	var jsonResults []JsonEvalResult

	for _, eval := range evals {
		result := runEval(eval, *verbose)

		var status string
		var reason *string

		switch result.Status {
		case Pass:
			passed++
			status = "pass"
		case Fail:
			failed++
			status = "fail"
			reason = &result.Reason
		case Skip:
			skipped++
			status = "skip"
			reason = &result.Reason
		}

		if *jsonOutput {
			jsonResults = append(jsonResults, JsonEvalResult{
				ID:       eval.ID,
				Name:     eval.Name,
				Category: eval.Category,
				Status:   status,
				Reason:   reason,
			})
		} else {
			switch result.Status {
			case Pass:
				if !*failuresOnly {
					fmt.Printf("%s%sPASS%s %s - %s\n", colorBold, colorGreen, colorReset, eval.ID, eval.Name)
				}
			case Fail:
				fmt.Printf("%s%sFAIL%s %s - %s\n       %s%s%s\n",
					colorBold, colorRed, colorReset, eval.ID, eval.Name,
					colorDim, result.Reason, colorReset)
			case Skip:
				if !*failuresOnly {
					fmt.Printf("%s%sSKIP%s %s - %s\n       %s%s%s\n",
						colorBold, colorYellow, colorReset, eval.ID, eval.Name,
						colorDim, result.Reason, colorReset)
				}
			}
		}
	}

	if *jsonOutput {
		report := JsonReport{
			Runner:  "go",
			Results: jsonResults,
			Summary: JsonSummary{
				Passed:  passed,
				Failed:  failed,
				Skipped: skipped,
				Total:   len(evals),
			},
		}
		output, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(output))
	} else {
		fmt.Printf("\n%sResults%s: %s%d%s passed, ", colorBold, colorReset, colorGreen, passed, colorReset)
		if failed > 0 {
			fmt.Printf("%s%d%s failed, ", colorRed, failed, colorReset)
		} else {
			fmt.Printf("%d failed, ", failed)
		}
		fmt.Printf("%s%d%s skipped\n\n", colorYellow, skipped, colorReset)
	}

	if failed > 0 {
		os.Exit(1)
	}
}

func loadEvals(path, categoryFilter, idFilter string) ([]Eval, error) {
	var evals []Eval

	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("cannot access path %s: %w", path, err)
	}

	var files []string
	if info.IsDir() {
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, err
		}
		for _, entry := range entries {
			if strings.HasSuffix(entry.Name(), ".jsonl") {
				files = append(files, filepath.Join(path, entry.Name()))
			}
		}
	} else {
		files = []string{path}
	}

	for _, filePath := range files {
		f, err := os.Open(filePath)
		if err != nil {
			return nil, fmt.Errorf("failed to open %s: %w", filePath, err)
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		lineNum := 0
		for scanner.Scan() {
			lineNum++
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}

			var eval Eval
			if err := json.Unmarshal([]byte(line), &eval); err != nil {
				return nil, fmt.Errorf("failed to parse line %d in %s: %w", lineNum, filePath, err)
			}

			// Apply filters
			if categoryFilter != "" && eval.Category != categoryFilter {
				continue
			}
			if idFilter != "" && eval.ID != idFilter {
				continue
			}

			evals = append(evals, eval)
		}

		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("error reading %s: %w", filePath, err)
		}
	}

	return evals, nil
}

func runEval(eval Eval, verbose bool) Result {
	// Check for required API keys
	if eval.RequiresKey != "" {
		if os.Getenv(eval.RequiresKey) == "" {
			return Result{Skip, fmt.Sprintf("%s not set", eval.RequiresKey)}
		}
	}

	if verbose {
		fmt.Printf("  given: %s\n", eval.Given)
		fmt.Printf("  when: %s\n", eval.When)
		fmt.Printf("  then: %s\n", eval.Then)
	}

	// Dispatch based on category
	switch eval.Category {
	case "tools":
		return runToolEval(eval)
	case "hooks":
		return runHookEval(eval)
	case "agent":
		return runAgentEval(eval)
	case "subagent":
		return runSubagentEval(eval)
	case "transcript":
		return runTranscriptEval(eval)
	case "mcp":
		return runMCPEval(eval)
	case "llm":
		return runLLMEval(eval)
	default:
		return Result{Skip, fmt.Sprintf("Unknown category: %s", eval.Category)}
	}
}

func runToolEval(eval Eval) Result {
	// TODO: Implement against mux tool registry
	return Result{Skip, "Tool eval implementation pending"}
}

func runHookEval(eval Eval) Result {
	// TODO: Implement against mux hook system
	return Result{Skip, "Hook eval implementation pending"}
}

func runAgentEval(eval Eval) Result {
	// TODO: Implement against mux agent/orchestrator
	return Result{Skip, "Agent eval implementation pending"}
}

func runSubagentEval(eval Eval) Result {
	// TODO: Implement against mux subagent system
	return Result{Skip, "Subagent eval implementation pending"}
}

func runTranscriptEval(eval Eval) Result {
	// TODO: Implement against mux transcript persistence
	return Result{Skip, "Transcript eval implementation pending"}
}

func runMCPEval(eval Eval) Result {
	// TODO: Implement against mux MCP client
	return Result{Skip, "MCP eval implementation pending"}
}

func runLLMEval(eval Eval) Result {
	// TODO: Implement against mux LLM providers
	return Result{Skip, "LLM eval implementation pending"}
}
