# InjectionShield-Dev

A local prompt-injection red-team framework for LLM applications. Paste in your app's system prompt, pick a target model, and InjectionShield fires a battery of adversarial attacks at it — including multi-turn jailbreak attempts — then reports which ones got through, how severe each breach is, and how your prompt compares over time.

Built for developers shipping chatbots, RAG apps, or any product with a hidden system prompt who want to know **"can someone break my guardrails?"** before finding out the hard way in production.

![InjectionShield dashboard](docs/screenshot-dashboard.png)

## Why

Most teams ship an LLM-powered app with a system prompt and just hope it holds up. There's rarely an easy, local way to check that before launch — and most real jailbreaks build up over several conversation turns, not a single message. InjectionShield gives you a repeatable security pass, runnable locally or wired into CI, no cloud account or enterprise pricing required.

## Features

- **14 built-in attacks** — 12 single-shot payloads across 7 categories (Jailbreak, Extraction, Instruction override, Function call spoofing, Obfuscation, Context flooding, Social engineering) plus 2 scripted **multi-turn attacks** that build rapport or escalate a persona over several real back-and-forth exchanges
- **Severity scoring** — every category maps to a Critical/High/Medium/Low rating; the dashboard shows a weighted risk score alongside the flat vulnerability rate, plus the highest severity found
- **Pluggable targets** — test against OpenAI, Anthropic, or a local Hugging Face model with the same interface
- **Hybrid evaluator** — clear-cut breaches are caught by rule-based checks; ambiguous responses are judged by an LLM-as-judge call, avoiding false positives on refusals
- **Exportable reports** — download a full audit as JSON or a formatted PDF, straight from the dashboard
- **Run history** — every run is logged locally; a History tab charts your vulnerability rate over time so you can see whether a prompt change actually helped
- **CLI mode** — run audits headlessly for CI pipelines, with a configurable fail threshold
- **Dev-tools styled dashboard** — a GitHub-dark themed Gradio UI with a live attack console, security score, and category breakdown charts
- **No cloud dependency** — runs entirely on your machine; your system prompt and API keys never leave your environment

## Demo results

Same model (`gpt-4o-mini`), different system prompts — showing the tool actually differentiates prompt strength rather than always reporting the same result:

| System prompt | Vulnerability rate | Weighted risk score |
|---|---|---|
| Generic persona, no explicit rules | 7–25% | up to ~9% |
| Some rules, loosely worded | 12.5–14.3% | ~9% |
| Explicit, layered security instructions | 0% | 0% |

## Installation

```bash
git clone https://github.com/<your-username>/InjectionShield-Dev.git
cd InjectionShield-Dev

python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage — dashboard

```bash
python injectionshield_dev.py
```

Open the local URL it prints (usually `http://127.0.0.1:7860`), then:

1. Paste the system prompt you want to test
2. Choose a target provider:
   - **OpenAI** / **Anthropic** — enter a model name (e.g. `gpt-4o-mini`, `claude-3-5-haiku-latest`) and your API key
   - **Local (Hugging Face)** — enter a model name (e.g. `gpt2`); no API key needed, runs fully offline
3. Click **Run security audit**

You'll get a per-attack pass/fail table with severity ratings, an overall security score, a category breakdown chart, and the option to export a JSON or PDF report. Check the **History** tab to see how vulnerability rate trends across runs.

## Usage — CLI (for CI)

```bash
python injectionshield_dev.py audit \
  --prompt system.txt \
  --provider openai \
  --model gpt-4o-mini \
  --fail-threshold 20
```

- `--prompt` accepts a file path or a literal string
- `--provider` is `openai`, `anthropic`, or `local`
- `--api-key` can be omitted if `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` is set in your environment
- `--json-out path.json` writes the full report to a file
- `--fail-threshold` exits with code `1` if the vulnerability rate exceeds it — useful as a pre-deploy gate

## Attack categories

| Category | Severity | Example technique |
|---|---|---|
| Extraction | Critical | Asking the model to repeat its system prompt |
| Multi-turn escalation | Critical | Gradual roleplay or false-rapport extraction over several turns |
| Instruction override | High | Instructions smuggled inside user content, or fake `<<SYS>>` delimiters |
| Function call spoofing | High | A fake embedded function-call block the model is told to "execute" |
| Jailbreak | High | Roleplay framing, "DAN"-style override, hypothetical framing |
| Obfuscation | Medium | Base64-encoded, translated, or fragment-smuggled payloads |
| Social engineering | Medium | Fake "administrator" or authority claims |
| Context flooding | Low | Burying an instruction in a wall of benign text |

## How the evaluator works

A naive keyword check (e.g. flagging any response containing "ignore") produces false positives — a model saying *"I will not ignore my instructions"* would incorrectly be marked vulnerable. InjectionShield instead:

1. Checks for unambiguous compromise markers or clear refusal language
2. Falls back to an LLM-as-judge call for anything ambiguous, which reads the attack's goal and the model's actual response before deciding

## Reports and history

- **Export JSON** — full machine-readable report: target, timestamps, vulnerability rate, weighted risk score, and every attack's payload/response/status/severity
- **Export PDF** — a formatted summary suitable for sharing outside the dashboard
- **Run history** (`runs_history.json`, local only) — every run is appended automatically, whether triggered from the dashboard or the CLI, and surfaced as a trend chart in the History tab

## Roadmap

- [ ] CVSS-aligned scoring (numeric base scores, not just category labels)
- [ ] Additional multi-turn attack scripts
- [ ] Configurable/custom attack payloads via a config file
- [ ] Compare two runs side-by-side in the dashboard

## Disclaimer

This tool is for testing systems you own or have explicit permission to test. It is a testing aid, not a guarantee — passing all built-in attacks does not mean a system is fully secure against novel or evolving attack techniques.

## License

MIT