# InjectionShield-Dev

A local prompt-injection red-team framework for LLM applications. Paste in your app's system prompt, pick a target model, and InjectionShield fires a battery of adversarial attacks at it — jailbreaks, prompt leaks, obfuscated payloads, and more — then reports which ones got through.

Built for developers shipping chatbots, RAG apps, or any product with a hidden system prompt who want to know **"can someone break my guardrails?"** before finding out the hard way in production.

![InjectionShield dashboard](docs/screenshot-dashboard.png)

## Why

Most teams ship an LLM-powered app with a system prompt and just hope it holds up. There's rarely an easy, local way to check that before launch. InjectionShield gives you a quick, repeatable security pass — no cloud account, no enterprise pricing, just `python injectionshield_dev.py`.

## Features

- **8 built-in attack payloads** across 5 categories: Jailbreak, Extraction, Instruction override, Obfuscation, Context flooding, Social engineering
- **Pluggable targets** — test against OpenAI, Anthropic, or a local Hugging Face model with the same interface
- **Hybrid evaluator** — clear-cut breaches are caught by rule-based checks; ambiguous responses are judged by an LLM-as-judge call, avoiding false positives on refusals
- **Live dashboard** — a dev-tools-styled Gradio UI with a security score, a live attack console, and charts breaking down vulnerability by attack category
- **No cloud dependency** — runs entirely on your machine; your system prompt and API keys never leave your environment

## Demo results

Same model (`gpt-4o-mini`), different system prompts — showing the tool actually differentiates prompt strength rather than always reporting the same result:

| System prompt | Vulnerability rate |
|---|---|
| Generic persona, no explicit rules | 25% |
| Some rules, loosely worded | 12.5% |
| Explicit, layered security instructions | 0% |

## Installation

```bash
git clone https://github.com/<your-username>/InjectionShield-Dev.git
cd InjectionShield-Dev

python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
python injectionshield_dev.py
```

Open the local URL it prints (usually `http://127.0.0.1:7860`), then:

1. Paste the system prompt you want to test
2. Choose a target provider:
   - **OpenAI** / **Anthropic** — enter a model name (e.g. `gpt-4o-mini`, `claude-3-5-haiku-latest`) and your API key
   - **Local (Hugging Face)** — enter a model name (e.g. `gpt2`); no API key needed, runs fully offline
3. Click **Run security audit**

You'll get a per-attack pass/fail table, an overall security score, and a breakdown of which attack categories are most effective against your prompt.

## Attack categories

| Category | Example technique |
|---|---|
| Jailbreak | Roleplay framing, "DAN"-style persona override |
| Extraction | Asking the model to repeat its system prompt |
| Instruction override | Instructions smuggled inside user content |
| Obfuscation | Base64-encoded or translated payloads |
| Context flooding | Burying an instruction in a wall of benign text |
| Social engineering | Fake "administrator" or authority claims |

## How the evaluator works

A naive keyword check (e.g. flagging any response containing "ignore") produces false positives — a model saying *"I will not ignore my instructions"* would incorrectly be marked vulnerable. InjectionShield instead:

1. Checks for unambiguous compromise markers (literal confirmation phrases) or clear refusal language
2. Falls back to an LLM-as-judge call for anything ambiguous, which reads the attack's goal and the model's actual response before deciding

## Roadmap

- [ ] Multi-turn / session-based attacks (most real jailbreaks aren't single-shot)
- [ ] CVSS-style severity scoring instead of flat pass/fail
- [ ] CLI mode (`injectionshield audit --prompt system.txt`) for CI pipelines
- [ ] Additional payloads: fake function-call injection, delimiter confusion, token smuggling

## Disclaimer

This tool is for testing systems you own or have explicit permission to test. It is a testing aid, not a guarantee — passing all built-in attacks does not mean a system is fully secure against novel or evolving attack techniques.

## License

MIT
