# memento_agent

A personal assistant that starts as an **autonomous research agent**: you type a question, it searches the web on its own, reads pages, checks whether the evidence is corroborated, and returns a **cited answer**.

This repository is built in stages. **Stage 1 (this package)** is a Python 3.11+ CLI that runs on Windows and Linux. Later stages add standing searches, memory, a Windows tray + Gemini Live voice UI, and an interactive graph.

## Stage 1: autonomous search loop

```text
question
   → plan sub-queries (YouTube / Reddit / blogs+news / web)
   → search (Gemini Grounding with Google Search)
   → fetch pages (browser-like GET; honest fallbacks)
   → judge: enough corroborated evidence? gaps?
   → search again (budget cap) or write a cited brief
```

The loop is a small plain-Python module (`memento_agent.agent.research`), not a heavyweight agent framework. That keeps sub-queries, sources, and answers explicit in SQLite so stage 2 (standing searches) and stage 3 (memory) can query the same file, and a tray/voice UI can call `research()` without going through the CLI.

### Example

```powershell
memento research "Best hotels in Agonda, Goa, and which other South Goa areas are best to cover on a trip. Search YouTube, travel blogs and Reddit."
```

A good result:

- Lists hotels in **tiers** (budget / mid-range / splurge) with why
- Suggests **other South Goa areas** (Palolem, Patnem, …) with reasons
- Cites every claim, **grouped by source type**
- Is honest when Reddit threads could not be opened or a YouTube citation is title-only

## Windows setup

### 1. Python 3.11+

Install from [python.org](https://www.python.org/downloads/windows/) and tick **Add python.exe to PATH**.

```powershell
python --version
```

### 2. Free Gemini API key

1. Open [Google AI Studio](https://aistudio.google.com/apikey) and create an API key (free tier is enough).
2. In the project folder, copy `.env.example` to `.env` and paste the key:

```env
GEMINI_API_KEY=your-key-here
```

Or set it for the current PowerShell session (do not commit the key):

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

Command Prompt:

```cmd
set GEMINI_API_KEY=your-key-here
```

### 3. Install

```powershell
git clone https://github.com/SATWIK-1CH/memento_agent.git
cd memento_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If PowerShell blocks the venv script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.\.venv\Scripts\python.exe -m memento_agent ...` instead.

### 4. Run

```powershell
memento research "Best hotels in Agonda, Goa, and which other South Goa areas are best to cover on a trip. Search YouTube, travel blogs and Reddit."
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--json` | Machine-readable result (for a future tray app) |
| `--model` | Gemini model for plan / judge / write |
| `--search-model` | Model used **with** Google Search grounding |
| `--max-iterations` | Default `3` |
| `--max-searches` | Default `8` grounded searches per question |
| `--db` | SQLite path (default `%USERPROFILE%\.memento\memento.db`) |

```powershell
memento runs
memento show <run-id>
```

Linux / macOS is the same with `source .venv/bin/activate` and `~/.memento/memento.db`.

## Default model (free tier)

Checked against [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) (November 2026):

| Model | Free tokens | Free Grounding with Google Search |
| --- | --- | --- |
| **`gemini-2.5-flash`** (default) | yes | **yes**, ~**500** grounded requests/day, shared with Flash-Lite |
| `gemini-2.5-flash-lite` | yes | same 500 RPD pool |
| Gemini 3.x / 4.x Flash (e.g. `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-4-hydrogen`) | often yes | **not on the free tier** (paid: 5,000 searches/month then $14/1k, 1,500 RPD cap) |

The default is `gemini-2.5-flash` because it is the strongest **free-tier** model that can actually search. Gemini 2.5 Flash is deprecated in AI Studio and, **from 1 December 2026**, API calls may route to `gemini-3.5-flash-lite` (tokens still free, **search grounding is not**). If search starts failing around then, set:

```env
MEMENTO_SEARCH_MODEL=gemini-2.5-flash-lite
```

`gemini-2.5-flash-lite` is documented as still served for existing API users. Brand-new keys sometimes cannot call 2.5 models at all; if that happens, either enable billing for a Gemini 3 model with search, or plug in another `SearchProvider` later (Tavily/Brave).

To keep **free search** on 2.5 and use a newer model only for writing:

```env
MEMENTO_MODEL=gemini-3.8-flash
MEMENTO_SEARCH_MODEL=gemini-2.5-flash
```

Rate limits: 429s retry with backoff. A **daily** quota error is not retried; the CLI says so and mentions the midnight Pacific reset.

## Architecture

```text
CLI (typer)  ──►  research()  ──►  SQLite ~/.memento/memento.db
                     │
                     ├─ GeminiLLM          plan / judge / write (JSON schemas)
                     ├─ SearchProvider     default: Gemini google_search tool
                     │                     (site: steering for YouTube, Reddit, blogs)
                     └─ Fetcher            httpx GET + trafilatura
```

- **Search is an interface** (`memento_agent.search.SearchProvider`). Stage 1 ships `GeminiSearchProvider`. A Tavily/Brave adapter can implement the same `search(query, source_type=...)` method.
- **Persistence** (`runs`, `sub_queries`, `sources`, `judgments`) is the hook for stage 2 (diff new hits) and stage 3 (preferences + already-seen URLs).
- **Corroboration:** the critic must tick `entity_checks` for named hotels/places. A name that appears in only one listicle is treated as a gap; the next iteration searches for an official/current listing and Reddit mentions. Parked/dead domains are marked `dead` and are not cited.
- **Source types** in the question (`Search YouTube, travel blogs and Reddit`) steer both queries and the Sources section of the answer.

## Known limitations (from real Agonda-style fetches)

- **Reddit** is the weakest full-page source. Search snippets (e.g. r/goatravel) are useful; thread HTML, `old.reddit.com`, and `.json` often time out or hit a login wall. HEAD requests 403; we only GET, with a browser User-Agent. Unopened threads are cited as **search snippet only**.
- **YouTube** is not watched and transcripts are not used (auto-captions, especially Hindi, are messy). Citations are **“featured in video &lt;title&gt;”** from oEmbed / title / description.
- **Official hotel sites** sometimes 406 or time out unless the GET looks like a browser; we retry 406 with a broader `Accept`. Advertised domains can be **parked/dead** — those URLs are validated and dropped from live citations.
- **Listicles lie.** Names get copied wrong (a restaurant listed as a hotel, a 2019 rename). The loop asks for **two source types** before treating a name as confirmed.
- Gemini Grounding sometimes returns redirect URLs (`vertexaisearch.cloud.google.com/...`); the fetcher follows redirects and stores the final URL.
- Google’s grounding ToS can require showing Search Suggestions HTML in production UIs. This CLI prints the queries it used instead.

## Tests

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Unit tests mock Gemini and HTTP. A live smoke test (`tests/test_live_smoke.py`) runs the Agonda query only when `GEMINI_API_KEY` is set; CI skips it unless that secret exists.

## What stage 2 should build on

- Re-run a stored question on a schedule.
- Diff `sources.url` against previous runs for that question; report **only new findings**.
- Reuse `judgments.entity_checks` so a renamed hotel is not “new” every week.
- Keep the same `research()` entry point and SQLite file (`MEMENTO_DB_PATH`).
