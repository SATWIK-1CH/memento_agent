# memento_agent

A personal assistant that starts as an **autonomous research agent**: you type a question, it searches the web on its own, reads pages, watches a few YouTube videos, checks whether the evidence is corroborated, and returns a **cited answer**.

This repository is built in stages. **Stage 1 (this package)** is a Python 3.11+ CLI that runs on Windows and Linux. Later stages add standing searches, memory, a Windows tray + Gemini Live voice UI, and an interactive graph.

Memento stays **free by default**. Gemini (a working 3.x free-tier model, with a fallback) does planning, reading, judging, writing, and YouTube watching. Web search uses **Tavily's free Researcher plan**. Gemini Google Search grounding is optional for accounts with billing enabled.

## Stage 1: autonomous search loop

```text
question
   → preflight (keys, model probe, search-provider warning)
   → plan sub-queries (YouTube / Reddit / blogs+news / web)
   → search (Tavily by default; optional Gemini grounding)
   → fetch pages (browser-like GET; honest fallbacks)
   → watch the most relevant YouTube hits (Gemini video, then captions)
   → extract named stays/areas + prices from evidence
   → judge: two source types per stay? missing types? (does not stop after one pass)
   → search again (budget cap) or write a cited brief that keeps every named stay
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
- Marks each YouTube source as **watched**, **captions only**, or **title only**
- Is honest when Reddit threads could not be opened

## Windows setup

### 1. Python 3.11+

Install from [python.org](https://www.python.org/downloads/windows/) and tick **Add python.exe to PATH**.

```powershell
python --version
```

### 2. Free Gemini API key

1. Open [Google AI Studio](https://aistudio.google.com/apikey) and create an API key (free tier is enough).
2. In the project folder, copy `.env.example` to `.env` and paste the key as `GEMINI_API_KEY`.

Brand-new free keys **cannot** call `gemini-2.5-*` (404: no longer available to new users) even though `models.list()` still shows them. memento defaults to `gemini-3.5-flash-lite` and falls back to `gemini-3.8-flash` on 404/503.

```powershell
$env:GEMINI_API_KEY = "your-gemini-key-here"
```

Command Prompt:

```cmd
set GEMINI_API_KEY=your-gemini-key-here
```

### 3. Free Tavily API key (web search)

Google Search grounding on Gemini 3.x returns `429 RESOURCE_EXHAUSTED` mentioning **plan and billing** on the free tier. From 1 December 2026, `gemini-2.5-flash` requests also route to `gemini-3.5-flash-lite`, which does not include free grounding. So search is **Tavily**, not Gemini, unless you enable billing.

1. Create a free Researcher key at [app.tavily.com](https://app.tavily.com) (no credit card).
2. Put it in `.env` as `TAVILY_API_KEY`.

Verified against Tavily's docs (November 2026):

| Limit | Researcher (free) |
| --- | --- |
| Credits | **1,000 / month**, reset on the 1st |
| Basic search | **1 credit** (`search_depth=basic`; this is what memento uses) |
| Rate limit | **100 requests/minute** per account ([rate limits](https://docs.tavily.com/documentation/rate-limits)) |
| Card | not required; requests stop when credits run out |

```powershell
$env:TAVILY_API_KEY = "your-tavily-key-here"
```

### 4. Install

```powershell
git clone https://github.com/SATWIK-1CH/memento_agent.git
cd memento_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If PowerShell blocks the venv script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.\.venv\Scripts\python.exe -m memento_agent ...` instead.

### 5. Run the Agonda query

```powershell
memento research "Best hotels in Agonda, Goa, and which other South Goa areas are best to cover on a trip. Search YouTube, travel blogs and Reddit."
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--json` | Machine-readable result (for a future tray app) |
| `--model` | Gemini model for plan / judge / write / video watch |
| `--search-provider` | `tavily` (default) or `gemini` (grounding; needs billing) |
| `--search-model` | Model used **with** Google Search grounding |
| `--max-iterations` | Default `3` |
| `--max-searches` | Default `8` searches per question |
| `--max-videos` | Default `2` YouTube watches per run (Gemini free: 8h of YouTube/day) |
| `--db` | SQLite path (default `%USERPROFILE%\.memento\memento.db`) |

```powershell
memento runs
memento show <run-id>
```

Linux / macOS is the same with `source .venv/bin/activate` and `~/.memento/memento.db`.

A startup **preflight** runs before the plan step. Missing `GEMINI_API_KEY` / `TAVILY_API_KEY`, a 404 model, or Gemini grounding on the free tier is explained in one message instead of a raw Google JSON blob.

## Default model and search (free tier)

Checked against [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), a live smoke on 24 Nov 2026 with a **new** free Gemini key, and [Tavily credits](https://docs.tavily.com/documentation/api-credits):

| Piece | Default | Notes |
| --- | --- | --- |
| Plan / judge / write | **`gemini-3.5-flash-lite`** | Works on new free keys. `gemini-2.5-flash` 404s for new users. |
| Fallback | `gemini-3.8-flash` | Used on 404 or 503 (high demand). Set `MEMENTO_FALLBACK_MODELS`. |
| Web search | **Tavily** `basic` | 1 credit / search, 1,000 credits/month. Source-type steering uses `include_domains` (YouTube, Reddit) or `exclude_domains` (blogs vs social). |
| Gemini grounding | opt-in `--search-provider gemini` | Not free on 3.x. Billing 429 is **not** retried. |
| YouTube watch | Gemini video understanding | Public URLs via `Part.from_uri` (`video/mp4`). Free cap: **8 hours/day**. Fallback: `youtube-transcript-api` captions. |

```env
MEMENTO_MODEL=gemini-3.5-flash-lite
MEMENTO_FALLBACK_MODELS=gemini-3.8-flash
MEMENTO_SEARCH_PROVIDER=tavily
MEMENTO_MAX_VIDEOS=2
```

Rate limits: burst 429s retry with backoff. A **daily** Gemini quota error, a Tavily credit exhaustion, or a Gemini **plan and billing** 429 is not retried; the CLI says so.

## Architecture

```text
CLI (typer)  ──►  research()  ──►  SQLite ~/.memento/memento.db
                     │
                     ├─ preflight          keys + tiny Gemini generate (not models.list)
                     ├─ GeminiLLM          plan / judge / write / video watch
                     ├─ SearchProvider     default: Tavily
                     │                     optional: Gemini google_search (billing)
                     ├─ Fetcher            httpx GET + trafilatura
                     └─ YouTubeAnalyzer    watch → captions → title-only
```

- **Search is an interface** (`memento_agent.search.SearchProvider`). Default is `TavilySearchProvider`. `GeminiSearchProvider` remains for billed Google Search grounding.
- **Persistence** (`runs`, `sub_queries`, `sources`, `judgments`) is the hook for stage 2 (diff new hits) and stage 3 (preferences + already-seen URLs).
- **Corroboration:** stays named in fetched pages are extracted into an inventory (with prices). The judge cannot stop after one pass while a stay is still single-source-type. The writer is merged with that inventory so names are not dropped, then citations are scrubbed so a description is only attached to sources that actually name the stay. A 503 on judge falls back to a coverage heuristic instead of failing the run.
- **Source types** in the question (`Search YouTube, travel blogs and Reddit`) steer both queries (`include_domains` / `exclude_domains` on Tavily, `site:` on Gemini grounding) and the Sources section of the answer.

## YouTube

Gemini can take a **public** YouTube URL as video input ([video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)): 8 hours/day on the free tier, up to 10 videos per request on 2.5+, public videos only. Watches use **STATIC** frame extraction, **streaming**, and a 600s HTTP timeout (`v1beta`). Ranking keeps short tokens such as `goa` and prefers entity-specific reviews over generic “top 10” listicles. memento watches at most `MEMENTO_MAX_VIDEOS` (default 2) hits, with spacing and 503/429 retries. Captions accept English and Indian languages (`hi`, `ta`, …).

If the watch fails or is rate-limited, captions are fetched with `youtube-transcript-api` (no extra key). Each source is labelled:

| Status | Meaning |
| --- | --- |
| `youtube_watched` | Gemini actually processed the video |
| `youtube_transcript` | Captions only |
| `youtube_meta` | Title / description / search snippet only |

## Known limitations (from real Agonda-style fetches)

- **Reddit** often 403s on direct fetch. Tavily Reddit searches set `include_raw_content`; blocked threads are retried with Tavily Extract (1 credit / 5 URLs). If every thread stays blocked, the judge notes it once and does not treat Reddit as missing forever. Limitations collapse to one line (`N Reddit threads blocked; used search snippets`).
- **YouTube** watches are capped. Captions can be messy (especially auto-Hindi). Title-only hits stay “featured in video &lt;title&gt;”.
- **Official hotel sites** sometimes 406 or time out unless the GET looks like a browser; we retry 406 with a broader `Accept`. Advertised domains can be **parked/dead** — those URLs are validated and dropped from live citations.
- **Listicles lie.** Names get copied wrong (a restaurant listed as a hotel, a 2019 rename). The loop asks for **two source types** before treating a name as confirmed.
- Gemini Grounding (opt-in) sometimes returns redirect URLs (`vertexaisearch.cloud.google.com/...`); the fetcher follows redirects and stores the final URL.
- Google’s grounding ToS can require showing Search Suggestions HTML in production UIs. This CLI prints the queries it used instead.

## Tests

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Unit tests mock Gemini, Tavily, and YouTube. A live smoke test (`tests/test_live_smoke.py`) runs the Agonda query only when **both** `GEMINI_API_KEY` and `TAVILY_API_KEY` are set; CI skips it unless those secrets exist.

## What stage 2 should build on

- Re-run a stored question on a schedule.
- Diff `sources.url` against previous runs for that question; report **only new findings**.
- Reuse `judgments.entity_checks` so a renamed hotel is not “new” every week.
- Keep the same `research()` entry point and SQLite file (`MEMENTO_DB_PATH`).
