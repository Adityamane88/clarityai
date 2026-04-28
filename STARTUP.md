# ClarityAI — How to run it on Windows

This is the only doc you need to start. Follow it top to bottom, do not skip steps.

## Step 1 — Install the two prerequisites (one-time)

You need Python and Node.js installed before anything else works.

1. **Python 3.11 or 3.12**: https://www.python.org/downloads/
   - During install, **tick the box "Add python.exe to PATH"**. This is the most common reason setup fails on Windows.
2. **Node.js 20 LTS**: https://nodejs.org/

After installing both, open a fresh PowerShell window and verify:

```powershell
python --version
node --version
npm --version
```

You should see versions printed for all three. If any of them says "not recognized", reinstall and tick the PATH option.

## Step 2 — Put your Groq API key in `backend\.env`

Open `backend\.env` in Notepad. Find this line:

```
LLM_API_KEY=PASTE_YOUR_GROQ_KEY_HERE
```

Replace `PASTE_YOUR_GROQ_KEY_HERE` with the key you got from https://console.groq.com (it starts with `gsk_`). Save and close.

That's the only edit you need to make for the app to give real answers.

## Step 3 — Run the app

Double-click `start.bat` in the project root.

The first run takes 2–4 minutes (it sets up a Python virtual environment and installs all dependencies). Subsequent runs take about 5 seconds.

When it's done, two new terminal windows open (backend + frontend) and your browser opens to http://localhost:5173.

## What you should see when it works

- The sidebar shows a green **"LLM: llama-3.3-70b-versatile"** indicator (not yellow "LLM not connected").
- The right panel shows **2 sample documents** already loaded (auto-seeded on first run).
- Type a question in the box and press Enter. The answer streams in, with citation chips like `[S1]` linking to the source.

## Stopping the app

Close the two terminal windows that opened (titled "ClarityAI backend" and "ClarityAI frontend").

## Common problems and fixes

**"python is not recognized"** → Python isn't on PATH. Reinstall Python and tick "Add python.exe to PATH".

**Sidebar shows "LLM not connected"** → The `.env` key isn't loaded. Check that `backend\.env` has a real key on the `LLM_API_KEY=` line (no quotes, no spaces), then close the backend terminal window and run `start.bat` again.

**Answers say "The LLM provider rejected the API key (HTTP 401)"** → The key is wrong or expired. Generate a new one at https://console.groq.com/keys and paste it into `backend\.env`.

**Answers say "HTTP 404 ... model llama-3.3-70b-versatile may not exist"** → Groq retired that model name. Open https://console.groq.com/docs/models, copy a current chat model id, and replace `CHAT_MODEL=` in `backend\.env`.

**Port 8000 or 5173 already in use** → Something else is running on that port. Either close the other app or change the port in `start.bat` and `backend\.env` (`CORS_ORIGINS`).

**npm install fails** → Delete `frontend\node_modules` and `frontend\package-lock.json`, then run `start.bat` again.

## Optional upgrades (for stronger answers)

### Web research (let the assistant pull live web sources)

1. Get a free Tavily key: https://app.tavily.com (1000 free searches/month)
2. In `backend\.env`, set:
   ```
   ENABLE_WEB_RESEARCH=true
   TAVILY_API_KEY=tvly-your-key-here
   ```
3. Restart the backend window.

In the chat composer, you'll now see the "Force research" mode actually pull live sources and cite them inline like `[W1]`, `[W2]`.

### Dense retrieval (much better matching against your uploaded docs)

Groq doesn't serve embeddings. The easiest free option is Google Gemini:

1. Get a free key at https://aistudio.google.com/apikey
2. In `backend\.env`:
   ```
   ENABLE_DENSE_RETRIEVAL=true
   EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
   EMBEDDING_API_KEY=your-gemini-key
   EMBEDDING_MODEL=text-embedding-004
   ```
3. Restart the backend, then click "Reindex" in the right panel of the app.

Now retrieval uses a real semantic search instead of just keyword overlap. Answers about your uploaded docs become noticeably sharper.

## What to upload to make it feel like *your* app

The right-side panel has an upload button. It accepts:
- PDFs
- Markdown (`.md`)
- Plain text (`.txt`)
- CSV (`.csv`)
- JSON (`.json`)

Upload your domain documents — runbooks, manuals, notes, policies, reports — and the app starts answering from your material with citations. **This is the single biggest difference between a generic chatbot and one that feels production-grade.**

## How it actually works (one paragraph)

When you send a message, the backend retrieves the top matching chunks from your uploaded knowledge, decides whether to also pull live web sources (based on the question and your "Auto/Off/Force research" setting), runs a safety check, then sends the question + the retrieved evidence + the conversation history to the Groq LLM with a system prompt that tells it to be specific and cite sources. The LLM streams the answer back word-by-word. The whole turn (sources, route used, model output, feedback) is stored in SQLite so you can come back to any session and so you can later export your best conversations as fine-tuning data.

That's it. No "training a model on the internet" — just a strong existing model grounded in your data and the live web when needed.
