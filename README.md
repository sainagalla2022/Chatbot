# Local Chatbot: ask questions about your documents, fully offline

A chatbot that runs **entirely on your own computer**. Upload a PDF or TXT file (or paste some text), then ask questions about it in plain English. There are no cloud AI services, no API keys, and no GPU needed. Your documents and questions never leave your machine.

**What you can do**

- **Upload a PDF or TXT file**, or **paste text**, and ask questions about it. Answers come with their sources (file and page).
- **Never a dead end:** if the app cannot find a direct answer, it shows the closest parts of your document (with page numbers) instead of just saying "I don't know". Spelling mistakes are forgiven.
- **Ask "what is this document about?"**, "whose resume is this?" or "what is the email?" and get exact details copied from the document.
- **Keep documents separate:** every browser session has its own private set of files.
- **Ask a SQLite database in plain English** (through the API; read-only, so it can never change your data).

**Stack:** FastAPI, LangChain, TinyLlama (through Ollama), FAISS, `sentence-transformers/all-MiniLM-L6-v2`, pypdf, SQLAlchemy + SQLite, pytest.

**Contents:** [Quick start](#quick-start) · [Full setup guide](#full-setup-guide-step-by-step) · [Using the chat page](#using-the-chat-page) · [Use a better model](#use-a-better-model-optional) · [Configuration](#configuration) · [Troubleshooting](#troubleshooting) · [Project layout](#project-layout) · [Architecture](#architecture) · [API examples](#api-examples) · [Safety design](#safety-design) · [Known limitations](#known-limitations) · [How to test](#how-to-test)

> **Privacy note.** The only thing downloaded from the internet, besides the tools you install, is a small embedding model (about 90 MB) from Hugging Face the first time the server starts. It is cached afterwards. Once it is cached you can set `HF_HUB_OFFLINE=1` to guarantee no further network access.

---

## Quick start

If you already have **Git, Python 3.12 (or [`uv`](https://docs.astral.sh/uv/)) and [Ollama](https://ollama.com)** installed, these commands are everything you need. New to this? Use the [full setup guide](#full-setup-guide-step-by-step) below.

```bash
git clone <repository-url> Chatbot && cd Chatbot

uv venv --python 3.12 venv && source venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch FIRST
uv pip install -r requirements.txt

ollama pull tinyllama                                                    # the language model (~640 MB)

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/>, upload a file, and ask a question.

---

## Full setup guide (step by step)

These steps were written for **WSL2 with Ubuntu**, and they work the same on Linux and macOS (the install commands for Ollama differ, see step 2). Native Windows has not been tested; use WSL2.

### Step 0. What you need

| Requirement | Details |
|---|---|
| Operating system | Windows 11/10 with **WSL2 + Ubuntu**, or Linux, or macOS |
| Python | **3.12** (the version this project was built and tested on) |
| Git | to clone the project |
| [Ollama](https://ollama.com) | runs the language model locally |
| RAM | about **5 GB free** (TinyLlama needs roughly 1 GB inside Ollama and the app about 0.5 GB; a 3B model needs about 2.5 GB) |
| Disk | about **3 GB free**: about 1.3 GB for the Python packages, 640 MB for TinyLlama, 90 MB for the embedding model (add 1.9 GB if you also use `qwen2.5:3b`) |
| Internet | needed during setup only (downloads); the app runs offline afterwards |

Check what you already have:

```bash
git --version
python3 --version        # 3.12.x if you plan to use plain venv instead of uv
```

**WSL2 not set up yet?** In Windows PowerShell (as administrator) run `wsl --install`, restart, then open the "Ubuntu" app. Do all the steps below inside that Ubuntu terminal.

### Step 1. Get the code

```bash
git clone <repository-url> Chatbot
cd Chatbot
```

Replace `<repository-url>` with the address of this repository. All the following commands are run from inside the `Chatbot` folder.

### Step 2. Install Ollama and download the model

Ollama is the program that runs the language model on your computer.

**Ubuntu / WSL2:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

(The installer may ask for your password because it installs a system service.)

**macOS:** `brew install ollama`, or download the app from <https://ollama.com/download>.

**Windows (optional alternative):** you can install Ollama for Windows and keep everything else in WSL2. The app reaches it at `http://127.0.0.1:11434`, which is the default, so nothing else needs changing.

Make sure Ollama is running, then download the model:

```bash
ollama serve          # only if Ollama is not already running; leave this terminal open
```

Open a **second terminal** for the next commands (or skip `ollama serve` if the Ollama service is already running):

```bash
ollama pull tinyllama                        # one-time download, about 640 MB
curl http://127.0.0.1:11434/api/tags         # the answer should mention "tinyllama"
```

If the installer set Ollama up as a service, you can start it with `sudo systemctl start ollama` instead of `ollama serve`. In WSL2 without systemd, use `ollama serve`.

### Step 3. Create the Python environment and install packages

This project uses [`uv`](https://docs.astral.sh/uv/), a fast Python package manager. If you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal afterwards (or run `source ~/.bashrc`) so the `uv` command is found.

Then, from inside the `Chatbot` folder:

```bash
uv venv --python 3.12 venv         # creates the virtual environment in ./venv (downloads Python 3.12 if needed)
source venv/bin/activate           # turn it on; your prompt now starts with (venv)
```

Install **PyTorch for CPU first**, then everything else. The order matters: without it, the installer may download a much larger GPU version of PyTorch that you do not need.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

The second command takes a few minutes and downloads roughly 1 GB.

<details>
<summary>Prefer plain <code>pip</code> instead of <code>uv</code>?</summary>

```bash
python3.12 -m venv venv            # on Ubuntu you may first need: sudo apt install python3.12-venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

</details>

> **Every time you open a new terminal**, run `cd Chatbot && source venv/bin/activate` again before using the project.

### Step 4. (Optional) Create the sample database

Only needed if you want to try the SQL question feature through the API (`POST /sql`). The chat page does not use it.

```bash
python scripts/create_sample_db.py
```

This creates `sample.db` with 6 made-up customers and 20 made-up orders. (`*.db` files are git-ignored, which is why you have to create it yourself.)

### Step 5. Start the app

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete.` in the terminal. The **first start takes longer** (a minute or two) because the embedding model (about 90 MB) is downloaded once; later starts take a few seconds.

Leave this terminal open while you use the app. Stop it any time with **Ctrl+C**.

Now open:

- **Chat page:** <http://127.0.0.1:8000/>
- **API documentation:** <http://127.0.0.1:8000/docs>

On Windows with WSL2 you can open these addresses in your normal Windows browser.

### Step 6. Check that everything works

1. The top right of the chat page should show **"Ollama (tinyllama): online"** with a green dot.
2. Click **Paste text**, paste a few sentences, click **Add text**, and ask a question about them.
3. Optional: run the automated tests (they need neither Ollama nor any download):

   ```bash
   pytest
   ```

4. Optional: try the model from the command line, without the web page:

   ```bash
   python scripts/try_rag.py
   ```

If something does not work, see [Troubleshooting](#troubleshooting).

### Updating later

```bash
cd Chatbot && git pull
source venv/bin/activate
uv pip install -r requirements.txt
```

Then stop the server (Ctrl+C), start it again, and reload the page with Ctrl+F5.

---

## Using the chat page

| Control | What it does |
|---|---|
| **Upload PDF / TXT** | Choose a `.pdf` or `.txt` file (up to 10 MB). Wait for "Uploaded and indexed …". |
| **Paste text** | Opens a box. Paste any text, then click **Add text** (or press Ctrl+Enter). Each paste becomes its own document ("Pasted text 1", "Pasted text 2", …). |
| **Question box** | Type a question (up to 1,000 characters) and press Enter or **Send**. |
| **New session** | Starts fresh: the chat is cleared, and the new session has none of the earlier files. |

**Good questions** use the document's own words:

- "What is the email address?" or "How do I contact the owner?" (the exact email and phone are copied from the text)
- "What is this document about?" or "Whose resume is this?" (type, name and title are copied exactly)
- "What is the hourly rate?" or "Where did he work most recently?"

Under each answer, **Sources** shows which file and page it came from, and **Show matching text** opens the exact passages the answer was based on. Check important answers against them, because a small model can make mistakes.

**If the app can't find a direct answer**, it does not stop at "I don't know". It shows *Related text from your document*: the closest passages, copied from your file with their pages, so you can read the place yourself. If none of your words occur in the document, it says so honestly and shows the closest parts anyway, warning that they may not be related.

**Typos and short questions are fine.** "waht is kuberntes" or just "git" still finds the right pages. The app matches exact words and close spellings (using the words your document really contains), and blends that with its search by meaning.

**Where are my files?** Each upload is copied to `data/sessions/<session id>/uploads/`. The page shows the first 8 characters of the session id next to the buttons. Sessions older than 24 hours are deleted automatically when the server starts. The `data/` folder is git-ignored.

**Privacy between sessions:** a session can only see its own files. Files are never shared between sessions.

---

## Use a better model (optional)

TinyLlama is small and sometimes wrong. If your computer has the RAM, a larger model gives noticeably better answers, with no code changes:

```bash
ollama pull qwen2.5:3b            # about 1.9 GB, needs roughly 2.5 GB of RAM while running
ollama stop tinyllama             # optional: frees the memory TinyLlama was using

OLLAMA_MODEL=qwen2.5:3b uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The top right of the page then reads **"Ollama (qwen2.5:3b): online"**. Start without `OLLAMA_MODEL=…` to go back to TinyLlama. Larger models are slower on a CPU. Some settings (short answers, how much text is sent to the model) were tuned for TinyLlama's small 2,048-token window; see [Configuration](#configuration).

`3b` means 3 billion parameters (the numbers inside the model). More parameters usually means better reading and fewer mistakes, but needs more memory and time.

---

## Configuration

Every setting has a default and can be overridden with an environment variable put in front of the command (see [app/config.py](app/config.py)):

```bash
MAX_UPLOAD_BYTES=20971520 TOP_K=5 uvicorn app.main:app --host 127.0.0.1 --port 8000
```

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `tinyllama` | Ollama model name |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama address |
| `LLM_TIMEOUT` | `120` | Seconds to wait for Ollama to send *any* output before giving up |
| `LLM_MAX_TOKENS` | `200` | Maximum length of one model answer (stops runaway, repeating output) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Text splitting (characters) |
| `TOP_K` | `3` | Passages retrieved per question |
| `FULL_CONTEXT_MAX_CHARS` | `4500` | If all the text in a session is this short, it is sent to the model in full (no search) |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Upload and paste size limit, counted in bytes |
| `DATA_DIR` | `./data` | Where sessions are stored |
| `SQLITE_PATH` | `./sample.db` | Database used by the `/sql` endpoint |
| `SQL_MAX_ROWS` | `50` | Maximum rows returned by `/sql` |
| `SESSION_MAX_AGE_HOURS` | `24` | Sessions older than this are deleted at startup |
| `RAG_DEBUG` | `0` | `1` logs the full trace of every question (chunks, scores, the exact prompt) and enables `POST /debug/chat` |

Settings are read once, when the server starts. There is no `.env` file support: put the variables in front of the command as shown above. To use your own database for `/sql`: `SQLITE_PATH=./mydata.db uvicorn app.main:app --port 8000` (it is opened read-only).

**Run a single worker** (the default). The per-session locks that protect the search index live inside one process, so `--workers 2` or more would not be safe.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| The page says **"Ollama … offline"**, or answers fail with "Ollama is not reachable" (HTTP 503) | Ollama is not running. Start it with `ollama serve` (or `sudo systemctl start ollama`), then check `curl http://127.0.0.1:11434/api/tags`. |
| Error mentions the **model is missing** | Run `ollama pull tinyllama` (or the model you set in `OLLAMA_MODEL`). |
| `ModuleNotFoundError: No module named …` | The virtual environment is not active. Run `source venv/bin/activate` from the project folder. |
| `command not found: uv` | Install it (Step 3), then close and reopen the terminal. |
| `Address already in use` when starting | Something else uses port 8000. Use another port: `--port 8001`, and open that address instead. |
| The page says **"The running server is an older version and doesn't support this yet"** | The server was started before you updated the code. Stop it with Ctrl+C, start it again, then reload the page with **Ctrl+F5**. |
| The page says "Your session had expired" | The session was cleaned up (they last 24 hours). A new one was started; upload your document again. |
| The **first question is slow**, or "Thinking…" lasts a while | Ollama is loading the model into memory (10 to 20 seconds for larger models). If it stays stuck for minutes, press Ctrl+C on the server, run `ollama stop <model>`, and start again. |
| The **first start takes minutes** | The embedding model (about 90 MB) is being downloaded once. Later starts are fast. It needs internet the first time only. |
| Download of the embedding model fails | Check your internet or proxy for that first start. After it is cached, `HF_HUB_OFFLINE=1` keeps the app fully offline. |
| "No text could be extracted from this PDF" | The PDF is a scanned image. There is no OCR; upload a text-based PDF (one where you can select the text). |
| `/sql` says "Database file not found" | Run `python scripts/create_sample_db.py`, or point `SQLITE_PATH` at your database. |
| The app or Ollama is killed / very slow | Not enough free RAM. Close other programs, use `tinyllama`, and `ollama stop` any model you are not using. |
| The answer is wrong or vague | See [Known limitations](#known-limitations). Ask a more specific question using the document's words, check **Sources**, or [use a bigger model](#use-a-better-model-optional). |

---

## Debugging: see what the app does for a question

When an answer is wrong, first find out whether the **search** failed (the right text was never retrieved) or the **model** failed (it had the right text but answered badly). Three tools show this:

**1. The log.** Every question logs one summary line in the server terminal (no document text in it):

```
INFO:     question="what is git" route=search chunks_sent=3/574 answered=True time=1.2s
```

**2. The command-line tool** (run from the project folder with the environment active):

```bash
python -m app.debug --list                                   # sessions you can use
python -m app.debug "what is git" --model qwen2.5:3b         # newest session with documents
python -m app.debug "list all docker commands" --session 2bd5844b --rows 20
python -m app.debug "git" --retrieval-only                   # search only, skip the language model
python -m app.debug "what is git" --json                     # the same trace as JSON
```

It prints the original question, the typo-fixed and rewritten search query, the keywords, **every retrieved chunk with its page and scores** (blended score, meaning-search rank and distance, keyword rank and score, and a `*` for the chunks that were sent to the model), the number of chunks sent, **the exact prompt**, the raw model output, and why a fallback was used.

**3. Full trace in the log and an HTTP endpoint.** Start the server with `RAG_DEBUG=1` to log the whole trace of every question, and to enable `POST /debug/chat` (same body as `/chat`; it returns `{"result": ..., "trace": ...}`). Without `RAG_DEBUG=1` the endpoint does not exist. It is off by default because the trace contains your document text.

---

## Project layout

```
Chatbot/
  app/
    config.py        all settings in one place (env-var overridable)
    sessions.py      session creation, strict ID validation, cleanup
    rag.py           ingest documents + answer questions (search, overview, contact details)
    search.py        typo-tolerant keyword search and excerpts (no models, plain Python)
    tracing.py       records and prints what happened for a question (logging, trace text)
    debug.py         command-line tool: python -m app.debug "question"
    sql_agent.py     natural language -> safe read-only SELECT (used by /sql)
    main.py          FastAPI app and all endpoints
  static/index.html  the chat page (plain HTML + CSS + JS, no build step, works offline)
  scripts/
    create_sample_db.py   builds sample.db (customers + orders)
    try_rag.py            ingests a sample text and asks 3 questions with the real model
  tests/             pytest suite (needs no Ollama, no model downloads)
  requirements.txt   pinned Python packages (install CPU PyTorch first)
  data/              created at runtime: your sessions and uploaded files (git-ignored)
```

## Architecture

```
  Browser (static/index.html)                    curl / any HTTP client
             |                                            |
             +-------------------+------------------------+
                                 v
+-----------------------------------------------------------------------+
|  FastAPI  (app/main.py)                                               |
|  /session  /upload/{id}  /paste/{id}  /chat  /sql  /health  /  /docs |
|  validation, size limits, error -> HTTP status mapping                |
|                                                                       |
|  Loaded ONCE at startup (lifespan):  embedding model   LLM client     |
+---------+---------------------------------+---------------------------+
          |                                 |
          v                                 v
+---------------------------+     +---------------------------------+
| RAG  (app/rag.py)         |     | SQL agent (app/sql_agent.py)    |
|                           |     |                                 |
| ingest: PDF/TXT -> chunks |     | question -> prompt(schema)      |
|   -> embeddings -> FAISS  |     |   -> TinyLlama -> clean_sql()   |
| ask: overview / contact   |     |   -> is_safe()  --no--> BLOCKED |
|   details copied exactly, |     |   -> READ-ONLY SQLite -> rows   |
|   otherwise top chunks    |     |   -> TinyLlama explains rows    |
|   -> LLM -> short answer  |     +----------------+----------------+
+-----+---------------+-----+                      |
      |               |                            v
      v               |                  +-------------------+
+-------------------+ |                  | sample.db         |
| Sessions          | |                  | (mode=ro)         |
| (app/sessions.py) | |                  +-------------------+
| data/sessions/    | |
|  <32-hex id>/     | |
|   uploads/        | |
|   index/  (FAISS) | |
+-------------------+ |
                      v
        +---------------------------+
        | Ollama  127.0.0.1:11434   |
        | model: tinyllama          |
        +---------------------------+
```

---

## API examples

The chat page uses these same endpoints. Interactive documentation is at <http://127.0.0.1:8000/docs>.

```bash
# Health: is the server up, and can it reach Ollama?
curl http://127.0.0.1:8000/health
# {"status":"ok","ollama":true,"model":"tinyllama"}

# Create a session (keeps one user's documents separate from everyone else's)
SID=$(curl -s -X POST http://127.0.0.1:8000/session | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Upload a .pdf or .txt file (max 10 MB)
curl -F "file=@handbook.txt" http://127.0.0.1:8000/upload/$SID
# {"message":"Uploaded and indexed handbook.txt.","chunks":12,"filename":"handbook.txt"}

# ...or paste text instead of uploading a file (same 10 MB limit; each paste is its own document)
curl -X POST http://127.0.0.1:8000/paste/$SID \
  -H "Content-Type: application/json" \
  -d '{"text": "Our office closes at 5 PM on Fridays."}'
# {"message":"Added Pasted text 1.","chunks":1,"filename":"Pasted text 1"}

# Ask a question about the uploaded documents
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SID\", \"question\": \"What is the refund policy?\"}"
# {"answer":"...","sources":[{"file":"handbook.txt","page":null}]}

# Ask a question about the SQLite database (no session needed; needs sample.db, see Step 4)
curl -X POST http://127.0.0.1:8000/sql \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total order amount for each customer name?"}'
# {"sql":"SELECT c.name, SUM(o.amount) ...","rows":[{"name":"Alice Johnson","total":1640.99}, ...],"answer":"..."}

# A destructive request is refused
curl -X POST http://127.0.0.1:8000/sql \
  -H "Content-Type: application/json" -d '{"question": "Delete all orders"}'
# {"sql":"DELETE FROM orders ...","answer":"Blocked: only single SELECT queries are allowed."}
```

### Error codes

| Status | When |
|---|---|
| 400 | Bad input: empty or over-long question (max 1000 chars), malformed session ID, empty or blank text, empty/corrupt/scanned file, missing sample database |
| 404 | Unknown session |
| 413 | Upload or paste larger than 10 MB |
| 415 | File type other than `.pdf` / `.txt` |
| 503 | Ollama is unreachable, timed out, or the model is missing (the message says to run `ollama serve` or check the service) |
| 500 | Unexpected error: a generic message only; details go to the server log, never to the client |

---

## Safety design

### SQL mode: five layers

A small model can be talked into writing bad SQL, so nothing depends on it behaving.

1. **Prompt:** the model is told to output one `SELECT` and nothing else.
2. **`clean_sql()`:** strips markdown fences, a leading `sql`, and a trailing `;`, and drops any text after the first blank line. Only the query itself is ever considered.
3. **`is_safe()`:** the query must start with `SELECT` (or `WITH ... SELECT`), contain **no `;`** (one statement only), contain **no comments** (`--`, `/* */`, which can hide tricks), and match none of these whole words: `insert update delete drop alter create attach detach pragma replace vacuum truncate load_extension`.
4. **Read-only database connection:** SQLite is opened with `mode=ro` plus `PRAGMA query_only=ON`. **Even if layers 1 to 3 were somehow fooled, SQLite itself refuses to write.** A test proves this by deliberately bypassing `is_safe()`.
5. **Row cap:** at most 50 rows are fetched.

A blocked query returns `Blocked: only single SELECT queries are allowed.` and is never executed. The check errs on the side of blocking, so a legitimate query that uses a blocked word (even inside a quoted string, or the `replace()` function) is refused.

### Session isolation

- Every session is a random 128-bit ID (`uuid4().hex`) with its own folder: `data/sessions/<id>/uploads` and `.../index`.
- **All** file access goes through `session_dir()`, which requires the ID to match `^[0-9a-f]{32}$`, resolves the real path, and confirms it is still inside `data/sessions`. That rejects `../x`, `a/b`, absolute paths, empty IDs and non-hex strings, including in delete and cleanup.
- Each session has its own search index, so one session's question can only retrieve text from its own documents. A test uploads a "secret" to session A and confirms session B gets nothing.
- A per-session lock makes concurrent uploads to the same session safe.
- Uploads are stored under `<random>_<sanitized name>` inside the session's `uploads/` folder. The size limit is enforced while streaming, and rejected or half-written files are deleted. Pasted text is limited in bytes, the same way.
- Sessions older than 24 hours are removed at startup (only inside `data/sessions`).
- FAISS indexes are loaded with `allow_dangerous_deserialization=True`. That is safe here only because the app itself creates every index file; users can only upload `.pdf` and `.txt` files into a different folder.
- The chat page inserts all document and model text as plain text (never as HTML), so a malicious document cannot inject code into the page. It also loads nothing from the internet.

### What this does *not* protect

There is no user authentication: the session ID is the only "key", and the SQL endpoint is shared by everyone. The server binds to `127.0.0.1`. **Do not expose it to a network or the internet as-is.** Files in `data/` are stored unencrypted on disk.

---

## Known limitations

TinyLlama is a **1-billion-parameter** model chosen because it runs on a CPU in about 1 GB of RAM. It is small, and it shows:

- **It often ignores instructions.** It rarely replies with the exact sentence `I don't know based on the documents.`; it usually paraphrases ("not mentioned in the given text"). The app detects common phrasings and then shows the closest passages of your document instead, but it can miss unusual ones.
- **Spelling help has limits.** Words shorter than 4 letters ("gti") are not corrected, and only very close spellings are matched, so a heavily misspelled word may not find anything. The typo fixes for question types ("documenbt", "emial", "sumary") cover a short, safe list of words, on purpose, so that real words such as "contract" are never rewritten into something else.
- **It can still hallucinate or mix up details.** With a document that has several numbers (an offer letter with a rate, hours, dates and IDs), it can pick the wrong one. It also mistypes names and emails when copying, which is why emails, phone numbers and the opening lines are copied by the app instead. Always check **Sources**.
- **It sees only a little text at a time.** Unless the whole text is short (see `FULL_CONTEXT_MAX_CHARS`), only the top 3 passages (about 1,500 characters) are sent, and TinyLlama has a 2,048-token context window. Questions that need information spread across a long document (summaries, "list everything", totals) work poorly. It is not comparable to cloud assistants such as ChatGPT or Claude, which are far larger and can read whole documents.
- **Tables in PDFs lose their layout.** Text is extracted line by line, so column alignment is often lost. Exact counts and totals over a table are not reliable.
- **SQL quality is limited.** It handles simple counts, sums and joins, but sometimes uses columns that do not exist or invents tables. It also sometimes miscounts when *describing* a result, so trust the returned `rows` over the sentence. Errors are reported in a friendly way, and nothing unsafe can run.
- **Answers are kept short on purpose.** TinyLlama does not know when to stop and tends to repeat itself, so each answer ends at its first blank line (an intro that ends in a colon keeps the list that follows it), is capped at `LLM_MAX_TOKENS`, and is trimmed back to a full sentence if the cap cut it off. A legitimate multi-paragraph answer will be shortened.
- **Some questions are answered by the app, not the model.** Overview questions ("what is this about?", "whose resume is this?", "what is the candidate's name?") use the file name and first lines, copied exactly, plus one sentence from the model. Contact questions ("what is the email?") copy emails, phone numbers and links out of the text by pattern. For long documents, contact details are only looked for at the top, so sample addresses inside a manual are not mistaken for the owner's.
- **The same file uploaded twice is indexed twice** (and so is the same text pasted twice), which can push a short document over the "send it whole" limit and worsen answers. Use **New session** when you switch documents.
- **No OCR:** scanned PDFs (images of text) are rejected with an explanatory message.
- **Speed:** on a CPU an answer takes a few seconds; the first request after idle is slower while Ollama loads the model.

Swapping in a larger model is the easiest way to improve quality; see [Use a better model](#use-a-better-model-optional).

---

## How to test

```bash
source venv/bin/activate
pytest
```

The tests **do not need Ollama or a model download.** The LLM and the embeddings are replaced with fakes, and every test uses its own temporary data folder. They cover:

- `tests/test_sessions.py`: folders, ID validation, isolation, cleanup
- `tests/test_rag.py`: ingestion, rejection of bad files, prompts, short answers, overview and contact questions, misspelled and one-word questions, the "closest parts" fallback, concurrent uploads
- `tests/test_search.py`: typo correction, keyword scoring, blending two searches, excerpts
- `tests/test_sql_safety.py`: `is_safe()` accept/reject cases, "Delete all orders" is blocked, the read-only connection refuses writes
- `tests/test_api.py`: every endpoint and error code, uploads and pasted text, session isolation, and checks on the chat page (its script, no internet resources)

One test that checks the page's JavaScript syntax is skipped if Node.js is not installed; that is fine.

To try the **real** model on a sample text (Ollama must be running):

```bash
python scripts/try_rag.py
```
