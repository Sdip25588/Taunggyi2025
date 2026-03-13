# 📚 Taunggyi English Tutor

> **AI-powered personalized English education platform** for Grades 1–3, inspired by **Perplexity AI** and **Synthesis Tutor**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-orange)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Features

- 🤖 **AI Tutor** — Google Gemini 1.5 Flash explains phonics, reading, and spelling step-by-step
- 📄 **RAG-Grounded** — Every lesson is anchored to the actual curriculum PDFs via FAISS vector search
- 🧑‍🏫 **Preface-Driven Teaching** — Teaching methodology extracted directly from McGuffey textbook prefaces
- 📊 **Progress Dashboard** — Track accuracy, streak, lessons completed, and topic mastery
- 🏅 **Badges & Gamification** — Earn badges for milestones, maintain daily streaks
- 📝 **Interactive Quizzes** — Multiple-choice and fill-in-the-blank with instant feedback
- 📈 **Adaptive Difficulty** — Automatically adjusts lesson level based on student performance
- 🔊 **Text-to-Speech** — Read lessons aloud via `edge-tts` (free) or Azure TTS (optional)
- 🎨 **Visual Aids** — Alphabet charts, phonics sound charts, word family diagrams
- 💾 **Persistent Profiles** — SQLite database stores progress across sessions
- 👤 **Simple Login** — Username-only (no password), expandable to full auth later

---

## 🏗️ Architecture

```
Taunggyi2025/
├── main.py                    # Streamlit app entry point
├── config.py                  # API keys, model routing, PDF paths, TTS/RAG config
├── gui_engine.py              # Streamlit UI: chat, sidebar, quizzes, dashboard, TTS
├── human_engine.py            # Human-like teaching prompts from PDF prefaces
├── visual_teacher.py          # Matplotlib educational visuals
├── learning_orchestrator.py   # Session coordination, RAG + LLM workflow
├── ai_teacher.py              # Gemini LLM calls + FAISS RAG pipeline
├── student.py                 # SQLite student profile tracking
├── mistake_analyzer.py        # Error detection and explanation
├── adaptive_path.py           # Adaptive difficulty adjustment
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment variables
├── curriculum/                # Place curriculum PDFs here
│   ├── .gitkeep
│   └── README.md
├── data/                      # SQLite DB + FAISS index (auto-created)
│   └── .gitkeep
└── README.md
```

### How It Works

```
Student Input
     │
     ▼
Learning Orchestrator ──► Intent Detection (lesson/quiz/review/visual)
     │
     ├──► RAG Pipeline (ai_teacher.py)
     │         └── FAISS similarity search → relevant PDF chunks
     │
     ├──► Human Engine (human_engine.py)
     │         └── Build preface-guided teaching prompt
     │
     ├──► Gemini 1.5 Flash LLM
     │         └── Grounded, step-by-step response
     │
     ├──► Mistake Analyzer (mistake_analyzer.py)
     │         └── Detect errors, generate corrections
     │
     ├──► Student DB (student.py)
     │         └── SQLite: update progress, record mistakes
     │
     └──► Adaptive Path (adaptive_path.py)
               └── Adjust difficulty, recommend next topic
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Sdip25588/Taunggyi2025.git
cd Taunggyi2025
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Add Curriculum PDFs

Place the following files in the `curriculum/` folder:
- `phonics.pdf` — Phonics curriculum for Grades 1–3
- `reading.pdf` — McGuffey's First Eclectic Reader
- `Spelling.pdf` — McGuffey's Eclectic Spelling Book

See `curriculum/README.md` for details.

### 5. Run the App

```bash
streamlit run main.py
```

Open your browser to `http://localhost:8501` 🎉

---

## 🔑 Getting API Keys

### Google Gemini (Required)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Copy the key into your `.env` file as `GEMINI_API_KEY`
4. **Free tier available** — no credit card needed

### Azure TTS (Optional — for premium voice quality)

The app uses **`edge-tts`** by default (free, no key needed). Azure TTS is optional:

1. Go to [Azure Portal](https://portal.azure.com)
2. Create a **Speech Services** resource (Free F0 tier: 500K characters/month)
3. Copy **Key 1** and **Region** to your `.env`:
   ```env
   AZURE_SPEECH_KEY=your_key_here
   AZURE_SPEECH_REGION=eastus
   TTS_PROVIDER=azure
   ```

---

## 📖 Curriculum Subjects

| Subject | Source | Grade |
|---------|--------|-------|
| **Phonics** | Phonics curriculum PDF | 1–3 |
| **Reading** | McGuffey's First Eclectic Reader | 1–2 |
| **Spelling** | McGuffey's Eclectic Spelling Book | 1–3 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local, free) |
| Vector Store | FAISS (local) |
| PDF Extraction | LangChain + PyPDF2 |
| Web UI | Streamlit |
| TTS | `edge-tts` (default) / Azure TTS (optional) |
| Database | SQLite |
| Visuals | Matplotlib |

---

## 🧪 Manual QA Steps

### Professor-Led Conversation Flow

1. **Start the app:**
   ```bash
   streamlit run main.py
   ```
2. **Login:** Enter any name (e.g. `Aung`) and select Grade 1. Click **Start Learning!**
3. **Greeting check:** The AI should immediately greet you with a *varied* message (e.g. "Good morning, Aung! How are you today?"). The greeting changes each session.
4. **Check-in reply:** Type a short reply such as `I'm good!` — the AI should acknowledge your mood and then introduce today's topic.
5. **Doubt handling:** Before the lesson begins, type a question like `What is a vowel?` — the AI should answer the question *first*, then smoothly transition into the lesson.
6. **Professor-chosen lesson:** The AI should announce the topic (e.g. "Today we'll work on Short Vowels — here's why…"). The student does **not** choose the topic.
7. **Variety check (multi-session):** Log out, log back in. The topic/subject chosen in the second session should differ from the first (Strategy B variety rotation).
8. **Gradual upgrade:** Answer quiz questions correctly several times. The AI should note your progress and gradually increase difficulty or move to the next topic.
9. **Warm style check:** If you answer incorrectly, the AI should say something like "Almost there — let's try together!" rather than a harsh negative.
10. **TTS/emoji check:** Click **🔊 Read Aloud** on any AI message. Emojis should *not* be read aloud; the spoken text should be clean and natural.

### Session-State Smoke Test

After login, open the Streamlit debug panel (add `?debug=true` to the URL, or add `st.write(st.session_state)` temporarily) and verify these keys exist with correct initial values:

| Key | Expected initial value |
|-----|------------------------|
| `conv_state` | `"GREETING"` (`CONV_GREETING`) |
| `greeting_done` | `False` |
| `todays_focus` | `None` |
| `chat_history` | `[]` |
| `current_subject` | `"Phonics"` (or student's saved subject) |

---



- [ ] 7 subjects (Math, Science, History, Geography, Myanmar, Art)
- [ ] Multi-model routing (OpenAI GPT-4, Anthropic Claude)
- [ ] Parent/teacher dashboard
- [ ] Gamified lesson maps (like Duolingo)
- [ ] Voice input (speech-to-text)
- [ ] Mobile-responsive layout
- [ ] Multiplayer/classroom mode

---

## 🛠️ Troubleshooting

### `zsh: command not found: code` on macOS

This means the VS Code `code` command-line tool is not on your `PATH` yet. Fix it in three steps:

#### Step 1 — Install the `code` shell command from VS Code

1. Open **Visual Studio Code** (download it from <https://code.visualstudio.com/> if needed).
2. Press **Cmd + Shift + P** to open the Command Palette.
3. Type **Shell Command: Install 'code' command in PATH** and press **Enter**.
4. VS Code will show: *"Shell command 'code' successfully installed in PATH."*

#### Step 2 — Restart Terminal

The change only takes effect in a **new** Terminal window. Quit Terminal completely and reopen it:

```bash
# Quit Terminal (Cmd + Q), then reopen it, then verify:
which code
code --version
```

You should see a path like `/usr/local/bin/code` and a version number such as `1.90.0`.

#### Step 3 — If `code` is still not found after restarting

Run the following commands to add VS Code to your `PATH` manually:

```bash
# Confirm the binary exists
ls "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"

# Add it to your shell profile permanently
echo 'export PATH="$PATH:/Applications/Visual Studio Code.app/Contents/Resources/app/bin"' >> ~/.zshrc
source ~/.zshrc

# Verify
which code
code --version
```

#### Alternatives when `code` is not available

If you still cannot use `code`, open files with one of these options instead:

```bash
# Open files in VS Code via macOS (works even without the PATH fix)
open -a "Visual Studio Code" gui_engine.py learning_orchestrator.py main.py

# Edit directly in Terminal with nano (no install required)
nano gui_engine.py
# Nano tips:
#   Ctrl+W  — search (e.g. type <<<<<<< to jump to conflict markers)
#   Ctrl+O  — save
#   Ctrl+X  — exit
```

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
