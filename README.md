# school_trainer

## Goal
A simple, adaptive learning trainer for children:
- **Math**: addition/subtraction within `1..100`, with progressive difficulty.
- **German**: open written answers, with live grammar/spelling feedback and adaptive challenge.
- **1x1**: multiplication facts `1..10`, high-repetition with adaptive weak-fact focus.
- **Uhrzeit**: analog clock reading with adaptive stage unlocks.

The app tracks weaknesses from past attempts and focuses future questions on those skills.

## Super-Easy Mac Install
1. Clone the repository:
```bash
git clone https://github.com/<your-org-or-user>/school_trainer.git
cd school_trainer
```
2. In Finder, open the `school_trainer` folder and **double-click**:
- `Install_Desktop_Launchers.command`
3. Go to Desktop and run one of these:
- `Mathe.command`: starts the adaptive math trainer (addition/subtraction 1..100).
- `Deutsch.command`: starts the adaptive German writing trainer (free-answer questions with feedback).
- `1x1.command`: starts the adaptive 1x1 multiplication trainer.
- `Uhrzeiten.command`: starts the adaptive analog clock trainer.
- `Update.command`: pulls the latest changes from git (`fetch` + `pull --ff-only`).

Notes:
- Launchers are standalone scripts and include custom icons.
- If macOS blocks first run: right-click -> **Open** once.

## Run Without Desktop Launchers
From repo root:
```bash
python3 -m apps.mathe_trainer
python3 -m apps.deutsch_trainer
python3 -m apps.times_trainer
python3 -m apps.uhrzeit_trainer
```

## Modular Structure
```text
apps/
  mathe_trainer.py       # Math UI + generation + adaptive session flow
  deutsch_trainer.py     # German UI + free-writing checks + adaptive session flow
  times_trainer.py       # 1x1 UI + adaptive weak-fact repetition
  uhrzeit_trainer.py     # Analog clock UI + adaptive time-reading training

core/
  adaptive_core.py       # Shared adaptive difficulty/tag logic
  trainer_data.py        # Shared JSONL event persistence

data/
  trainer_data.jsonl     # Runtime learning history

assets/icons/
  math.png               # Web-sourced math icon
  german.png             # Web-sourced german icon
  times.png              # Web-sourced 1x1 icon
  clock.png              # Web-sourced clock icon
  update.png             # Web-sourced update icon

Install_Desktop_Launchers.command  # Generates the Desktop launchers
```

## Technical Features
- Shared event log (`JSONL`) for both subjects.
- Per-tag adaptive difficulty with weighted weakness targeting.
- App-specific start difficulty from latest logged difficulty.
- Math hardness scales by difficulty bands; constrained to `1..100`.
- German hardness scales by prompt level + writing requirements (minimum words/verbs).
- Live German error highlighting (incorrect words shown in red).
- Unicode-safe text handling (umlauts/ß).
- Fullscreen keyboard-first UI for child-friendly usage.

## Data Format (trainer_data.jsonl)
Each line is a JSON event with fields like:
- `type`: `session_start | attempt | session_end`
- `app`: `math | german`
- `session_id`
- `t` (unix timestamp)
- `difficulty`
- `tags`

`attempt` events also include user response details (`typed`, `correct`, `rt`, etc.).

## Update Workflow
Use Desktop `Update_Repo.command` or run:
```bash
git fetch --all --prune
git pull --ff-only
```
