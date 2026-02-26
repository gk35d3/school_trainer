# school_trainer

Two fullscreen pygame trainers:
- `mathe_trainer.py`
- `deutsch_trainer.py`
- shared adaptive logic: `adaptive_core.py`

Both apps now write to one shared file in the project root:
- `trainer_data.jsonl`

## Run

```bash
python3 mathe_trainer.py
python3 deutsch_trainer.py
```

## Shared Data Format

Each line in `trainer_data.jsonl` is JSON with at least:
- `type`: `session_start` | `attempt` | `session_end`
- `app`: `math` | `german`
- `session_id`
- `t` (unix timestamp)

Attempt lines also include:
- `q_index`
- `correct`
- `rt` (response time)
- `difficulty`
- `tags` (skill tags used for weakness targeting)

## Harmonized Session Structure

Both trainers use fixed sets of questions:
- `SESSION_QUESTIONS = 40`
- session start difficulty = latest logged difficulty for that app (`math` or `german`)

## German Exercise Type

The German trainer now uses only:
- `open_question` (free composition answer)

Checks focus on:
- grammar basics (capitalization, verb presence, sentence length)
- punctuation at sentence end
- keyword-based meaning coverage
- likely spelling errors near expected anchor words

## Focus Retuning

Math now prioritizes:
- `add_carry`
- `sub_borrow`
- two-digit place-value stability

German now prioritizes:
- noun capitalization
- consonant clusters (`sch`, `ch`)
- consonant doubling (`nn`, `mm`, `tt`)
- verb endings
- punctuation
- umlaut/ß handling

German prompt generation has been expanded with broader vocabulary and more sentence templates.
