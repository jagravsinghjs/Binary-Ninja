## Algorithm 4: LLM-Based Report Generation (`chat_llm.py`)

**Input:** `transcript_with_emotion.json` (words + acoustic features + text emotion)
**Output:** `report.json`, `mental_state.png`

1. Load segments from input file
2. Send segments to local LLM (Ollama, `qwen2.5:7b-instruct`) with a decision-support system prompt (JSON-constrained output)
3. Parse response → `{segments with distress_score, clinician_summary, patient_message}`
4. If trailing emoji block detected in `patient_message` → redistribute emoji across sentences (`redistribute_trailing_emoji`)
5. Compute `avg_first`, `avg_second` from first/second half of `distress_score` sequence
6. If `(avg_second − avg_first) ≥ 1.5` **or** final score `≥ 6` → append doctor-nudge sentence to `patient_message` (code-level decision, not LLM-decided)
7. Save result as `report.json`
8. Plot `distress_score` vs. segment start time → `mental_state.png`