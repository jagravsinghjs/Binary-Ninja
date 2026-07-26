# 1.7 Limitations

The items below are constraints inherent to the current approach — trade-offs that follow
directly from design choices (local, private, offline-capable) rather than gaps that more
development time alone would close. They are kept separate from Section 1.8 (Future Work),
which lists improvements that *are* achievable within the current approach.

- **Hardware dependency.** The system requires a CUDA-capable GPU with sufficient VRAM to run
  Whisper-medium, the text-emotion classifier, and a 7B-parameter Ollama model together.
  Without a GPU, transcription and classification fall back to CPU and become too slow for a
  live conversation to feel conversational.

- **Local model capability ceiling.** `qwen2.5:7b-instruct` is substantially smaller and less
  capable than cloud-hosted models (e.g. GPT-4-class or Claude). This is a direct consequence
  of prioritizing privacy and offline operation over raw output quality, and it is the reason
  a meaningful share of the reliability work in this project — output formatting, emoji
  handling, gating when to mention a doctor — was implemented deterministically in code rather
  than left to the model's judgment. A smaller model cannot be relied on to consistently
  follow nuanced instructions on its own.

- **Turn latency.** Each conversational turn requires transcription, acoustic feature
  extraction, text-emotion classification, and an LLM call, executed in sequence. The
  interaction is not instantaneous, and pacing degrades further on less capable hardware.

- **English-only in practice.** Both the transcription model and the text-emotion classifier
  perform best in English. Given the project's target context (India), this is a genuine gap
  in the current build rather than a hypothetical one.

- **Acoustic signal fragility.** Pitch, energy, and pause-based features depend on reasonably
  clean audio. Background noise or a low-quality microphone degrades the reliability of the
  acoustic layer — a limitation inherent to acoustic-feature-based analysis in general, not
  specific to this implementation.

- **No clinical validation.** Distress scores, acoustic feature thresholds, and text-emotion
  mappings are engineering heuristics, not validated against real diagnosed patient
  populations or peer-reviewed clinical benchmarks. They are intended as decision-support
  signals for a clinician, not clinical measurements, and should not be interpreted as such.

- **Environment fragility.** The system depends on matched CUDA/cuDNN driver versions, a
  running local Ollama instance, and specific library versions. It is not, in its current
  form, a portable install for an arbitrary machine.
