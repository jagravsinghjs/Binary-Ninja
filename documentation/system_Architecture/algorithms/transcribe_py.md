\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[ruled,vlined,linesnumbered]{algorithm2e}
\usepackage{amsmath}

\SetKwInput{KwInput}{Input}
\SetKwInput{KwOutput}{Output}

\begin{document}

\section*{1.4 Algorithms}

\begin{algorithm}[H]
\caption{Speech-to-Text Transcription (\texttt{transcribe.py})}
\KwInput{Audio file (\texttt{.wav})}
\KwOutput{\texttt{transcript.txt}, \texttt{transcript.json}}

Load \texttt{WhisperModel} (\textit{medium}, device = CUDA, compute\_type = float16)\;
\textit{segments}, \textit{info} $\leftarrow$ model.transcribe(audio\_path, beam\_size = 5)\;
\textit{transcript} $\leftarrow$ empty list\;
\ForEach{segment in segments}{
    Extract \textit{start}, \textit{end}, \textit{text} from segment\;
    Write formatted line to \texttt{transcript.txt}\;
    Append \{start, end, text\} to \textit{transcript}\;
}
Save \textit{transcript} as \texttt{transcript.json}\;
\end{algorithm}

\vspace{4mm}

\begin{algorithm}[H]
\caption{Acoustic Feature Extraction (\texttt{voice\_emotion.py})}
\KwInput{Audio file (\texttt{.wav}), \texttt{transcript.json}}
\KwOutput{\texttt{transcript\_with\_emotion.json} (acoustic features + arousal label per segment)}

Load audio waveform $y$ at native sample rate $sr$ using \texttt{librosa}\;
\ForEach{segment in transcript}{
    $seg \leftarrow y[\,\text{start} \cdot sr : \text{end} \cdot sr\,]$\;
    \If{$seg$ is empty}{ skip segment }
    Estimate pitch contour $f_0$ via \texttt{librosa.pyin} (range 75--450 Hz)\;
    Remove octave-jump outliers from $f_0$ using median-based filtering\;
    Compute $pitch\_mean$, $pitch\_std$ from cleaned $f_0$\;
    Compute RMS energy $\rightarrow$ $energy\_mean$, $energy\_std$\;
    Compute zero-crossing rate and spectral centroid\;
    Compute $pause\_ratio$ from voiced/silent interval split (top\_db = 25)\;
    \uIf{$pitch\_std > 40$ \textbf{and} $energy\_std > 0.02$}{
        $arousal \leftarrow$ \textit{high\_arousal}\;
    }
    \uElseIf{$pause\_ratio > 0.3$}{
        $arousal \leftarrow$ \textit{low\_arousal}\;
    }
    \Else{
        $arousal \leftarrow$ \textit{neutral}\;
    }
    Attach features and $arousal$ label to segment\;
}
Save enriched segments as \texttt{transcript\_with\_emotion.json}\;
\end{algorithm}

\vspace{4mm}

\begin{algorithm}[H]
\caption{Text Emotion Enrichment (\texttt{emotion.py})}
\KwInput{\texttt{transcript\_with\_emotion.json} from Algorithm 2 (words + acoustic features)}
\KwOutput{\texttt{transcript\_with\_emotion.json} (adds text-emotion scores per segment)}

Load \texttt{j-hartmann/emotion-english-distilroberta-base} classifier (GPU)\;
\ForEach{segment in input}{
    $prediction \leftarrow$ classifier(segment.text)\;
    $emotions \leftarrow$ \{label: score \textbf{for} each (label, score) in prediction\}\;
    Attach $emotions$ to segment as $text\_emotion$\;
}
Save segments (now containing words, acoustic features, \textbf{and} text emotion) as
\texttt{transcript\_with\_emotion.json}\;
\end{algorithm}

\vspace{4mm}

\begin{algorithm}[H]
\caption{LLM-Based Report Generation (\texttt{chat\_llm.py})}
\KwInput{\texttt{transcript\_with\_emotion.json} (words + acoustic features + text emotion)}
\KwOutput{\texttt{report.json}, \texttt{mental\_state.png}}

Load segments from input file\;
Send segments to local LLM (Ollama, \textit{qwen2.5:7b-instruct}) with a decision-support
system prompt (JSON-constrained output)\;
Parse response $\rightarrow$ \{segments with $distress\_score$, $clinician\_summary$,
$patient\_message$\}\;
\If{trailing emoji block detected in $patient\_message$}{
    Redistribute emoji across sentences (\texttt{redistribute\_trailing\_emoji})\;
}
Compute $avg\_first$, $avg\_second$ from first/second half of $distress\_score$ sequence\;
\If{$(avg\_second - avg\_first) \geq 1.5$ \textbf{or} final score $\geq 6$}{
    Append doctor-nudge sentence to $patient\_message$ (code-level decision, not LLM-decided)\;
}
Save result as \texttt{report.json}\;
Plot $distress\_score$ vs. segment start time $\rightarrow$ \texttt{mental\_state.png}\;
\end{algorithm}

\vspace{4mm}

\begin{algorithm}[H]
\caption{Live Voice Check-in Session (\texttt{voice\_chat.py})}
\KwInput{Live microphone audio, turn by turn}
\KwOutput{\texttt{report.json}, \texttt{mental\_state.png} (per session, auto-saved)}

Load Whisper (\textit{medium}, CUDA) and text-emotion classifier once at startup\;
Initialize conversation with a fixed opening message\;
$turns \leftarrow$ empty list\;
\Repeat{user selects \emph{Generate Report}}{
    Record one turn (Enter to start / Enter to stop) $\rightarrow$ save to temporary \texttt{.wav}\;
    Transcribe temporary audio $\rightarrow$ $turn\_text$\;
    \If{$turn\_text$ is empty}{ discard temp file, prompt again, \textbf{continue} }
    Extract acoustic features and arousal label from temporary audio\;
    Classify $turn\_text$ with text-emotion classifier\;
    \textbf{Delete temporary audio file} (raw audio never persists past this point)\;
    Build hidden context note: acoustic cues + top text-emotion labels\;
    Append $turn\_text$ + context note to conversation history\;
    $reply \leftarrow$ Ollama conversational LLM call (listening/no-advice system prompt)\;
    Display $reply$ to user\;
    Append \{$turn\_text$, features, $text\_emotion$, elapsed time\} to $turns$\;
    Prompt user: press Enter to continue, or select \emph{Generate Report} to end\;
}
\If{$turns$ is non-empty}{
    Send full $turns$ history to Ollama with report system prompt $\rightarrow$
    \{per-turn $distress\_score$, $clinician\_summary$, $patient\_message$\}\;
    Apply emoji redistribution and doctor-nudge logic (as in Algorithm 4)\;
    Create timestamped session folder\;
    Save \texttt{report.json} and $distress\_score$-vs-time graph (\texttt{mental\_state.png})\;
}
\end{algorithm}

\end{document}