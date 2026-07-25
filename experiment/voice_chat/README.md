# Voice Chat Module

Executes an interactive chat loop across dynamic user turns, accumulating context until final report generation.

## DFD Level 0

[ Input: Interactive User Audio Stream / Turns ]
       │
       ▼
[ Process: Continuous Chat Loop (Librosa + Session State) ]
       │
       ▼
[ Trigger: Report Generation Request ]
       │
       ▼
[ Output: Session Report File (report.json) ]

## Overview
- **Execution:** Interactive continuous loop until user terminates/requests report.
- **Input:** Live/sequential user voice audio inputs across conversation turns.
- **Algorithm/Engine:** Turn-taking state manager with `Librosa` feature extraction per segment.
- **Output:** Aggregated session state exported into a final report document upon loop completion.