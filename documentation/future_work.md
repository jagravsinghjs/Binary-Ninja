# 1.8 Future Work

### Authentication & Access Control
- Patient accounts (username/password) so sessions and reports are private to each user.
- Doctor accounts, with a dashboard showing how many check-in sessions a patient has taken
  and access to their report history.
- Per-user data isolation — currently all reports are saved in a single shared folder with
  no ownership boundary; anyone with filesystem access can view, copy, or delete any
  patient's session.
- Audit logging — track which doctor viewed which patient's report and when, once doctor
  accounts exist.

### Data Reliability & Storage
- Replace flat-file session storage with a proper database (SQLite, as originally scoped).
  This also fixes a real bug: session folders are currently named by timestamp down to the
  second, so two sessions ending in the same second overwrite each other. A database with
  auto-incrementing or UUID-based session IDs removes this risk entirely and makes per-user
  queries (e.g. "all sessions for patient X") practical instead of scanning folders.
- Encryption at rest for stored reports — these are sensitive mental-health records
  currently saved as plaintext JSON.
- A data retention policy — auto-archive or delete sessions after a defined period, or once
  a doctor has reviewed them, rather than accumulating indefinitely.

### Clinical Depth
- Cross-session trend view for the doctor. The current graph only shows the distress
  trajectory *within* a single session; a doctor would benefit more from seeing the trend
  across several weeks of check-ins at a glance.
- Calibrate the acoustic-feature heuristic thresholds (e.g. pitch variability, energy
  variability cutoffs) against real recorded data rather than the initial estimated values
  used in this prototype.
- Multilingual support. Given the target context is India, Hindi and other regional
  languages matter for real-world reach — both transcription and text-emotion scoring are
  currently English-centric.
- **Escalation alerting** (flagging a patient whose distress is climbing sharply between
  doctor visits) is a natural extension but is explicitly *not* treated as a simple
  engineering feature here — it would require clinical input on appropriate thresholds and
  false-positive handling before any implementation, given the sensitivity of automated
  alerts in a mental-health context.

### Multimodal Signal Expansion
- **Video-based facial emotion analysis.** If the check-in is done with the camera on
  (opt-in), facial expression analysis could be added as a third signal alongside acoustic
  features and text emotion — potentially catching distress cues that neither voice nor
  words fully capture on their own, similar to how the acoustic layer already catches
  "I'm fine" said in a voice that suggests otherwise. As with escalation alerting, this
  needs explicit treatment before implementation, not just a feature add: it would require
  clear, informed consent (camera use is a much larger privacy step than audio), a policy on
  whether video is ever stored or purely processed in-memory like audio currently is, and
  validation that the added signal actually improves report quality rather than just adding
  noise and computational cost.

### Frontend & Performance
- A properly designed frontend (the current one is a functional, minimal placeholder built
  to validate the pipeline, not a final UI).
- Text-to-speech for spoken assistant replies (currently text-only).
- General code optimization — reducing model load time, evaluating smaller/quantized models
  where latency matters more than maximum accuracy, and batching where applicable.
