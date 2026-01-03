# Phishing Detector — Update

This folder contains a cleaned, minimal phishing detection prototype with an incremental online trainer and an interactive Streamlit UI for inference and feedback-driven online learning.

Quick start
-----------
1. (Optional) Create & activate a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r Update/requirements.txt
```

3. Run the Streamlit app:

```bash
python -m streamlit run Update/streamlit_app.py --server.port 8501
```

4. Optional: run or retrain the incremental online model from the CLI:

```bash
python Update/online_trainer.py --epochs 2 --model-dir "Update/Trained model" --resume
```

What the Streamlit app does
- Accept `.eml` uploads or manual `sender`, `subject`, `body` input.
- Shows heuristic feature values and critical flags (homoglyphs, shorteners, IPs in URLs, etc.).
- Displays predictions from the incremental online model (and from a saved Keras hybrid model if present).
- Accepts user feedback (phishing / legitimate) and incrementally updates the online model; feedback is persisted to `Update/Trained model/feedback.csv`.

Files you should care about
- `streamlit_app.py` — interactive UI and feedback workflow.
- `online_trainer.py` — incremental trainer (uses `river` when available; sklearn fallback supported).
- `features.py` — robust feature extractor detecting homoglyphs, URLs, urgency words, etc.
- `model_builder.py` — Keras hybrid model builder (text + manual features).
- `data_loader.py` — dataset loader used for initial/batch training.
- `Trained model/` — model artifacts, feedback.csv, and versioned online model saves.

Notes & next steps
- The app updates the lightweight incremental model online; the Keras model (if present) is shown for comparison but not fine-tuned in the Streamlit session.
- If the Keras model fails to load the app shows the full load error and a retrain command: `python Update/main.py --mode train`.
- I can add an automated periodic Keras fine-tuning workflow that consumes labeled feedback and checkpoints safely — tell me if you want that.

Maintenance
-----------
- To export a versioned copy of the online model use the app's "Export Latest Online Model (versioned)" button or call `trainer.save(versioned=True)` from a small script.

Stopping the Streamlit app
--------------------------
- If you started Streamlit in a terminal, press `Ctrl+C` to stop it.
- To stop it from another shell (Linux/macOS):

```bash
# kill by process name
pkill -f streamlit || true

# or kill the process listening on port 8501 (if used):
kill $(lsof -t -i:8501) 2>/dev/null || true
```

These commands are safe to run; the `|| true` ensures they don't error if no process is found.
