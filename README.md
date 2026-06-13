# SBERT-Talent-Scorer

A highly optimized, high-throughput candidate ranking pipeline designed to shortlist the top 100 candidates for a **Senior AI Engineer** role from a pool of 100,000 profiles. The pipeline evaluates candidate profiles using custom heuristics, behavioral multipliers, and advanced semantic matching.

## Key Features

* **Honeypot Filter**: Robust consistency heuristics (duration mismatches, fake expert skill counts) to automatically exclude auto-disqualifying candidate accounts.
* **Stage 1 Fast Filter**: Enforces hard constraints (experience range of 2–15 years, filtering out service-company-only history, high-recall keywords matching) to reduce candidate pool size by 94% in under 5.5 seconds.
* **Stage 2 ONNX Semantic Scorer**: Dense vector space similarity check against the Job Description utilizing an **INT8 Quantized `all-MiniLM-L6-v2`** model loaded via **ONNX Runtime** for high-efficiency CPU execution.
* **Behavioral Multipliers**: Scales core technical/semantic fit scores dynamically using 23 platform signals (notice period, recruiter response rate, profile completeness, etc.).
* **Deterministic Tie-Breaking**: Breaks score ties deterministically using candidate ID ascending.

---

## Tech Stack

* **Language**: Python 3.14+
* **ML Inference**: ONNX Runtime (`onnxruntime` + `transformers` tokenizer)
* **Model**: INT8 Quantized `all-MiniLM-L6-v2` (compacted from 90MB to **22.8MB**)
* **Parsing Utilities**: `orjson` (for high-speed JSONL streaming), `python-docx`
* **Test Suite**: `pytest`

---

## Prerequisites

Ensure you have the following installed on your machine:
* Python 3.14+
* pip (Python package installer)

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Ayush-Sonavane/SBERT-Talent-Scorer.git
cd SBERT-Talent-Scorer
```

### 2. Download and Extract the Dataset
Download the challenge dataset ZIP file from Google Drive and extract its contents directly into the root of the repository:
* **Dataset Download Link**: [Download Dataset ZIP](https://drive.usercontent.google.com/download?id=1MfD47XvVdRKBGRAyzGOxDCEf2ve96Jjo&export=download&authuser=0)
* **Extraction Directory**: Ensure the extracted folder `[PUB] India_runs_data_and_ai_challenge` is located at the root of the repository directory.

### 3. Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Run the Ranking Pipeline
To run the end-to-end candidate shortlisting:
```bash
python rank.py --candidates "./[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --out submission.csv
```
This will parse the job description from the Word document, filter the pool, compute semantic embeddings, score each profile, and write the top 100 shortlist to `submission.csv`.

### 5. Run the Unit Tests
Execute the test suite to verify implementation correctness:
```bash
python -m pytest tests/ -v
```

---

## Architecture Overview

### Directory Structure

```
├── models/
│   └── all-MiniLM-L6-v2/           # Pre-quantized ONNX model & tokenizer assets
├── src/
│   ├── honeypot.py                 # Profile heuristic check logic
│   ├── retrieval.py                # Stage 1 fast keyword & company filters
│   ├── scorer.py                   # Dynamic behavioral modifier and score calculations
│   ├── semantic.py                 # ONNX matcher class and mean-pooling logic
│   └── utils.py                    # Fast docx loading & stream JSON utilities
├── tests/                          # 18 pytest unit tests
├── rank.py                         # Command-line entrypoint script
├── requirements.txt                # Project dependencies
└── submission.csv                  # Shortlisted candidate CSV
```

### Data Flow

```
Raw JSONL Streams → Honeypot / Exp Filter (Stage 1) → High-Recall Keywords
                                                              ↓
Final Shortlist (CSV) ← Sort & Tie-break ← Score Modifier ← ONNX Embedding (Stage 2)
```

---

## Performance Analytics

The pipeline is optimized for standard CPU execution. Profiling on the **100,000 candidate dataset** yielded:

* **Model Initialization**: `0.437s` (Vastly reduced memory overhead by eliminating PyTorch imports)
* **Stage 1 Fast Filter**: `5.341s` (Processed at a throughput of **18,721.6 candidates/sec**)
* **Stage 2 Batch Encoding**: `72.058s` (Inference on 6,008 retrieved candidates using INT8 quantized ONNX, max-sequence length `64`, and batch size `512`)
* **Scoring & Sorting**: `0.761s`
* **Total Runtime**: **78.63 seconds** (7.2x speedup compared to the 512-second PyTorch baseline, well within the 5-minute/300s sandbox limit)

---

## Troubleshooting

### Missing Model Weights
**Error:** `FileNotFoundError: No ONNX model found at 'models/all-MiniLM-L6-v2'...`
* **Solution:** Verify that `models/all-MiniLM-L6-v2/model_quantized.onnx` is present in your working copy.

### Command Line Arguments Missing
**Error:** `rank.py: error: the following arguments are required: --candidates, --out`
* **Solution:** Ensure you specify the paths: `python rank.py --candidates <path_to_jsonl> --out <output_path>`.
