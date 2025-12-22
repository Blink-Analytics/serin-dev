# 🎯 JD Objective Ranker & Scorer

A Python tool that analyzes Job Description objectives and assigns **Importance Scores** (1-10) and **Difficulty Scores** (1-10) to each task. Choose between a fast modular approach or a high-accuracy cross-encoder model.

---

## 📂 Directory Structure

```
obj-imp-ranker/
├── cross_encoder_scorer.py      # High Accuracy Approach (Recommended)
├── modular_approach/            # Fast/Modular Approach
│   ├── main.py                  # Entry point
│   ├── config.py                # Configuration (weights, models)
│   ├── data_loader.py           # CSV/JSON parsing
│   ├── engine.py                # Vector generation
│   └── logic.py                 # Scoring logic
├── sources/                     # Input data folder
│   └── database.csv             # Raw input file
├── output/                      # Results folder
│   ├── scored_output.csv        # Modular approach results
│   └── cross_encoder_results.csv # Cross-encoder results
└── requirements.txt
```

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

**Required packages:**
- `sentence-transformers`
- `numpy`
- `pandas`

---

## 🏃 How to Run

### Option A: Fast Approach (Bi-Encoder)

This uses a **modular architecture** with separated components. Great for understanding the codebase and quick iterations.

```bash
cd modular_approach
python main.py
```

**Output:** `output/scored_output.csv`

---

### Option B: High Accuracy Approach (Cross-Encoder) ⭐⭐⭐ (this one had worked)

This uses a **Cross-Encoder model** that deeply understands context (e.g., knows "FSDP" relates to "Distributed Training").

```bash
python cross_encoder_scorer.py
```

**Output:** `output/cross_encoder_results.csv`

---

## 📥 Input / Output

### Input
Place your raw CSV file in the `sources/` folder:
- **File:** `sources/database.csv`
- **Expected columns:** Job descriptions with objectives/tasks

### Output
Results will be saved in the `output/` folder:
- `scored_output.csv` - Results from the modular approach
- `cross_encoder_results.csv` - Results from the cross-encoder approach

Each output includes:
- Original objective text
- **Importance Score** (1-10)
- **Difficulty Score** (1-10)

---

## 🧠 Architecture Comparison

| Feature | Fast Approach | High Accuracy Approach |
|---------|---------------|------------------------|
| Model Type | Bi-Encoder | Cross-Encoder |
| Speed | ⚡ Fast | 🐢 Slower |
| Accuracy | Good | 🎯 Excellent |
| Context Understanding | Basic | Deep semantic reasoning |
| Best For | Quick iterations | Production use |

---

## 📝 Notes

- The **Cross-Encoder** approach is recommended for production use due to better reasoning capabilities
- Make sure your input CSV is properly formatted
- Both approaches handle messy data through robust parsing in `data_loader.py`

---

## 🤝 Contributing

For questions or improvements, please contact the development team.