# LLM-Based Importance & Difficulty Scorer

Score job objectives using state-of-the-art open-source models via Groq.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd d:\Serin\serin-dev\imp-difficulty-ranker\v0.1.1
pip install -r requirements.txt
```

### 2. Get Groq API Key

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Create an API key
4. Copy the key (you'll enter it in the app)

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

This will open the app in your browser at `http://localhost:8501`

## 📋 How to Use

### Step 1: Configure (Sidebar)
1. **Provider**: Select "Groq" (only option in Phase 1)
2. **API Key**: Paste your Groq API key
3. **Model**: Choose from 7 open-source models:
   - `llama-3.3-70b-versatile` ⭐ Recommended (best accuracy)
   - `deepseek-r1-distill-llama-70b` 🧠 Reasoning-focused
   - `qwen2.5-72b-instruct` 🌍 Multilingual
   - `qwen2.5-7b-instruct` ⚡ Fast
   - `llama-3.1-8b-instant` ⚡⚡ Fastest
   - `mixtral-8x7b-32768` 📄 Long context
   - `gemma2-9b-it` 🔵 Google

### Step 2: Upload CSV
- Click "📁 Upload your CSV dataset"
- Upload your CSV file (e.g., `Serin Algo Evaluation - Algo_Eval_v0.1.csv`)
- App will auto-detect columns:
  - Job_Description (or similar)
  - Objective (or similar)

### Step 3: Configure Columns
- Verify/select the correct columns from dropdowns
- If you have ground truth labels, check the box and select:
  - Golden_Importance
  - Golden_Difficulty

### Step 4: Start Scoring
- Click "🚀 Start Scoring"
- Watch the progress bar and live metrics:
  - Rows completed
  - Average latency
  - Estimated time remaining
  - Errors count

### Step 5: Review Results
- Preview scored data (first 20 rows)
- View summary statistics:
  - Mean Importance
  - Mean Difficulty
  - Mean Absolute Error (if ground truth exists)
  - Average response time

### Step 6: Download
- Click "⬇️ Download Scored CSV"
- File will be named: `scored_results_YYYYMMDD_HHMMSS.csv`

## 📊 Output Columns

The app adds these new columns to your CSV:

| Column | Description |
|--------|-------------|
| `LLM_Importance` | Importance score (0-10) |
| `LLM_Difficulty` | Difficulty score (0-10) |
| `LLM_Response_Time_ms` | API latency in milliseconds |
| `LLM_Error` | Error message (null if success) |

If ground truth exists, also adds:
| Column | Description |
|--------|-------------|
| `Importance_Delta` | abs(LLM - Ground Truth) |
| `Difficulty_Delta` | abs(LLM - Ground Truth) |
| `MAE` | Mean Absolute Error |

## 🔧 Processing Settings

**Batch Size**: Number of rows to process before pausing (default: 30)
- Groq free tier: 30 requests/second
- Adjust based on your rate limits

**Max Retries**: Number of retry attempts for failed requests (default: 3)
- If LLM returns invalid JSON or API fails, retry up to 3 times
- Failed rows get `-1` scores and error message logged

## 🎯 Model Comparison

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| **llama-3.3-70b** | 70B | Medium | Best accuracy |
| **deepseek-r1** | 70B | Medium | Chain-of-thought reasoning |
| **qwen2.5-72b** | 72B | Medium | Multilingual tasks |
| qwen2.5-7b | 7B | Fast | Cost-efficient |
| llama-3.1-8b | 8B | Fastest | Testing/prototyping |
| mixtral-8x7b | 46B | Fast | Long context (32K) |
| gemma2-9b | 9B | Fast | Google ecosystem |

## 📝 Example Usage

```bash
# 1. Run the app
streamlit run streamlit_app.py

# 2. In the browser:
#    - Sidebar: Enter Groq API key
#    - Sidebar: Select "llama-3.3-70b-versatile"
#    - Main: Upload "Serin Algo Evaluation - Algo_Eval_v0.1.csv"
#    - Main: Verify columns (Job_Description, Objective)
#    - Main: Check "I have ground truth labels"
#    - Main: Select Golden_Importance, Golden_Difficulty
#    - Main: Click "Start Scoring"
#    - Wait for processing (~1-2 minutes for 30 rows)
#    - Download "scored_results_20251223_143022.csv"
```

## 🐛 Troubleshooting

### "API error: 401"
- Your API key is invalid or expired
- Get a new key from https://console.groq.com

### "API error: 429"
- You've hit rate limits
- Reduce batch size
- Wait 1 minute and try again

### "Invalid JSON response"
- LLM returned malformed JSON
- Will automatically retry (up to 3 times)
- If still fails, row will be flagged with error

### Scores are all -1
- Check error log (expandable section at bottom)
- Likely API key or connectivity issue

## 📂 File Structure

```
v0.1.1/
├── obj_sys_prompt.py       # System prompt generator (already exists)
├── llm_scorer.py            # Groq API integration (NEW)
├── streamlit_app.py         # Streamlit UI (NEW)
├── requirements.txt         # Dependencies (NEW)
└── README.md                # This file (NEW)
```

## 🚧 Coming Soon (Phase 2)

- [ ] Gemini integration (Google's models)
- [ ] Side-by-side model comparison
- [ ] Export metrics report (markdown)
- [ ] Pause/resume processing
- [ ] Real-time score visualization

## 💡 Tips

1. **Start with small batches**: Test with 5-10 rows first
2. **Use fastest model for testing**: llama-3.1-8b-instant
3. **Use best model for production**: llama-3.3-70b-versatile or deepseek-r1
4. **Monitor costs**: Groq free tier = 14,400 requests/day
5. **Compare models**: Run same dataset through multiple models to find best fit

## 📊 Performance Benchmarks

For 100 rows (estimated):

| Model | Time | Accuracy | Use Case |
|-------|------|----------|----------|
| llama-3.3-70b | ~3 min | ⭐⭐⭐⭐⭐ | Production |
| deepseek-r1 | ~3 min | ⭐⭐⭐⭐⭐ | High-stakes decisions |
| qwen2.5-72b | ~3 min | ⭐⭐⭐⭐ | Multilingual |
| llama-3.1-8b | ~1 min | ⭐⭐⭐ | Testing |

---

**Built with ❤️ using Streamlit + Groq**
