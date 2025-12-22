# Interview Latency Analytics Dashboard

A Streamlit dashboard for visualizing and analyzing interview latency metrics from your Supabase database.

## Features

- **Real-time Connection Status**: Visual indicator showing database connectivity
- **Key Metrics**: Total records, average latency, P95 latency, success rate, unique sessions
- **Step Analysis**: Box plots and statistics showing latency distribution by step
- **Time Series Analysis**: Track latency trends over time
- **Heatmap Visualization**: View latency patterns across steps and turn numbers
- **Session Deep Dive**: Analyze individual interview sessions with timeline visualization
- **Error Analysis**: Identify and track error patterns
- **Raw Data Explorer**: Filter and download data for further analysis

## Setup

### 1. Install Dependencies

```bash
cd Interview-latency-analysis
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `Interview-latency-analysis` folder:

```bash
cp .env.example .env
```

Then edit the `.env` file with your Supabase credentials:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

You can find these values in your Supabase dashboard under **Settings > API**.

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Dashboard Sections

### Key Metrics
- **Total Records**: Number of latency records in the selected time range
- **Avg Latency**: Mean latency across all steps
- **P95 Latency**: 95th percentile latency (useful for SLA monitoring)
- **Success Rate**: Percentage of operations that completed successfully
- **Unique Sessions**: Number of distinct interview sessions

### Step Analysis
- Box plot showing latency distribution for each step
- Status breakdown (success/error/skipped) pie chart
- Detailed statistics table with mean, median, std dev, min, max, and count

### Trends Over Time
- Line chart showing average latency trends per step over time
- Helps identify performance degradation or improvements

### Latency Heatmap
- Visual representation of average latency by step and turn number
- Useful for identifying patterns in multi-turn conversations

### Session Deep Dive
- Select individual sessions to analyze
- Timeline visualization showing step execution order and duration
- Detailed table of all steps in the session

### Error Analysis
- Table of errors grouped by step and error message
- Helps identify common failure points

### Raw Data Explorer
- Filter data by step, status, and minimum latency
- Download filtered data as CSV for external analysis

## Database Schema

The dashboard connects to the `interview_latency_metrics` table with the following schema:

```sql
create table public.interview_latency_metrics (
  metric_id uuid not null default gen_random_uuid (),
  session_id uuid not null,
  turn_number integer not null default 0,
  step_name text not null,
  latency_ms numeric not null,
  started_at timestamp with time zone not null,
  completed_at timestamp with time zone not null,
  status text not null,
  error_message text null,
  metadata jsonb null,
  created_at timestamp with time zone null default now()
);
```

## Troubleshooting

### "Missing SUPABASE_URL or SUPABASE_KEY"
- Ensure you have created a `.env` file with valid credentials
- Check that the environment variables are named correctly

### "Connection Failed"
- Verify your Supabase URL is correct (should be `https://xxx.supabase.co`)
- Check that your API key has the necessary permissions
- Ensure your IP is not blocked by Supabase network policies

### "No data found"
- Check that the `interview_latency_metrics` table exists
- Verify that there is data in the selected date range
- Try increasing the "Days to analyze" slider

## Customization

You can modify `app.py` to:
- Add additional visualizations
- Change color schemes
- Add new filters or metrics
- Connect to additional tables for cross-referencing
