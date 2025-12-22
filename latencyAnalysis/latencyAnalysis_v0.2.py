import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Interview Latency Analytics_v0.2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def init_supabase_connection():
    """Initialize Supabase connection"""
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            return None, "Missing SUPABASE_URL or SUPABASE_KEY environment variables"
        
        supabase: Client = create_client(url, key)
        return supabase, None
    except Exception as e:
        return None, str(e)


def check_connection(supabase: Client, debug: bool = False) -> tuple[bool, str]:
    """Check if the database connection is active"""
    try:
        # Try a simple query to check connection
        response = supabase.table("interview_latency_metrics").select("metric_id").limit(1).execute()
        if debug:
            return True, f"Connection successful. Sample query returned {len(response.data) if response.data else 0} records."
        return True, "Connected"
    except Exception as e:
        error_msg = str(e)
        if debug:
            return False, f"Connection failed: {error_msg}"
        return False, error_msg


def run_diagnostic_queries(supabase: Client) -> dict:
    """Run diagnostic queries to understand the database state"""
    results = {}
    
    # Test 1: Simple count query
    try:
        response = supabase.table("interview_latency_metrics").select("*", count="exact").execute()
        results['count_query'] = {
            'success': True,
            'count': response.count if hasattr(response, 'count') else 'N/A',
            'records_returned': len(response.data) if response.data else 0
        }
    except Exception as e:
        results['count_query'] = {'success': False, 'error': str(e)}
    
    # Test 2: Query without any filters
    try:
        response = supabase.table("interview_latency_metrics").select("*").limit(10).execute()
        results['simple_query'] = {
            'success': True,
            'records': len(response.data) if response.data else 0,
            'data': response.data[:2] if response.data else None  # First 2 records
        }
    except Exception as e:
        results['simple_query'] = {'success': False, 'error': str(e)}
    
    # Test 3: Query with just one field
    try:
        response = supabase.table("interview_latency_metrics").select("metric_id").limit(5).execute()
        results['single_field_query'] = {
            'success': True,
            'records': len(response.data) if response.data else 0
        }
    except Exception as e:
        results['single_field_query'] = {'success': False, 'error': str(e)}
    
    # Test 4: Check current user and key info
    try:
        # Get the URL and key type
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        results['auth_info'] = {
            'success': True,
            'url_configured': bool(url),
            'key_configured': bool(key),
            'key_prefix': key[:20] + "..." if len(key) > 20 else "key too short",
            'note': 'Verify you are using the correct key (anon key for public access, or service_role for admin)'
        }
    except Exception as e:
        results['auth_info'] = {'success': False, 'error': str(e)}
    
    return results


@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_latency_data(_supabase: Client, days_back: int = 30, debug: bool = False) -> pd.DataFrame:
    """Fetch latency metrics from Supabase"""
    try:
        # Calculate the date range
        start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        if debug:
            st.write(f"**Debug Info:**")
            st.write(f"- Start date filter: {start_date}")
            st.write(f"- Days back: {days_back}")
            st.write(f"- Current time: {datetime.now().isoformat()}")
        
        # Build query step by step
        query = _supabase.table("interview_latency_metrics").select("*")
        
        # Only apply date filter if days_back is reasonable
        if days_back < 9000:
            query = query.gte("created_at", start_date)
        
        # Order and execute
        query = query.order("created_at", desc=True)
        
        # Execute with a reasonable limit first
        if debug:
            st.write("- Executing query...")
        
        response = query.execute()
        
        if debug:
            st.write(f"- Query executed successfully")
            st.write(f"- Response type: {type(response)}")
            st.write(f"- Response has 'data' attribute: {hasattr(response, 'data')}")
            st.write(f"- Response has 'count' attribute: {hasattr(response, 'count')}")
            
            if hasattr(response, 'count'):
                st.write(f"- Response count: {response.count}")
            
            if hasattr(response, 'data'):
                st.write(f"- Response.data type: {type(response.data)}")
                st.write(f"- Records returned: {len(response.data) if response.data else 0}")
                
                if response.data and len(response.data) > 0:
                    sample = response.data[0]
                    st.write(f"- Sample record created_at: {sample.get('created_at', 'N/A')}")
                    st.write(f"- Sample record keys: {list(sample.keys())}")
                    st.json(sample, expanded=False)
        
        if response.data and len(response.data) > 0:
            df = pd.DataFrame(response.data)
            
            if debug:
                st.write(f"- DataFrame created successfully")
                st.write(f"- DataFrame shape: {df.shape}")
                st.write(f"- DataFrame columns: {list(df.columns)}")
                st.write(f"- DataFrame head:")
                st.dataframe(df.head(2))
            
            # Convert timestamps - use format='mixed' to handle varying formats
            df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
            df['started_at'] = pd.to_datetime(df['started_at'], format='mixed', utc=True)
            df['completed_at'] = pd.to_datetime(df['completed_at'], format='mixed', utc=True)
            df['latency_ms'] = pd.to_numeric(df['latency_ms'])
            
            if debug:
                st.write(f"- Date range in data: {df['created_at'].min()} to {df['created_at'].max()}")
                st.write(f"- Timestamp parsing successful")
            
            return df
        else:
            if debug:
                st.write("- No data in response")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        if debug:
            import traceback
            st.write("**Full Traceback:**")
            st.code(traceback.format_exc())
        return pd.DataFrame()


def display_connection_status(is_connected: bool):
    """Display the database connection status"""
    if is_connected:
        st.success("✓ Connected to Supabase")
    else:
        st.error("✗ Disconnected from Supabase")


# =============================================================================
# CHART FUNCTIONS - Organized by complexity level
# =============================================================================

def create_step_avg_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Simple horizontal bar chart showing average latency per step"""
    step_avg = df.groupby('step_name')['latency_ms'].mean().sort_values(ascending=True)
    
    # Color bars based on latency (green=fast, yellow=medium, red=slow)
    colors = []
    for val in step_avg.values:
        if val < step_avg.median() * 0.5:
            colors.append('#28a745')  # Green - fast
        elif val < step_avg.median() * 1.5:
            colors.append('#ffc107')  # Yellow - medium
        else:
            colors.append('#dc3545')  # Red - slow
    
    fig = go.Figure(go.Bar(
        x=step_avg.values,
        y=step_avg.index,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.0f} ms' for v in step_avg.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Average Latency by Step",
        xaxis_title="Average Latency (ms)",
        yaxis_title="",
        height=max(300, len(step_avg) * 40),
        margin=dict(l=10, r=100, t=40, b=40),
        showlegend=False
    )
    return fig


def create_total_time_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Pie chart showing where total time is spent"""
    step_total = df.groupby('step_name')['latency_ms'].sum().sort_values(ascending=False)
    
    fig = px.pie(
        values=step_total.values,
        names=step_total.index,
        title="Total Time Distribution by Step",
        hole=0.4  # Donut chart
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_step_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing mean, median, and p95 for each step"""
    step_stats = df.groupby('step_name')['latency_ms'].agg(['mean', 'median', lambda x: x.quantile(0.95)])
    step_stats.columns = ['Mean', 'Median', 'P95']
    step_stats = step_stats.sort_values('Mean', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Mean', x=step_stats.index, y=step_stats['Mean'], marker_color='#007bff'))
    fig.add_trace(go.Bar(name='Median', x=step_stats.index, y=step_stats['Median'], marker_color='#28a745'))
    fig.add_trace(go.Bar(name='P95', x=step_stats.index, y=step_stats['P95'], marker_color='#dc3545'))
    
    fig.update_layout(
        title="Latency Statistics by Step (Mean vs Median vs P95)",
        xaxis_title="Step",
        yaxis_title="Latency (ms)",
        barmode='group',
        height=400,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_latency_distribution_chart(df: pd.DataFrame, step_name: str = None) -> go.Figure:
    """Histogram showing latency distribution"""
    if step_name:
        data = df[df['step_name'] == step_name]['latency_ms']
        title = f"Latency Distribution: {step_name}"
    else:
        data = df['latency_ms']
        title = "Overall Latency Distribution"
    
    fig = px.histogram(
        data,
        nbins=50,
        title=title,
        labels={'value': 'Latency (ms)', 'count': 'Frequency'}
    )
    fig.update_layout(height=350, showlegend=False)
    fig.update_traces(marker_color='#007bff')
    return fig


def create_latency_box_plot(df: pd.DataFrame) -> go.Figure:
    """Box plot showing latency distribution by step with outliers"""
    # Sort steps by median latency
    step_order = df.groupby('step_name')['latency_ms'].median().sort_values(ascending=False).index.tolist()
    
    fig = px.box(
        df,
        x="step_name",
        y="latency_ms",
        color="status",
        title="Latency Distribution by Step (with outliers)",
        labels={"latency_ms": "Latency (ms)", "step_name": "Step Name"},
        color_discrete_map={
            "success": "#28a745",
            "error": "#dc3545",
            "skipped": "#ffc107"
        },
        category_orders={"step_name": step_order}
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart showing latency trends over time"""
    df_copy = df.copy()
    df_copy['date'] = df_copy['created_at'].dt.date
    daily_avg = df_copy.groupby(['date', 'step_name'])['latency_ms'].mean().reset_index()
    
    fig = px.line(
        daily_avg,
        x="date",
        y="latency_ms",
        color="step_name",
        title="Latency Trends Over Time",
        labels={"latency_ms": "Average Latency (ms)", "date": "Date", "step_name": "Step"}
    )
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig


def create_daily_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing daily request volume"""
    df_copy = df.copy()
    df_copy['date'] = df_copy['created_at'].dt.date
    daily_counts = df_copy.groupby('date').size().reset_index(name='count')
    
    fig = px.bar(
        daily_counts,
        x='date',
        y='count',
        title="Daily Request Volume",
        labels={'count': 'Number of Operations', 'date': 'Date'}
    )
    fig.update_traces(marker_color='#007bff')
    fig.update_layout(height=300)
    return fig


def create_turn_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of latency by turn number and step"""
    pivot_df = df.pivot_table(
        values='latency_ms',
        index='step_name',
        columns='turn_number',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot_df,
        title="Latency Heatmap: Step vs Turn Number",
        labels={"x": "Turn Number", "y": "Step Name", "color": "Avg Latency (ms)"},
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig.update_layout(height=400)
    return fig


def create_percentile_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart showing percentile distribution for each step"""
    percentiles = [50, 75, 90, 95, 99]
    step_names = df['step_name'].unique()
    
    data = []
    for step in step_names:
        step_data = df[df['step_name'] == step]['latency_ms']
        for p in percentiles:
            data.append({
                'step': step,
                'percentile': f'P{p}',
                'latency': step_data.quantile(p/100)
            })
    
    perc_df = pd.DataFrame(data)
    
    fig = px.line(
        perc_df,
        x='percentile',
        y='latency',
        color='step',
        markers=True,
        title="Percentile Distribution by Step"
    )
    fig.update_layout(
        height=400,
        xaxis_title="Percentile",
        yaxis_title="Latency (ms)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig


def create_session_timeline_chart(df: pd.DataFrame, session_id: str) -> go.Figure:
    """Gantt-style timeline for a specific session"""
    session_df = df[df['session_id'] == session_id].copy()
    session_df = session_df.sort_values('started_at')
    
    fig = px.timeline(
        session_df,
        x_start="started_at",
        x_end="completed_at",
        y="step_name",
        color="status",
        title=f"Session Timeline",
        color_discrete_map={
            "success": "#28a745",
            "error": "#dc3545",
            "skipped": "#ffc107"
        }
    )
    fig.update_layout(height=350)
    return fig


def create_session_waterfall_chart(df: pd.DataFrame, session_id: str) -> go.Figure:
    """Waterfall chart showing cumulative time for a session"""
    session_df = df[df['session_id'] == session_id].copy()
    session_df = session_df.sort_values('started_at')
    
    fig = go.Figure(go.Waterfall(
        name="Latency",
        orientation="v",
        x=session_df['step_name'],
        y=session_df['latency_ms'],
        textposition="outside",
        text=[f'{v:.0f}' for v in session_df['latency_ms']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Session Latency Breakdown (Cumulative)",
        xaxis_title="Step",
        yaxis_title="Latency (ms)",
        height=350,
        xaxis_tickangle=-45
    )
    return fig


def create_error_rate_by_step_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing error rate by step"""
    error_rates = df.groupby('step_name').apply(
        lambda x: (x['status'] == 'error').mean() * 100
    ).sort_values(ascending=False)
    
    colors = ['#dc3545' if v > 5 else '#ffc107' if v > 1 else '#28a745' for v in error_rates.values]
    
    fig = go.Figure(go.Bar(
        x=error_rates.index,
        y=error_rates.values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in error_rates.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Error Rate by Step",
        xaxis_title="Step",
        yaxis_title="Error Rate (%)",
        height=350,
        xaxis_tickangle=-45
    )
    return fig


def main():
    st.title("Interview Latency Analytics Dashboard")
    st.markdown("Analyze and optimize interview process performance")
    
    # Initialize Supabase connection
    supabase, error = init_supabase_connection()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Debug mode (moved to top)
        debug_mode = st.checkbox("Enable Debug Mode", value=False, help="Show detailed debugging information")
        run_diagnostics = st.checkbox("Run Diagnostic Tests", value=False, help="Run comprehensive database tests")
        
        st.divider()
        
        # Connection status
        st.subheader("Database Status")
        if supabase:
            is_connected, conn_msg = check_connection(supabase, debug=debug_mode)
            display_connection_status(is_connected)
            if debug_mode:
                st.info(conn_msg)
        else:
            display_connection_status(False)
            st.error(f"Connection Error: {error}")
            st.info("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
        
        st.divider()
        
        # Date range selector
        st.subheader("Date Range")
        days_back = st.slider("Days to analyze", min_value=1, max_value=365, value=30)
        
        # Option to fetch all data
        fetch_all = st.checkbox("Fetch all data (ignore date filter)", value=False)
        
        # Refresh button
        if st.button("Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    if supabase:
        conn_status, _ = check_connection(supabase, debug=False)
        
        if not conn_status:
            st.error("Unable to connect to database")
            return
        
        # Run diagnostic tests if requested
        if run_diagnostics:
            st.subheader("Diagnostic Tests")
            with st.spinner("Running diagnostic queries..."):
                diag_results = run_diagnostic_queries(supabase)
            
            for test_name, result in diag_results.items():
                with st.expander(f"Test: {test_name}", expanded=True):
                    if result.get('success'):
                        st.success("Test passed")
                        st.json(result)
                    else:
                        st.error("Test failed")
                        st.json(result)
            
            st.divider()
        
        # Fetch data
        if debug_mode:
            st.subheader("Debug Information")
        
        # If fetch_all is enabled, use a very large days_back value
        effective_days_back = 10000 if fetch_all else days_back
        
        df = fetch_latency_data(supabase, effective_days_back, debug=debug_mode)
        
        if df.empty:
            st.warning("No data found for the selected date range.")
            
            # Try to fetch without date filter for debugging
            if not fetch_all:
                st.info("Attempting to fetch all records to diagnose the issue...")
                df_all = fetch_latency_data(supabase, 10000, debug=True)
                
                if not df_all.empty:
                    oldest_date = df_all['created_at'].min()
                    newest_date = df_all['created_at'].max()
                    st.warning(f"Found {len(df_all)} total records in database.")
                    st.info(f"Date range in database: {oldest_date.strftime('%Y-%m-%d')} to {newest_date.strftime('%Y-%m-%d')}")
                    st.info(f"Try enabling 'Fetch all data' option in the sidebar or increase the days slider.")
                else:
                    st.error("No data found in interview_latency_metrics table.")
            
            return
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Performance Overview", "🔍 Session Deep Dive", "❌ Error Analysis"])
        
        with tab1:
            # =================================================================
            # PERFORMANCE OVERVIEW TAB
            # Focus: High-level metrics, time distribution, and trends
            # =================================================================
            st.header("Performance Overview")
            st.caption("High-level metrics and performance insights")
            
            # Key Metrics Row
            col1, col2, col3, col4, col5 = st.columns(5)
        
            with col1:
                total_time = df['latency_ms'].sum()
                st.metric(
                    "Total Time",
                    f"{total_time/1000:.1f}s" if total_time > 1000 else f"{total_time:.0f}ms",
                    help="Total latency across all operations"
                )
        
            with col2:
                avg_latency = df['latency_ms'].mean()
                st.metric(
                    "Avg Latency",
                    f"{avg_latency:.1f} ms",
                    help="Average latency per operation"
                )
        
            with col3:
                p95_latency = df['latency_ms'].quantile(0.95)
                st.metric(
                    "P95 Latency",
                    f"{p95_latency:.1f} ms",
                    help="95% of operations complete within this time"
                )
        
            with col4:
                success_rate = (df['status'] == 'success').mean() * 100
                delta_color = "normal" if success_rate >= 95 else "inverse"
                st.metric(
                    "Success Rate",
                    f"{success_rate:.1f}%",
                    delta="Good" if success_rate >= 95 else "Needs attention",
                    delta_color=delta_color,
                    help="Percentage of successful operations"
                )
        
            with col5:
                unique_sessions = df['session_id'].nunique()
                st.metric(
                    "Sessions",
                    f"{unique_sessions:,}",
                    help="Number of unique interview sessions"
                )
        
            # Quick insight
            slowest_step = df.groupby('step_name')['latency_ms'].mean().idxmax()
            slowest_avg = df.groupby('step_name')['latency_ms'].mean().max()
            st.info(f"Slowest step on average: **{slowest_step}** ({slowest_avg:.0f} ms)")
        
            st.divider()
        
            # =================================================================
            # TIME ANALYSIS - Where is time being spent?
            # =================================================================
            st.subheader("Time Distribution")
            st.caption("Understand where time is being spent in your interview process")
        
            col1, col2 = st.columns([3, 2])
        
            with col1:
                st.plotly_chart(create_step_avg_bar_chart(df), use_container_width=True)
        
            with col2:
                st.plotly_chart(create_total_time_breakdown_chart(df), use_container_width=True)
        
            # Quick stats table
            st.subheader("Step Performance Summary")
            step_summary = df.groupby('step_name').agg({
                'latency_ms': ['mean', 'count'],
                'status': lambda x: (x == 'success').mean() * 100
            }).round(1)
            step_summary.columns = ['Avg Latency (ms)', 'Call Count', 'Success Rate (%)']
            step_summary = step_summary.sort_values('Avg Latency (ms)', ascending=False)
            step_summary['% of Total Time'] = (
                df.groupby('step_name')['latency_ms'].sum() / df['latency_ms'].sum() * 100
            ).round(1)
            st.dataframe(step_summary, use_container_width=True)
        
            st.divider()
        
            # =================================================================
            # PERFORMANCE COMPARISON
            # =================================================================
            st.subheader("Performance Comparison")
            st.caption("Compare different statistical measures across steps")
        
            st.plotly_chart(create_step_comparison_chart(df), use_container_width=True)
        
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_daily_volume_chart(df), use_container_width=True)
            with col2:
                st.plotly_chart(create_error_rate_by_step_chart(df), use_container_width=True)
        
            st.divider()
        
            # =================================================================
            # TRENDS & PATTERNS
            # =================================================================
            st.subheader("Performance Trends")
            st.caption("Track performance changes over time")
        
            st.plotly_chart(create_trend_chart(df), use_container_width=True)
        
            # Heatmap for turn analysis
            if df['turn_number'].nunique() > 1:
                st.subheader("Turn Number Analysis")
                st.caption("How latency changes across conversation turns")
                st.plotly_chart(create_turn_heatmap(df), use_container_width=True)
        
            st.divider()
        
            # =================================================================
            # DISTRIBUTION ANALYSIS
            # =================================================================
            st.subheader("Latency Distribution")
            st.caption("Understand the spread and outliers in your latency data")
        
            col1, col2 = st.columns([2, 1])
        
            with col1:
                st.plotly_chart(create_latency_box_plot(df), use_container_width=True)
        
            with col2:
                selected_step_dist = st.selectbox(
                    "Select step for histogram",
                    options=['All Steps'] + list(df['step_name'].unique()),
                    key="dist_step"
                )
                if selected_step_dist == 'All Steps':
                    st.plotly_chart(create_latency_distribution_chart(df), use_container_width=True)
                else:
                    st.plotly_chart(create_latency_distribution_chart(df, selected_step_dist), use_container_width=True)
        
            # Percentile analysis
            st.plotly_chart(create_percentile_chart(df), use_container_width=True)
        
            # Detailed statistics table
            with st.expander("Detailed Statistics Table"):
                detailed_stats = df.groupby('step_name')['latency_ms'].agg([
                    ('Mean', 'mean'),
                    ('Median', 'median'),
                    ('Std Dev', 'std'),
                    ('Min', 'min'),
                    ('Max', 'max'),
                    ('P50', lambda x: x.quantile(0.5)),
                    ('P75', lambda x: x.quantile(0.75)),
                    ('P90', lambda x: x.quantile(0.9)),
                    ('P95', lambda x: x.quantile(0.95)),
                    ('P99', lambda x: x.quantile(0.99)),
                    ('Count', 'count')
                ]).round(2)
                st.dataframe(detailed_stats, use_container_width=True)
        
        with tab2:
            # =================================================================
            # SESSION DEEP DIVE TAB
            # Focus: Individual session analysis and comparison
            # =================================================================
            st.header("Session Deep Dive")
            st.caption("Analyze individual interview sessions in detail")
            
            # Session selector with better formatting
            sessions = df.groupby('session_id').agg({
                'created_at': 'min',
                'latency_ms': 'sum',
                'status': lambda x: (x == 'success').mean() * 100
            }).reset_index()
            sessions.columns = ['session_id', 'started', 'total_latency', 'success_rate']
            sessions = sessions.sort_values('started', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_session = st.selectbox(
                    "Select a session to analyze",
                    options=sessions['session_id'].tolist(),
                    format_func=lambda x: f"{x[:8]}... | {sessions[sessions['session_id']==x]['started'].iloc[0].strftime('%Y-%m-%d %H:%M')} | {sessions[sessions['session_id']==x]['total_latency'].iloc[0]:.0f}ms total"
                )
            with col2:
                session_info = sessions[sessions['session_id'] == selected_session].iloc[0]
                st.metric("Session Success Rate", f"{session_info['success_rate']:.0f}%")
            
            if selected_session:
                session_df = df[df['session_id'] == selected_session]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Latency", f"{session_df['latency_ms'].sum():.0f} ms")
                with col2:
                    st.metric("Steps Executed", len(session_df))
                with col3:
                    st.metric("Avg per Step", f"{session_df['latency_ms'].mean():.0f} ms")
                with col4:
                    errors = (session_df['status'] == 'error').sum()
                    st.metric("Errors", errors, delta="None" if errors == 0 else None, delta_color="normal")
                
                st.divider()
                
                # Session visualizations
                st.subheader("Session Timeline")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_session_timeline_chart(df, selected_session), use_container_width=True)
                with col2:
                    st.plotly_chart(create_session_waterfall_chart(df, selected_session), use_container_width=True)
                
                st.divider()
                
                # Session details table
                st.subheader("Session Step Details")
                session_display = session_df[['turn_number', 'step_name', 'latency_ms', 'status', 'started_at', 'error_message']].copy()
                session_display = session_display.sort_values('started_at')
                session_display['latency_ms'] = session_display['latency_ms'].round(1)
                st.dataframe(session_display, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # =================================================================
            # SESSION COMPARISON
            # =================================================================
            st.subheader("Session Comparison")
            st.caption("Compare sessions to identify patterns")
            
            # Show top slowest and fastest sessions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Slowest Sessions**")
                slowest = sessions.nlargest(5, 'total_latency')[['session_id', 'started', 'total_latency', 'success_rate']]
                slowest['session_id'] = slowest['session_id'].str[:12]
                slowest['total_latency'] = slowest['total_latency'].round(0).astype(int)
                slowest.columns = ['Session', 'Started', 'Total (ms)', 'Success %']
                st.dataframe(slowest, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Fastest Sessions**")
                fastest = sessions.nsmallest(5, 'total_latency')[['session_id', 'started', 'total_latency', 'success_rate']]
                fastest['session_id'] = fastest['session_id'].str[:12]
                fastest['total_latency'] = fastest['total_latency'].round(0).astype(int)
                fastest.columns = ['Session', 'Started', 'Total (ms)', 'Success %']
                st.dataframe(fastest, use_container_width=True, hide_index=True)
            
            # Session latency distribution
            st.subheader("Session Latency Distribution")
            session_totals = df.groupby('session_id')['latency_ms'].sum().reset_index()
            session_totals.columns = ['session_id', 'total_latency']
            
            fig = px.histogram(
                session_totals,
                x='total_latency',
                nbins=30,
                title="Distribution of Total Session Latency",
                labels={'total_latency': 'Total Latency (ms)', 'count': 'Number of Sessions'}
            )
            fig.update_traces(marker_color='#007bff')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # =================================================================
            # RAW DATA EXPLORER
            # =================================================================
            with st.expander("📥 Raw Data Explorer & Export"):
                st.caption("Filter and export raw data for custom analysis")
                
                # Filters
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    step_filter = st.multiselect(
                        "Filter by Step",
                        options=df['step_name'].unique(),
                        default=[]
                    )
                
                with col2:
                    status_filter = st.multiselect(
                        "Filter by Status",
                        options=df['status'].unique(),
                        default=[]
                    )
                
                with col3:
                    min_latency = st.number_input("Min Latency (ms)", min_value=0, value=0)
                
                with col4:
                    max_latency = st.number_input("Max Latency (ms)", min_value=0, value=int(df['latency_ms'].max()) + 1)
                
                # Apply filters
                filtered_df = df.copy()
                if step_filter:
                    filtered_df = filtered_df[filtered_df['step_name'].isin(step_filter)]
                if status_filter:
                    filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
                filtered_df = filtered_df[
                    (filtered_df['latency_ms'] >= min_latency) & 
                    (filtered_df['latency_ms'] <= max_latency)
                ]
                
                st.write(f"Showing {len(filtered_df):,} of {len(df):,} records")
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Filtered Data (CSV)",
                        data=csv,
                        file_name=f"latency_metrics_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with col2:
                    csv_all = df.to_csv(index=False)
                    st.download_button(
                        label="Download All Data (CSV)",
                        data=csv_all,
                        file_name=f"latency_metrics_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with tab3:
            # =================================================================
            # ERROR ANALYSIS TAB
            # Focus: Error tracking, debugging, and root cause analysis
            # =================================================================
            st.header("Error Analysis")
            st.caption("Identify, investigate, and troubleshoot failures")
            
            error_df = df[df['status'] == 'error']
            
            # Error Metrics Overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Errors", f"{len(error_df):,}")
            with col2:
                error_rate = len(error_df) / len(df) * 100
                delta_color = "normal" if error_rate < 5 else "inverse"
                st.metric("Error Rate", f"{error_rate:.2f}%", delta_color=delta_color)
            with col3:
                if not error_df.empty:
                    most_errors = error_df['step_name'].value_counts().idxmax()
                    st.metric("Most Failing Step", most_errors)
                else:
                    st.metric("Most Failing Step", "None")
            with col4:
                if not error_df.empty:
                    failed_sessions = error_df['session_id'].nunique()
                    st.metric("Sessions with Errors", f"{failed_sessions:,}")
                else:
                    st.metric("Sessions with Errors", "0")
            
            st.divider()
            
            if not error_df.empty:
                # =================================================================
                # ERROR BREAKDOWN BY STEP AND MESSAGE
                # =================================================================
                st.subheader("Error Breakdown")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Errors by Step**")
                    error_by_step = error_df['step_name'].value_counts().reset_index()
                    error_by_step.columns = ['Step', 'Error Count']
                    fig = px.bar(
                        error_by_step,
                        x='Step',
                        y='Error Count',
                        title="Error Count by Step",
                        color='Error Count',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=350, xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Error Rate by Step**")
                    error_rate_by_step = df.groupby('step_name').apply(
                        lambda x: (x['status'] == 'error').mean() * 100
                    ).sort_values(ascending=False).reset_index()
                    error_rate_by_step.columns = ['Step', 'Error Rate (%)']
                    fig = px.bar(
                        error_rate_by_step,
                        x='Step',
                        y='Error Rate (%)',
                        title="Error Rate by Step",
                        color='Error Rate (%)',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=350, xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # =================================================================
                # ERROR MESSAGES AND PATTERNS
                # =================================================================
                st.subheader("Error Messages")
                st.caption("Detailed error message breakdown")
                
                # Group errors by step and message
                error_details = error_df.groupby(['step_name', 'error_message']).agg({
                    'metric_id': 'count',
                    'session_id': 'nunique',
                    'latency_ms': 'mean'
                }).reset_index()
                error_details.columns = ['Step', 'Error Message', 'Occurrences', 'Affected Sessions', 'Avg Latency (ms)']
                error_details = error_details.sort_values('Occurrences', ascending=False)
                error_details['Avg Latency (ms)'] = error_details['Avg Latency (ms)'].round(1)
                
                st.dataframe(error_details, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # =================================================================
                # ERROR TIMELINE AND TRENDS
                # =================================================================
                st.subheader("Error Timeline")
                st.caption("Track error patterns over time")
                
                error_df_copy = error_df.copy()
                error_df_copy['date'] = error_df_copy['created_at'].dt.date
                
                # Daily error count
                error_by_date = error_df_copy.groupby(['date', 'step_name']).size().reset_index(name='count')
                
                fig = px.line(
                    error_by_date,
                    x='date',
                    y='count',
                    color='step_name',
                    title="Error Trends Over Time",
                    labels={'count': 'Error Count', 'date': 'Date', 'step_name': 'Step'},
                    markers=True
                )
                fig.update_layout(
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Error heatmap by day of week and hour (if enough data)
                if len(error_df) > 20:
                    st.subheader("Error Occurrence Pattern")
                    error_df_copy['hour'] = error_df_copy['created_at'].dt.hour
                    error_df_copy['day_of_week'] = error_df_copy['created_at'].dt.day_name()
                    
                    error_heatmap = error_df_copy.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
                    
                    # Order days of week correctly
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    error_pivot = error_heatmap.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
                    error_pivot = error_pivot.reindex(day_order)
                    
                    fig = px.imshow(
                        error_pivot,
                        title="Error Occurrence Heatmap (Day of Week vs Hour)",
                        labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Error Count'},
                        color_continuous_scale='Reds',
                        aspect='auto'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # =================================================================
                # SESSIONS WITH ERRORS
                # =================================================================
                st.subheader("Sessions with Errors")
                st.caption("Identify problematic sessions for investigation")
                
                # Get sessions that have errors
                error_sessions = error_df.groupby('session_id').agg({
                    'metric_id': 'count',
                    'created_at': 'min',
                    'step_name': lambda x: ', '.join(x.unique())
                }).reset_index()
                error_sessions.columns = ['Session ID', 'Error Count', 'First Error Time', 'Failed Steps']
                error_sessions = error_sessions.sort_values('Error Count', ascending=False)
                error_sessions['Session ID'] = error_sessions['Session ID'].str[:12] + "..."
                
                st.dataframe(error_sessions.head(20), use_container_width=True, hide_index=True)
                
                st.divider()
                
                # =================================================================
                # ERROR CORRELATION ANALYSIS
                # =================================================================
                st.subheader("Error Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Error Latency Analysis**")
                    st.caption("Do errors take longer to fail?")
                    
                    # Compare latency of errors vs success
                    latency_comparison = df.groupby('status')['latency_ms'].mean().reset_index()
                    latency_comparison.columns = ['Status', 'Avg Latency (ms)']
                    
                    fig = px.bar(
                        latency_comparison,
                        x='Status',
                        y='Avg Latency (ms)',
                        title="Average Latency by Status",
                        color='Status',
                        color_discrete_map={
                            'success': '#28a745',
                            'error': '#dc3545',
                            'skipped': '#ffc107'
                        }
                    )
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Turn Number Impact**")
                    st.caption("Do errors happen at specific turns?")
                    
                    # Error rate by turn number
                    turn_error_rate = df.groupby('turn_number').apply(
                        lambda x: (x['status'] == 'error').mean() * 100
                    ).reset_index()
                    turn_error_rate.columns = ['Turn Number', 'Error Rate (%)']
                    
                    fig = px.line(
                        turn_error_rate,
                        x='Turn Number',
                        y='Error Rate (%)',
                        title="Error Rate by Turn Number",
                        markers=True
                    )
                    fig.update_traces(line_color='#dc3545', marker_color='#dc3545')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # =================================================================
                # ERROR FILTERING AND EXPORT
                # =================================================================
                with st.expander("🔍 Filter and Export Error Data"):
                    st.caption("Filter errors for detailed investigation")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        error_step_filter = st.multiselect(
                            "Filter by Step",
                            options=error_df['step_name'].unique(),
                            default=[]
                        )
                    
                    with col2:
                        # Get unique error messages (truncated)
                        error_messages = error_df['error_message'].unique()
                        error_msg_filter = st.multiselect(
                            "Filter by Error Message",
                            options=error_messages,
                            default=[],
                            format_func=lambda x: x[:50] + "..." if len(x) > 50 else x
                        )
                    
                    with col3:
                        date_range = st.date_input(
                            "Filter by Date Range",
                            value=(error_df['created_at'].min().date(), error_df['created_at'].max().date()),
                            help="Select date range for errors"
                        )
                    
                    # Apply filters
                    filtered_errors = error_df.copy()
                    if error_step_filter:
                        filtered_errors = filtered_errors[filtered_errors['step_name'].isin(error_step_filter)]
                    if error_msg_filter:
                        filtered_errors = filtered_errors[filtered_errors['error_message'].isin(error_msg_filter)]
                    if len(date_range) == 2:
                        filtered_errors = filtered_errors[
                            (filtered_errors['created_at'].dt.date >= date_range[0]) &
                            (filtered_errors['created_at'].dt.date <= date_range[1])
                        ]
                    
                    st.write(f"Showing {len(filtered_errors):,} of {len(error_df):,} errors")
                    st.dataframe(filtered_errors, use_container_width=True, hide_index=True)
                    
                    # Export button
                    csv_errors = filtered_errors.to_csv(index=False)
                    st.download_button(
                        label="Download Error Data (CSV)",
                        data=csv_errors,
                        file_name=f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            else:
                st.success("🎉 No errors found in the selected date range!")
                st.info("Your system is running smoothly. All operations completed successfully.")
    
    else:
        st.error("Unable to connect to database")
        st.markdown("""
        ### Setup Instructions
        
        1. Create a `.env` file in the `Interview-latency-analysis` folder
        2. Add your Supabase credentials:
        
        ```
        SUPABASE_URL=your_supabase_project_url
        SUPABASE_KEY=your_supabase_anon_key
        ```
        
        3. Restart the Streamlit app
        """)


if __name__ == "__main__":
    main()
