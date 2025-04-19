import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit.components.v1 import html

st.title("Actual vs Predicted Values Visualization")
st.write("Upload CSV files containing 'actual' and 'predicted' values to visualize the comparison")

# File uploaders
col1, col2, col3 = st.columns(3)
with col1:
    actual_file = st.file_uploader("Upload 'Actual' CSV file", type="csv", key="actual_file")
with col2:
    pred_file = st.file_uploader("Upload 'Predicted' CSV file", type="csv", key="pred_file")
with col3:
    equity_file = st.file_uploader("Upload 'Equity Curve' CSV file (optional)", type="csv", key="equity_file")

# Parse CSV files
@st.cache_data
def parse_csv(file):
    try:
        df = pd.read_csv(file)
        # If the CSV has multiple columns, we'll take the first column
        if len(df.columns) > 1:
            st.info(f"Multiple columns detected in file. Using the first column: {df.columns[0]}")
            return df.iloc[:, 0].values
        return df.iloc[:, 0].values
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")
        return None

# Initialize session state for the current point
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
    st.session_state.current_actual = 0
    st.session_state.current_predicted = 0
    st.session_state.current_error = 0
    st.session_state.current_equity = 0

# Hide the number input with CSS
st.markdown("""
<style>
[data-testid="stNumberInput"] {
    position: absolute;
    width: 0;
    height: 0;
    opacity: 0;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# Process data when files are uploaded
if actual_file and pred_file:
    actual_values = parse_csv(actual_file)
    pred_values = parse_csv(pred_file)
    
    # Load equity data if available
    has_equity_data = False
    equity_values = None
    if equity_file:
        equity_values = parse_csv(equity_file)
        has_equity_data = True
    
    # Check if the arrays have the same length
    if len(actual_values) != len(pred_values):
        st.error("Error: The 'actual' and 'predicted' data must have the same number of points")
    elif has_equity_data and len(actual_values) != len(equity_values):
        st.error("Error: The 'equity curve' data must have the same number of points as 'actual' and 'predicted' data")
    else:
        st.success(f"Files loaded successfully! Found {len(actual_values)} data points.")
        
        # Calculate error metrics
        errors = np.abs(actual_values - pred_values)
        mse = np.mean(np.square(actual_values - pred_values))
        rmse = np.sqrt(mse)
        mae = np.mean(errors)
        
        # Create animation frames
        frames = []
        for i in range(len(actual_values)):
            # Data for each frame
            actual_trace = actual_values[:i+1]
            pred_trace = pred_values[:i+1]
            
            if has_equity_data:
                # Create frames for subplots with equity curve
                equity_trace = equity_values[:i+1]
                
                frame_data = [
                    go.Scatter(x=list(range(i+1)), y=actual_trace, mode='lines', line=dict(color='blue')),
                    go.Scatter(x=list(range(i+1)), y=pred_trace, mode='lines', line=dict(color='red')),
                    go.Scatter(x=list(range(i+1)), y=equity_trace, mode='lines', line=dict(color='green'))
                ]
            else:
                # Original frame data without equity curve
                frame_data = [
                    go.Scatter(x=list(range(i+1)), y=actual_trace, mode='lines', line=dict(color='blue')),
                    go.Scatter(x=list(range(i+1)), y=pred_trace, mode='lines', line=dict(color='red'))
                ]
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
        
        # Create base figure
        if has_equity_data:
            # Create figure with subplots for main visualization and equity curve
            fig = make_subplots(rows=1, cols=2, 
                                subplot_titles=("Actual vs Predicted", "Equity Curve"),
                                specs=[[{"type": "scatter"}, {"type": "scatter"}]],
                                column_widths=[0.6, 0.4])
            
            # Add traces for main visualization
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[actual_values[0]],
                    mode='lines', name='Actual',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[pred_values[0]],
                    mode='lines', name='Predicted',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # Add trace for equity curve
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[equity_values[0]],
                    mode='lines', name='Equity',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
            
            # Update x-axis ranges
            fig.update_xaxes(title_text="Time", range=[0, len(actual_values)], row=1, col=1)
            fig.update_xaxes(title_text="Time", range=[0, len(actual_values)], row=1, col=2)
            
            # Update y-axis ranges
            y_min = min(min(actual_values), min(pred_values))
            y_max = max(max(actual_values), max(pred_values))
            buffer = (y_max - y_min) * 0.1
            fig.update_yaxes(title_text="Value", range=[y_min - buffer, y_max + buffer], row=1, col=1)
            
            equity_min = min(equity_values)
            equity_max = max(equity_values)
            equity_buffer = (equity_max - equity_min) * 0.1
            fig.update_yaxes(title_text="Equity", range=[equity_min - equity_buffer, equity_max + equity_buffer], row=1, col=2)
            
        else:
            # Original single plot
            fig = go.Figure(
                data=[
                    go.Scatter(x=[0], y=[actual_values[0]], mode='lines', name='Actual', line=dict(color='blue')),
                    go.Scatter(x=[0], y=[pred_values[0]], mode='lines', name='Predicted', line=dict(color='red'))
                ]
            )
            
            # Update axes ranges
            y_min = min(min(actual_values), min(pred_values))
            y_max = max(max(actual_values), max(pred_values))
            buffer = (y_max - y_min) * 0.1
            fig.update_layout(
                xaxis=dict(title='Time', range=[0, len(actual_values)]),
                yaxis=dict(title='Value', range=[y_min - buffer, y_max + buffer])
            )
        
        # Add frames and buttons
        fig.frames = frames
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=1.1,
                    yanchor="top"
                )
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Frame: "
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "mode": "immediate"
                            }
                        ],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(len(frames))
                ]
            }],
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add hidden number input to track current frame
        current_frame = st.number_input("Current Frame", min_value=0, max_value=len(actual_values)-1, 
                                      value=0, key="current_frame", label_visibility="hidden")
        
        # JavaScript to sync Plotly animation with Streamlit metrics
        js_code = """
        <script>
        const streamlitDoc = window.parent.document;
        
        function updateStreamlitFrame(frameNum) {
            // Find the hidden number input
            const numberInputs = streamlitDoc.querySelectorAll('input[type="number"]');
            let hiddenInput = null;
            
            for (const input of numberInputs) {
                if (input.style.opacity === '0' || input.style.display === 'none' || 
                    input.parentElement.style.opacity === '0' || 
                    input.closest('[data-testid="stNumberInput"]')?.style.opacity === '0') {
                    hiddenInput = input;
                    break;
                }
            }
            
            if (!hiddenInput) return;
            
            // Update value and dispatch events
            hiddenInput.value = frameNum;
            hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            
            // Force Streamlit to receive the update
            const changeEvent = new Event('change', { bubbles: true });
            hiddenInput.dispatchEvent(changeEvent);
        }
        
        // Listen for Plotly frame changes
        window.addEventListener('message', function(e) {
            const message = e.data;
            if (message && message.type === 'plotly_animate') {
                const frameNumber = parseInt(message.frameNumber) || 0;
                updateStreamlitFrame(frameNumber);
            }
        });
        </script>
        """
        
        # Display the figure and inject JS
        st.plotly_chart(fig, use_container_width=True)
        html(js_code)
        
        # Update session state values based on the current frame
        st.session_state.current_idx = current_frame
        st.session_state.current_actual = actual_values[current_frame]
        st.session_state.current_predicted = pred_values[current_frame]
        st.session_state.current_error = errors[current_frame]
        if has_equity_data:
            st.session_state.current_equity = equity_values[current_frame]
        
        # Removed the Data Summary section as requested
        
        # Overall metrics
        st.subheader("Overall Performance Metrics")
        overall_cols = st.columns(3)
        with overall_cols[0]:
            st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
        with overall_cols[1]:
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
        with overall_cols[2]:
            st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
        
        # Add custom CSS for improved appearance
        st.markdown("""
        <style>
        /* Improve metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: bold;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
        }
        
        /* Make the plot container responsive */
        .js-plotly-plot, .plotly, .plot-container {
            width: 100%;
        }
        
        /* Better button styling */
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.info("Please upload both 'actual' and 'predicted' CSV files to visualize the data. Optionally add an equity curve CSV file for additional analysis.")
