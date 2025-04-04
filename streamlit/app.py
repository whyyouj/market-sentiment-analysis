import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from streamlit.components.v1 import html

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'actual_values' not in st.session_state:
    st.session_state.actual_values = None
if 'pred_values' not in st.session_state:
    st.session_state.pred_values = None
if 'current_actual' not in st.session_state:
    st.session_state.current_actual = "-"
if 'current_predicted' not in st.session_state:
    st.session_state.current_predicted = "-"
if 'current_error' not in st.session_state:
    st.session_state.current_error = "-"
if 'frame_change_trigger' not in st.session_state:
    st.session_state.frame_change_trigger = False
if 'frame_slider' not in st.session_state:
    st.session_state.frame_slider = st.session_state.current_frame

# Function to update the metrics based on current frame
def update_metrics(frame):
    if (st.session_state.actual_values is None or 
        frame <= 0 or 
        frame > len(st.session_state.actual_values)):
        st.session_state.current_actual = "-"
        st.session_state.current_predicted = "-"
        st.session_state.current_error = "-"
    else:
        idx = frame - 1
        actual = st.session_state.actual_values[idx]
        pred = st.session_state.pred_values[idx]
        error = abs(actual - pred)
        
        st.session_state.current_actual = f"{actual:.4f}"
        st.session_state.current_predicted = f"{pred:.4f}"
        st.session_state.current_error = f"{error:.4f}"

# Frame change callback
def on_frame_change():
    new_frame = int(st.session_state.new_frame_value)
    st.session_state.current_frame = new_frame
    update_metrics(new_frame)
    st.session_state.frame_change_trigger = not st.session_state.frame_change_trigger  # Toggle to force rerun

st.set_page_config(page_title="Actual vs Predicted", layout="wide")

st.title("Actual vs Predicted Values Visualization")
st.write("Upload CSV files containing 'actual' and 'predicted' values to visualize the comparison")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    actual_file = st.file_uploader("Upload 'Actual' CSV file", type="csv", key="actual_file")
with col2:
    pred_file = st.file_uploader("Upload 'Predicted' CSV file", type="csv", key="pred_file")

# Hidden widget for frame updates
frame_change_container = st.empty()

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

# Create placeholders
chart_placeholder = st.empty()

# Display real-time metrics
# st.subheader("Current Point Values")
# metrics_cols = st.columns(3)
# with metrics_cols[0]:
#     actual_metric = st.metric("Actual Value", st.session_state.current_actual)
# with metrics_cols[1]:
#     pred_metric = st.metric("Predicted Value", st.session_state.current_predicted)
# with metrics_cols[2]:
#     error_metric = st.metric("Absolute Error", st.session_state.current_error)

# Create an invisible element that will be used to trigger rerun
with frame_change_container:
    # Create a hidden number input that will be updated by JS
    st.number_input("New Frame", min_value=0, value=0, 
                    label_visibility="collapsed", 
                    key="new_frame_value", 
                    on_change=on_frame_change)
    
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

summary_placeholder = st.empty()

if actual_file and pred_file:
    # Parse the data
    actual_values = parse_csv(actual_file)
    pred_values = parse_csv(pred_file)
    
    if actual_values is not None and pred_values is not None:
        # Store values in session state
        st.session_state.actual_values = actual_values
        st.session_state.pred_values = pred_values
        
        # Check if the arrays have the same length
        if len(actual_values) != len(pred_values):
            st.error("Error: The 'actual' and 'predicted' data must have the same number of points")
        else:
            st.success(f"Files loaded successfully! Found {len(actual_values)} data points.")
            
            # Create x-axis values (indices)
            x_values = np.arange(len(actual_values))
            num_points = len(actual_values)
            
            # Calculate y-axis limits once
            y_min = min(np.min(actual_values), np.min(pred_values)) * 0.95
            y_max = max(np.max(actual_values), np.max(pred_values)) * 1.05
            
            # ========= CLIENT-SIDE ANIMATION WITH PLOTLY =========
            # Create full frames for the animation
            frames = []
            
            # For each point, create a frame with data up to that point
            for i in range(0, num_points + 1):
                frame_data = [
                    # Actual values trace
                    go.Scatter(
                        x=x_values[:i],
                        y=actual_values[:i],
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=2)
                    ),
                    # Predicted values trace
                    go.Scatter(
                        x=x_values[:i],
                        y=pred_values[:i],
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red', width=2)
                    )
                ]
                
                # Add current point markers if we have points
                if i > 0:
                    frame_data.extend([
                        # Current actual point
                        go.Scatter(
                            x=[i-1],
                            y=[actual_values[i-1]],
                            mode='markers',
                            marker=dict(color='blue', size=10),
                            showlegend=False
                        ),
                        # Current predicted point
                        go.Scatter(
                            x=[i-1],
                            y=[pred_values[i-1]],
                            mode='markers', 
                            marker=dict(color='red', size=10),
                            showlegend=False
                        )
                    ])
                
                frames.append(go.Frame(data=frame_data, name=str(i)))
            
            # Initial empty traces for starting state
            fig = go.Figure(
                data=[
                    go.Scatter(x=[], y=[], mode='lines', name='Actual', line=dict(color='blue', width=2)),
                    go.Scatter(x=[], y=[], mode='lines', name='Predicted', line=dict(color='red', width=2))
                ],
                frames=frames
            )
            
            # Add animation controls and slider
            animation_speed = num_points / 15  # Frames per second (15 second total duration)
            
            fig.update_layout(
                title='Actual vs Predicted Values',
                xaxis=dict(
                    title='Data Point',
                    range=[-1, len(x_values)+1]
                ),
                yaxis=dict(
                    title='Value',
                    range=[y_min, y_max]
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.1,
                        y=0,
                        xanchor="right",
                        yanchor="top",
                        buttons=[
                            dict(
                                label="▶️ Play",
                                method="animate",
                                args=[None, {"frame": {"duration": 1000/animation_speed, "redraw": True},
                                            "fromcurrent": True,
                                            "transition": {"duration": 0}}]
                            ),
                            dict(
                                label="⏸️ Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": True},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}]
                            )
                        ]
                    )
                ],
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "prefix": "Frame: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 0},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }
                            ],
                            "label": str(i),
                            "method": "animate"
                        }
                        for i in range(num_points + 1)
                    ]
                }],
                height=600
            )
            
            # Display the interactive chart
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # JavaScript to sync Plotly animation with Streamlit metrics
            js_code = """
            <script>
                const streamlitDoc = window.parent.document;
                
                function waitForPlotly() {
                    const plotDiv = document.querySelector('.js-plotly-plot');
                    if (!plotDiv) {
                        console.log("Waiting for Plotly...");
                        setTimeout(waitForPlotly, 300);
                        return;
                    }
                    
                    console.log("Plotly found, setting up listeners");
                    setupFrameListener(plotDiv);
                }
                
                function setupFrameListener(plotDiv) {
                    // Track the currently shown frame
                    let currentFrame = 0;
                    
                    // Listen for frame changes during animation
                    plotDiv.on('plotly_animatingframe', function(e) {
                        if (e && e.name) {
                            const frameNum = parseInt(e.name);
                            if (frameNum !== currentFrame) {
                                currentFrame = frameNum;
                                console.log("Animation frame changed to:", frameNum);
                                updateStreamlitFrame(frameNum);
                            }
                        }
                    });
                    
                    // Listen for manual slider changes in Plotly
                    plotDiv.on('plotly_sliderchange', function(e) {
                        if (e && e.step && e.step.value) {
                            const frameNum = parseInt(e.step.value);
                            if (frameNum !== currentFrame) {
                                currentFrame = frameNum;
                                console.log("Plotly slider changed to:", frameNum);
                                updateStreamlitFrame(frameNum);
                            }
                        }
                    });
                    
                    // Auto-play after a delay
                    setTimeout(function() {
                        const playButton = document.querySelector('.updatemenu-item');
                        if (playButton) {
                            console.log("Auto-playing animation");
                            playButton.click();
                        }
                    }, 1500);
                }
                
                function updateStreamlitFrame(frameNum) {
                    // Find the hidden number input
                    const numberInputs = streamlitDoc.querySelectorAll('input[type="number"]');
                    let hiddenInput = null;
                    
                    for (let input of numberInputs) {
                        const label = input.parentElement.querySelector('label');
                        if (label && label.textContent === "New Frame") {
                            hiddenInput = input;
                            break;
                        }
                    }
                    
                    if (hiddenInput) {
                        // Update the input value
                        hiddenInput.value = frameNum;
                        
                        // Dispatch events to trigger change
                        hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
                        hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
                        
                        // Focus and blur to ensure event triggers
                        hiddenInput.focus();
                        hiddenInput.blur();
                        
                        // Also click the input to ensure change is registered
                        setTimeout(() => {
                            hiddenInput.click();
                        }, 10);
                    } else {
                        console.error("Could not find the hidden number input");
                    }
                }
                
                // Start initialization when ready
                if (document.readyState === 'complete') {
                    waitForPlotly();
                } else {
                    window.addEventListener('load', waitForPlotly);
                }
            </script>
            """
            
            # Inject the JavaScript
            html(js_code, height=0)
            
            # Update metrics for the initial frame
            update_metrics(0)
            
            # Add summary section
            with summary_placeholder.container():
                st.subheader("Data Summary")
                mae = np.mean(np.abs(actual_values - pred_values))
                rmse = np.sqrt(np.mean(np.square(actual_values - pred_values)))
                
                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    st.metric("Mean Absolute Error", f"{mae:.4f}")
                with metrics_cols[1]:
                    st.metric("Root Mean Square Error", f"{rmse:.4f}")
                with metrics_cols[2]:
                    st.metric("Total Data Points", num_points)

# Add custom CSS for improved appearance
st.markdown("""
<style>
    /* Improve metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: bold;
        color: #555;
    }
    
    /* Fix Plotly animation controls */
    .updatemenu-item-text {
        font-weight: bold;
    }
    
    /* Improve Plotly slider appearance */
    .slider-rail, .slider-track {
        height: 6px !important;
    }
    
    .slider-handle {
        width: 14px !important;
        height: 14px !important;
    }
    
    /* Remove border around plotly graph */
    .js-plotly-plot .plotly .modebar {
        top: 10px !important;
    }
    
    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Better header styling */
    h1 {
        color: #1f77b4;
    }
    
    h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)
