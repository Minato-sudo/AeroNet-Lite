import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.grid_model import create_grid
from src.layout_validator import LayoutValidator
from src.astar_planner import astar

def draw_grid(grid, path=None):
    """Draws the 10x10 city grid using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Modern color palette based on your requirements
    colors = {
        'Residential': '#B3E5FC', # Light Blue
        'Commercial': '#FFE0B2',  # Peach/Orange
        'Industrial': '#D7CCC8',  # Brown
        'School': '#C8E6C9',      # Light Green
        'Hospital': '#FFCDD2',    # Pink
        'Open Field': '#F5F5F5'   # Very Light Grey
    }
    
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            # y coordinate is inverted so row 0 is at the top
            y = 9 - r 
            x = c
            
            facecolor = colors.get(cell.zone, '#FFFFFF')
            
            # Base tile
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1.5, edgecolor='white', facecolor=facecolor)
            ax.add_patch(rect)
            
            # Labels / Special Zones
            if cell.no_fly:
                rect = patches.Rectangle((x, y), 1, 1, linewidth=1.5, edgecolor='white', facecolor='#37474F') # Dark Grey
                ax.add_patch(rect)
                ax.text(x+0.5, y+0.5, 'NF', color='white', ha='center', va='center', fontweight='bold', fontsize=12)
            elif cell.is_hub:
                rect = patches.Rectangle((x, y), 1, 1, linewidth=1.5, edgecolor='white', facecolor='#2962FF') # Deep Blue
                ax.add_patch(rect)
                ax.text(x+0.5, y+0.5, 'HUB', color='white', ha='center', va='center', fontweight='bold', fontsize=12)
            elif cell.is_charging:
                rect = patches.Rectangle((x, y), 1, 1, linewidth=1.5, edgecolor='white', facecolor='#00BFA5') # Teal
                ax.add_patch(rect)
                ax.text(x+0.5, y+0.5, 'CHG', color='white', ha='center', va='center', fontweight='bold', fontsize=12)
            else:
                # Add a subtle text label for the zone type (R, C, I, etc)
                if cell.zone == 'Hospital':
                    ax.text(x+0.5, y+0.5, 'H', color='#D32F2F', ha='center', va='center', fontweight='bold', fontsize=14)
                else:
                    ax.text(x+0.5, y+0.5, cell.zone[0], color='#757575', ha='center', va='center', fontsize=10)
                
    # Overlay the Drone Path
    if path:
        path_x = [c + 0.5 for r, c in path]
        path_y = [9 - r + 0.5 for r, c in path]
        # Draw line
        ax.plot(path_x, path_y, color='#D50000', linewidth=3.5, linestyle='--', marker='o', markersize=6)
        # Highlight start and end
        ax.plot(path_x[0], path_y[0], color='green', marker='s', markersize=10, label='Start')
        ax.plot(path_x[-1], path_y[-1], color='purple', marker='*', markersize=14, label='Drop-off')

    ax.axis('off')
    return fig

def main():
    st.set_page_config(page_title="AeroNet Lite Dashboard", layout="wide")
    st.title("🚁 AeroNet Lite: Autonomous Drone Delivery Dashboard")
    st.markdown("Interactive Visualizer for Grid Layout, CSP Validation, and A* Path Planning.")
    
    grid = create_grid()
    
    col1, col2 = st.columns([2, 1.2])
    
    with col1:
        st.subheader("City Grid Visualization")
        
        # Interactive A* Routing Controls
        st.write("##### 📍 Interactive A* Drone Routing")
        c1, c2, c3 = st.columns(3)
        with c1:
            start_point = st.text_input("Start (row,col) [e.g. 1,3 for Hub]", "1,3")
        with c2:
            goal_point = st.text_input("Goal (row,col) [e.g. 7,5 for Res]", "7,5")
            
        path = None
        if st.button("🚀 Calculate Optimal Route", type="primary"):
            try:
                sr, sc = map(int, start_point.split(','))
                gr, gc = map(int, goal_point.split(','))
                path, cost, msg = astar((sr, sc), (gr, gc), grid)
                if path:
                    st.success(f"Route found! Total Manhattan Cost: {cost} moves.")
                else:
                    st.error(f"Routing failed: {msg}")
            except Exception as e:
                st.error("Invalid format. Please use 'row,col' like '1,3'.")
                
        # Draw the matplotlib figure
        fig = draw_grid(grid, path=path)
        st.pyplot(fig)
        
    with col2:
        st.subheader("📊 Layout Validation (CSP)")
        validator = LayoutValidator(grid)
        is_valid = validator.run_validation()
        
        if is_valid:
            st.success("✅ Grid Layout is 100% Valid!")
        else:
            st.error("❌ Grid Layout Violates Constraints")
            with st.expander("View CSP Failed Rules", expanded=True):
                for err in validator.errors:
                    st.markdown(f"- {err}")
                    
        st.subheader("🗺️ Map Legend")
        st.markdown('''
        - <span style="color:#2962FF;font-weight:bold;">HUB</span> : Drone Dispatch Hub
        - <span style="color:#00BFA5;font-weight:bold;">CHG</span> : Charging Pad
        - <span style="color:#37474F;font-weight:bold;">NF</span> : No-Fly Zone
        - **H** : Hospital (Medical Pickup)
        - **R** : Residential | **C** : Commercial | **S** : School | **I** : Industrial
        - 🔴 **Red Dashed Line** : Planned Drone Route
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
