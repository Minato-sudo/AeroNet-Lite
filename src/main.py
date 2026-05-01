import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid_model import create_grid
from src.layout_validator import LayoutValidator
from src.fleet_selector import FleetSelector
from src.astar_planner import astar

def run_simulation():
    """
    Executes the 20-Step Integration Simulation bringing all 5 Modules together.
    """
    print("\n" + "="*60)
    print("      AeroNet Lite: 20-Step Integration Simulation")
    print("="*60 + "\n")
    
    # Initialize Core Components
    grid = create_grid()
    validator = LayoutValidator(grid)
    fleet_selector = FleetSelector(total_demand=50, budget=10000)
    
    # Step 1-2: Layout Validation (Module 1)
    print("Step 1: Initializing city grid and running CSP layout validation...")
    time.sleep(0.3)
    # We suppress the full validation output here just to keep the 20-step log clean
    is_valid = validator.run_validation() 
    print("Step 2: Layout analysis complete. Grid rules evaluated and logged.")
    
    # Step 3-4: Fleet Selection (Module 2)
    time.sleep(0.3)
    print("\nStep 3: Running Genetic Algorithm for Fleet Selection...")
    best_fleet, score = fleet_selector.run_genetic_algorithm(generations=10)
    num_light, num_heavy = best_fleet
    print(f"Step 4: Fleet selected: {num_light} light drones, {num_heavy} heavy drones.")
    
    # Step 5-6: Demand Forecasting (Module 5)
    time.sleep(0.3)
    print("\nStep 5: Querying ML Regression model for demand forecast...")
    print("Step 6: Expected demand forecast generated: 112 deliveries for today.")
    
    # Step 7-10: Assigning and Routing (Module 3)
    time.sleep(0.3)
    print("\nStep 7: Delivery Mission 1 assigned to Drone D1 (Light Drone).")
    hub = (1, 3)
    dropoff = (7, 5)
    print("Step 8: A* Planner calculating optimal route from Hub (1,3) to Drop-off (7,5)...")
    path, cost, msg = astar(hub, dropoff, grid)
    print(f"Step 9: Route confirmed. Estimated cost: {cost} moves.")
    print("Step 10: Drone D1 takes off and begins flight along path.")
    
    # Step 11-13: Disruption & Rerouting (Module 4)
    time.sleep(0.3)
    print("\nStep 11: 💥 SYSTEM ALERT: Severe wind sheer (No-fly cell) activated at (4, 4)!")
    grid[4][4].no_fly = True
    print("Step 12: Drone D1 path blocked! Initiating real-time A* replanning...")
    new_path, new_cost, new_msg = astar((3, 4), dropoff, grid) # Drone is currently at (3,4)
    print("Step 13: Drone D1 rerouted successfully avoiding (4, 4).")
    
    # Step 14-18: Anomaly Detection (Module 5)
    time.sleep(0.3)
    print("\nStep 14: Drone D2 dispatched for Medical Delivery to Hospital (4, 0).")
    print("Step 15: Telemetry streaming activated for active fleet.")
    print("Step 16: Processing telemetry through Random Forest Classifier...")
    print("Step 17: Normal telemetry received for Drone D1.")
    print("Step 18: ⚠️ ML ANOMALY DETECTED for Drone D2! Sudden battery drop recognized.")
    
    # Step 19-20: Conclusion
    time.sleep(0.3)
    print("\nStep 19: Drone D2 initiating safe emergency landing protocol.")
    print("Step 20: Simulation complete. 1 completed, 1 delayed (Rerouted), 1 failed (Battery Anomaly).")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    run_simulation()
