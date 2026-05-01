import sys
import os
import time
from typing import List, Tuple

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.grid_model import Cell, create_grid
from src.astar_planner import astar

class DeliverySimulator:
    def __init__(self, grid: List[List[Cell]]):
        self.grid = grid
        
    def check_path_validity(self, path: List[Tuple[int, int]], current_idx: int) -> bool:
        """Checks if any remaining steps in the drone's current path have become no-fly zones."""
        for step in path[current_idx+1:]:
            if self.grid[step[0]][step[1]].no_fly:
                return False
        return True

    def simulate_delivery_with_disruption(self, start: Tuple[int, int], goal: Tuple[int, int], disruption_cell: Tuple[int, int], disruption_step: int):
        """
        Simulates step-by-step drone movement.
        Injects a No-Fly disruption mid-flight to trigger real-time rerouting.
        """
        print(f"\n" + "="*50)
        print("   AeroNet Lite: Real-Time Disruption Simulator")
        print("="*50)
        print(f"Mission: {start} -> {goal}")
        
        # 1. Initial Planning
        path, cost, msg = astar(start, goal, self.grid)
        if not path:
            print(f"Initial routing failed: {msg}")
            return
            
        print(f"Initial Route Planned: {path}\n")
        
        current_idx = 0
        current_loc = path[current_idx]
        total_steps_taken = 0
        
        # 2. Step-by-Step Flight Simulation
        while current_loc != goal:
            print(f"Step {total_steps_taken+1}: Drone flying at {current_loc}")
            
            # 3. Trigger Disruption mid-flight
            if total_steps_taken + 1 == disruption_step:
                print(f"\n💥 SYSTEM ALERT: Weather anomaly/Obstacle detected at {disruption_cell}!")
                print("   Updating Global Grid: Marking cell as NO-FLY.")
                self.grid[disruption_cell[0]][disruption_cell[1]].no_fly = True
            
            # 4. Check Path Integrity
            if not self.check_path_validity(path, current_idx):
                print(f"⚠️ DANGER: Upcoming path is blocked by NO-FLY zone!")
                print(f"   Initiating real-time replanning from {current_loc} to {goal}...")
                
                # 5. Rerun A* from current location
                new_path, new_cost, msg = astar(current_loc, goal, self.grid)
                if not new_path:
                    print(f"❌ Rerouting failed! No alternate route available. Drone initiating emergency landing.")
                    return
                
                print(f"✅ Rerouting successful! New safe path: {new_path}\n")
                
                # Update trajectory
                path = new_path
                current_idx = 0 # Reset index because new_path starts from our current_loc
                
            # Move to the next node in the path
            current_idx += 1
            current_loc = path[current_idx]
            total_steps_taken += 1
            time.sleep(0.05) # Tiny pause for realism if running interactively
            
        print(f"\n🎉 Delivery Complete! Safely reached destination {goal} in {total_steps_taken} total moves.")
        print("="*50 + "\n")

if __name__ == "__main__":
    test_grid = create_grid()
    
    # Let's fly from the Hub at (1, 3) to Residential Drop-off at (7, 5)
    hub_loc = (1, 3)
    goal_loc = (7, 5)
    
    sim = DeliverySimulator(test_grid)
    
    # The A* algorithm naturally likes to cut through (4, 4) or (5, 5) to get to (7, 5).
    # We will let the drone fly a few steps, then suddenly block (5, 5).
    disruption_point = (5, 5)
    
    # We trigger the disruption on step 5 of the journey
    sim.simulate_delivery_with_disruption(hub_loc, goal_loc, disruption_point, disruption_step=5)
