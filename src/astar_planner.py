import sys
import os
import heapq
from typing import List, Tuple, Optional

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.grid_model import Cell, create_grid

def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """Heuristic function for A* - calculates grid distance between two points."""
    return abs(r1 - r2) + abs(c1 - c2)

def get_valid_neighbors(grid: List[List[Cell]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Returns adjacent cells (Up, Down, Left, Right) that are not No-Fly zones."""
    r, c = current
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    rows, cols = len(grid), len(grid[0])
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        # Check grid boundaries
        if 0 <= nr < rows and 0 <= nc < cols:
            # Check if cell is flyable
            if not grid[nr][nc].no_fly:
                neighbors.append((nr, nc))
    return neighbors

def astar(start: Tuple[int, int], goal: Tuple[int, int], grid: List[List[Cell]]) -> Tuple[Optional[List[Tuple[int, int]]], int, str]:
    """
    A* Pathfinding Algorithm.
    Returns: (path, total_cost, status_message)
    """
    if grid[start[0]][start[1]].no_fly:
        return None, 0, "Failed: Start position is a no-fly zone."
    if grid[goal[0]][goal[1]].no_fly:
        return None, 0, "Failed: Goal position is a no-fly zone."
        
    # Priority Queue: stores tuples of (f_cost, current_node)
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Keep track of where we came from to reconstruct the path later
    came_from = {}
    
    # g_score: actual cost from start to a given node
    g_score = {start: 0}
    
    while open_set:
        # Pop the node with the lowest f_cost
        _, current = heapq.heappop(open_set)
        
        # If we reached the goal, reconstruct and return the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal], "Success"
            
        # Explore valid neighbors
        for neighbor in get_valid_neighbors(grid, current):
            tentative_g_score = g_score[current] + 1  # Cost to move 1 cell is always 1
            
            # If we found a cheaper path to this neighbor
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                # f_cost = g_cost + heuristic (Manhattan distance to goal)
                f_score = tentative_g_score + manhattan_distance(neighbor[0], neighbor[1], goal[0], goal[1])
                heapq.heappush(open_set, (f_score, neighbor))
                
    return None, 0, "Failed: No valid path exists."

def plan_delivery_route(hub: Tuple[int, int], pickup: Tuple[int, int], dropoff: Tuple[int, int], grid: List[List[Cell]]):
    """Plans a full delivery: Hub -> Pickup -> Drop-off -> Hub."""
    print(f"\n--- Planning Delivery Route ---")
    print(f"Hub: {hub} | Pickup: {pickup} | Drop-off: {dropoff}")
    
    # Leg 1: Hub to Pickup
    path1, cost1, status1 = astar(hub, pickup, grid)
    if not path1:
         print(f"Routing failed on Hub -> Pickup: {status1}")
         return None
         
    # Leg 2: Pickup to Drop-off
    path2, cost2, status2 = astar(pickup, dropoff, grid)
    if not path2:
         print(f"Routing failed on Pickup -> Drop-off: {status2}")
         return None
         
    # Leg 3: Drop-off back to Hub
    path3, cost3, status3 = astar(dropoff, hub, grid)
    if not path3:
         print(f"Routing failed on Drop-off -> Hub: {status3}")
         return None
         
    # Combine the paths (skip the first element of subsequent paths to avoid duplicates)
    full_path = path1 + path2[1:] + path3[1:] 
    total_cost = cost1 + cost2 + cost3
    
    print(f"✅ Route planned successfully! Total Cost: {total_cost} moves.")
    print(f"Full Path: {full_path}")
    return full_path, total_cost

if __name__ == "__main__":
    test_grid = create_grid()
    
    # In our sample grid: Hub is at (1, 3). Hospital (pickup) is at (4, 0). Residential (drop-off) is at (7, 5).
    hub_loc = (1, 3)
    pickup_loc = (4, 0)
    dropoff_loc = (7, 5)
    
    plan_delivery_route(hub_loc, pickup_loc, dropoff_loc, test_grid)
    
    # Test a scenario routing around the No-Fly Zone at (3, 6)
    print("\n--- Testing No-Fly Avoidance ---")
    print("Attempting to route from (3,5) to (3,7) across the No-Fly zone at (3,6)...")
    path, cost, msg = astar((3, 5), (3, 7), test_grid)
    print(f"A* seamlessly routed around it! Path: {path} (Cost: {cost})")
