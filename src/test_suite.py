import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid_model import create_grid
from src.layout_validator import LayoutValidator
from src.astar_planner import astar
from src.fleet_selector import FleetSelector

def run_tests():
    print("\n" + "="*60)
    print("      AeroNet Lite: Exhaustive Test Suite")
    print("="*60 + "\n")
    
    grid = create_grid()
    passed_tests = 0
    total_tests = 0

    def assert_test(test_name, condition):
        nonlocal passed_tests, total_tests
        total_tests += 1
        if condition:
            print(f"✅ PASS: {test_name}")
            passed_tests += 1
        else:
            print(f"❌ FAIL: {test_name}")

    print("--- 1. Testing A* Path Planner (Edge Cases) ---")
    
    # Test 1.1: Normal Path
    path, cost, msg = astar((1, 3), (7, 5), grid)
    assert_test("Normal route from Hub to Residential", path is not None and cost > 0)
    
    # Test 1.2: Start on No-Fly
    grid[0][0].no_fly = True
    path, cost, msg = astar((0, 0), (7, 5), grid)
    assert_test("Reject route if start point is a No-Fly zone", path is None and "Start position" in msg)
    grid[0][0].no_fly = False # reset
    
    # Test 1.3: Goal on No-Fly
    grid[9][9].no_fly = True
    path, cost, msg = astar((1, 3), (9, 9), grid)
    assert_test("Reject route if goal point is a No-Fly zone", path is None and "Goal position" in msg)
    grid[9][9].no_fly = False # reset
    
    # Test 1.4: Completely Blocked Target (Wall of No-Fly zones)
    # Surround (0,0) with No-Fly zones
    grid[0][1].no_fly = True
    grid[1][0].no_fly = True
    path, cost, msg = astar((1, 3), (0, 0), grid)
    assert_test("Fail gracefully when target is completely trapped", path is None and "No valid path exists" in msg)
    grid[0][1].no_fly = False
    grid[1][0].no_fly = False
    
    print("\n--- 2. Testing Layout Validator (CSP) ---")
    
    # Test 2.1: Default Sample Grid (Should have errors)
    validator = LayoutValidator(grid)
    is_valid = validator.run_validation()
    assert_test("Detect intentional flaws in sample grid (is_valid == False)", is_valid == False)
    assert_test("Detect Hospital too far from Hub rule violation", any("Hospital" in err for err in validator.errors))
    
    # Test 2.2: Fix flaws to achieve 100% valid grid
    # To definitively fix distance constraints for testing purposes, we blanket the grid with Hubs and Chargers
    for r in range(10):
        for c in range(10):
            grid[r][c].is_hub = True
            grid[r][c].is_charging = True
    
    # Suppress print statements for the test run
    sys.stdout = open(os.devnull, 'w')
    validator2 = LayoutValidator(grid)
    is_valid_fixed = validator2.run_validation()
    sys.stdout = sys.__stdout__ # restore prints
    
    assert_test("Grid achieves 100% Valid status when flaws are manually fixed", is_valid_fixed == True)
    
    print("\n--- 3. Testing Fleet Selector (Genetic Algorithm) ---")
    
    # Test 3.1: Normal Budget & Demand
    fs = FleetSelector(total_demand=50, budget=10000)
    fleet, score = fs.run_genetic_algorithm(generations=5, pop_size=10)
    assert_test("Successfully select optimal fleet under normal conditions", fleet is not None and score > 0)
    
    # Test 3.2: Impossible Budget
    fs_poor = FleetSelector(total_demand=500, budget=100) # Can't even afford 1 drone
    fleet_poor, score_poor = fs_poor.run_genetic_algorithm(generations=5, pop_size=10)
    cost = fleet_poor[0]*1000 + fleet_poor[1]*2500
    assert_test("Refuse to exceed budget even when demand is impossibly high", cost <= 100)
    
    print(f"\n============================================================")
    print(f"   Test Suite Complete: {passed_tests}/{total_tests} Tests Passed ({(passed_tests/total_tests)*100:.0f}%)")
    print(f"============================================================\n")

if __name__ == "__main__":
    run_tests()
