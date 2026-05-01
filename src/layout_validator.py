import sys
import os

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from src.grid_model import Cell, create_grid

class LayoutValidator:
    def __init__(self, grid: List[List[Cell]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.errors = []
        self.passed_rules = []

    def get_neighbors(self, row: int, col: int) -> List[Cell]:
        """Returns valid adjacent cells (up, down, left, right)."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                neighbors.append(self.grid[r][c])
        return neighbors

    def manhattan(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Calculates Manhattan distance between two coordinates."""
        return abs(r1 - r2) + abs(c1 - c2)

    def find_cells(self, condition) -> List[Cell]:
        """Helper to find all cells matching a condition."""
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if condition(self.grid[r][c]):
                    cells.append(self.grid[r][c])
        return cells

    def check_industrial_safety(self):
        """Rule 1: Industrial cells cannot be directly adjacent to Residential, School, or Hospital cells."""
        rule_passed = True
        industrial_cells = self.find_cells(lambda c: c.zone == 'Industrial')
        unsafe_zones = ['Residential', 'School', 'Hospital']
        
        for cell in industrial_cells:
            for neighbor in self.get_neighbors(cell.row, cell.col):
                if neighbor.zone in unsafe_zones:
                    rule_passed = False
                    self.errors.append(f"R1 Failed: Industrial cell ({cell.row}, {cell.col}) is adjacent to {neighbor.zone} cell ({neighbor.row}, {neighbor.col}). Suggested fix: convert one of them to Open Field.")
                    
        if rule_passed:
            self.passed_rules.append("R1: Industrial Safety (No industrial next to residential/school/hospital)")

    def check_residential_coverage(self):
        """Rule 2: Every Residential cell must be <= 3 cells away from a Hub."""
        rule_passed = True
        residential_cells = self.find_cells(lambda c: c.zone == 'Residential')
        hubs = self.find_cells(lambda c: c.is_hub)
        
        if not hubs:
            self.errors.append("R2 Failed: No hubs exist in the grid.")
            return

        for cell in residential_cells:
            min_dist = min(self.manhattan(cell.row, cell.col, hub.row, hub.col) for hub in hubs)
            if min_dist > 3:
                rule_passed = False
                self.errors.append(f"R2 Failed: Residential cell ({cell.row}, {cell.col}) is more than 3 cells away from every hub. Suggested fix: add a hub near ({cell.row}, {cell.col}) or convert that cell to Open Field.")
                
        if rule_passed:
            self.passed_rules.append("R2: Residential Coverage (All residential zones within 3 distance of a hub)")

    def check_hub_charging(self):
        """Rule 3: Every Hub must have a charging station within distance 2."""
        rule_passed = True
        hubs = self.find_cells(lambda c: c.is_hub)
        chargers = self.find_cells(lambda c: c.is_charging)
        
        if not chargers and hubs:
             self.errors.append("R3 Failed: No charging stations exist in the grid.")
             return

        for hub in hubs:
            min_dist = min(self.manhattan(hub.row, hub.col, chg.row, chg.col) for chg in chargers)
            if min_dist > 2:
                rule_passed = False
                self.errors.append(f"R3 Failed: Hub ({hub.row}, {hub.col}) has no charging station within distance 2. Suggested fix: add a CHG cell near the hub.")
                
        if rule_passed:
            self.passed_rules.append("R3: Hub Charging Access (All hubs within distance 2 of a charging station)")

    def check_medical_access(self):
        """Rule 4: Every Hospital must be marked as a medical pickup point and accessible to a hub (dist <= 4)."""
        rule_passed = True
        hospitals = self.find_cells(lambda c: c.zone == 'Hospital')
        hubs = self.find_cells(lambda c: c.is_hub)
        
        for hosp in hospitals:
            if not hosp.is_medical_pickup:
                rule_passed = False
                self.errors.append(f"R4 Failed: Hospital ({hosp.row}, {hosp.col}) is not marked as a medical pickup point. Suggested fix: set is_medical_pickup=True.")
            
            # Ensure hospital can actually be reached by a drone from a hub efficiently
            if hubs:
                min_dist = min(self.manhattan(hosp.row, hosp.col, hub.row, hub.col) for hub in hubs)
                if min_dist > 4:
                    rule_passed = False
                    self.errors.append(f"R4 Failed: Hospital ({hosp.row}, {hosp.col}) is too far (dist {min_dist}) from any hub for urgent delivery. Suggested fix: build a hub closer.")
                
        if rule_passed:
            self.passed_rules.append("R4: Medical Access (Hospitals correctly marked and within range of a hub)")

    def run_validation(self) -> bool:
        """Runs all rules and prints a professional report."""
        self.errors.clear()
        self.passed_rules.clear()
        
        self.check_industrial_safety()
        self.check_residential_coverage()
        self.check_hub_charging()
        self.check_medical_access()
        
        is_valid = len(self.errors) == 0
        
        print("\n" + "="*50)
        print("   AeroNet Lite: Layout Validation Report")
        print("="*50)
        print(f"Overall Layout Validity = {is_valid}\n")
        
        print("--- Passed Rules ---")
        if not self.passed_rules:
            print("  None.")
        for rule in self.passed_rules:
            print(f"  ✅ {rule}")
            
        print("\n--- Failed Rules ---")
        if not self.errors:
            print("  None. Grid is perfectly valid!")
        for err in self.errors:
            print(f"  ❌ {err}")
            
        print("="*50 + "\n")
        return is_valid

if __name__ == "__main__":
    grid = create_grid()
    validator = LayoutValidator(grid)
    validator.run_validation()
