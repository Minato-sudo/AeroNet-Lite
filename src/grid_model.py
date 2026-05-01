from dataclasses import dataclass
from typing import List

@dataclass
class Cell:
    row: int
    col: int
    zone_type: str  # 'Residential', 'Industrial', 'Hub', 'Hospital', 'Open Field'
    population_density: str  # 'Low', 'Medium', 'High'
    no_fly: bool = False

def create_grid() -> List[List[Cell]]:
    """
    Creates a 10x10 grid for the drone delivery simulation.
    Initializes specific zones: 2 hubs, 1 hospital, 3 residential zones, 1 industrial cell.
    """
    # Initialize a 10x10 grid with Open Field and Low density
    grid = [[Cell(row=r, col=c, zone_type='Open Field', population_density='Low') for c in range(10)] for r in range(10)]
    
    # Set 2 Hubs
    grid[1][1].zone_type = 'Hub'
    grid[8][8].zone_type = 'Hub'
    
    # Set 1 Hospital
    grid[5][5].zone_type = 'Hospital'
    grid[5][5].population_density = 'High'
    
    # Set 1 Industrial Cell
    grid[9][0].zone_type = 'Industrial'
    
    # Set 3 Residential Zones (blocks of cells)
    # Zone 1
    for r in range(2, 4):
        for c in range(2, 4):
            grid[r][c].zone_type = 'Residential'
            grid[r][c].population_density = 'High'
            
    # Zone 2
    for r in range(7, 9):
        for c in range(2, 4):
            if grid[r][c].zone_type == 'Open Field':  # Don't overwrite hub at 8,8 if it was there (it's at 8,8 anyway)
                grid[r][c].zone_type = 'Residential'
                grid[r][c].population_density = 'Medium'
                
    # Zone 3
    for r in range(2, 4):
        for c in range(7, 9):
            grid[r][c].zone_type = 'Residential'
            grid[r][c].population_density = 'High'
            
    # Add a couple of initial No-Fly zones just as an example
    grid[4][5].no_fly = True
    grid[5][4].no_fly = True
    
    return grid

def print_grid(grid: List[List[Cell]]):
    """Prints a simple text representation of the grid."""
    for row in grid:
        row_str = []
        for cell in row:
            if cell.no_fly:
                row_str.append('X')
            elif cell.zone_type == 'Hub':
                row_str.append('H')
            elif cell.zone_type == 'Hospital':
                row_str.append('+')
            elif cell.zone_type == 'Industrial':
                row_str.append('I')
            elif cell.zone_type == 'Residential':
                row_str.append('R')
            else:
                row_str.append('.')
        print(' '.join(row_str))

if __name__ == "__main__":
    my_grid = create_grid()
    print("AeroNet Lite Grid Initialized:")
    print("Legend: H=Hub, +=Hospital, I=Industrial, R=Residential, X=No-Fly, .=Open Field")
    print_grid(my_grid)
