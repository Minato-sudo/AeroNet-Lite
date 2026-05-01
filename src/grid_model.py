from dataclasses import dataclass
from typing import List

@dataclass
class Cell:
    row: int
    col: int
    zone: str  # 'Residential', 'Commercial', 'Industrial', 'School', 'Hospital', 'Open Field'
    density: int  # Numeric value, e.g., 5000
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0  # Estimated delivery demand

def create_grid() -> List[List[Cell]]:
    """
    Creates a 10x10 grid matching the exact layout from the 
    'Sample 10x10 Region Grid' specification.
    """
    # Initialize a 10x10 grid with Open Field
    grid = [[Cell(row=r, col=c, zone='Open Field', density=1000) for c in range(10)] for r in range(10)]
    
    # The exact layout from the image:
    # R=Residential, C=Commercial, O=Open Field, S=School, I=Industrial, H=Hospital
    # Hub=Hub, CHG=Charging, NF=No-fly
    layout = [
        ['R', 'R', 'C', 'C',   'O',   'O', 'S',  'S', 'O', 'O'],
        ['R', 'R', 'C', 'Hub', 'O',   'O', 'S',  'O', 'O', 'O'],
        ['R', 'C', 'C', 'O',   'CHG', 'O', 'O',  'O', 'I', 'I'],
        ['O', 'O', 'O', 'O',   'O',   'O', 'NF', 'O', 'I', 'I'],
        ['H', 'H', 'O', 'R',   'R',   'C', 'C',  'O', 'O', 'O'],
        ['H', 'O', 'O', 'R',   'R',   'C', 'Hub','C', 'O', 'O'],
        ['O', 'CHG','O','O',   'R',   'R', 'C',  'C', 'O', 'S'],
        ['O', 'O', 'O', 'O',   'O',   'R', 'R',  'O', 'O', 'S'],
        ['I', 'I', 'O', 'C',   'C',   'O', 'O',  'O', 'O', 'O'],
        ['I', 'I', 'O', 'C',   'C',   'O', 'O',  'H', 'H', 'O']
    ]
    
    for r in range(10):
        for c in range(10):
            char = layout[r][c]
            
            # Map the visual layout to the dataclass fields
            if char == 'R':
                grid[r][c].zone = 'Residential'
                grid[r][c].density = 8000
            elif char == 'C':
                grid[r][c].zone = 'Commercial'
                grid[r][c].density = 5000
            elif char == 'I':
                grid[r][c].zone = 'Industrial'
                grid[r][c].density = 2000
            elif char == 'S':
                grid[r][c].zone = 'School'
                grid[r][c].density = 4000
            elif char == 'H':
                grid[r][c].zone = 'Hospital'
                grid[r][c].density = 3000
                grid[r][c].is_medical_pickup = True
            elif char == 'Hub':
                grid[r][c].zone = 'Commercial' # Hubs are usually in Commercial/Open areas
                grid[r][c].is_hub = True
            elif char == 'CHG':
                grid[r][c].zone = 'Open Field'
                grid[r][c].is_charging = True
            elif char == 'NF':
                grid[r][c].zone = 'Open Field'
                grid[r][c].no_fly = True
                
    return grid

def print_grid(grid: List[List[Cell]]):
    """Prints a text representation of the grid matching the layout."""
    for row in grid:
        row_str = []
        for cell in row:
            if cell.no_fly:
                row_str.append('NF ')
            elif cell.is_hub:
                row_str.append('HUB')
            elif cell.is_charging:
                row_str.append('CHG')
            elif cell.zone == 'Residential':
                row_str.append('R  ')
            elif cell.zone == 'Commercial':
                row_str.append('C  ')
            elif cell.zone == 'Industrial':
                row_str.append('I  ')
            elif cell.zone == 'School':
                row_str.append('S  ')
            elif cell.zone == 'Hospital':
                row_str.append('H  ')
            else:
                row_str.append('O  ')
        print(' '.join(row_str))

if __name__ == "__main__":
    my_grid = create_grid()
    print("AeroNet Lite Grid Initialized (Sample Layout):")
    print_grid(my_grid)
