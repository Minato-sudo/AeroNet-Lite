# AeroNet Lite - Drone Delivery Simulation

A drone delivery simulator for the BSDS-SemesterProject-AI-SP2026.
It models a city as a 10x10 grid with layout validation, drone fleet selection, path planning via A*, handling disruptions, demand forecasting, and anomaly detection.

## Project Structure
```
aeronet_lite/
  data/
    raw/
    processed/
  src/
    grid_model.py
    # (other python modules to be added)
  notebooks/
  report/
    figures/
```

## Setup
Activate the virtual environment:
```bash
source venv/bin/activate
```

Run the grid model to see the initial state:
```bash
python src/grid_model.py
```
