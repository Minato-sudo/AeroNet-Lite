# 🚁 AeroNet Lite: Autonomous Drone Delivery

AeroNet Lite is a Python-based AI simulation framework for an autonomous urban drone delivery network. This project fulfills all requirements for the **BSDS-SemesterProject-AI-SP2026**.

It features a robust modular pipeline that integrates Constraint Satisfaction Problems (CSP), Genetic Algorithms, A* Search Pathfinding, and Machine Learning.

## 🌟 Features / Modules

- **Module 1: Layout Validation (CSP)**
  Validates a 10x10 city grid against 4 strict urban planning rules (Industrial safety, Residential coverage, Hub charging access, and Medical access).
- **Module 2: Fleet Selection (Genetic Algorithm)**
  Evolves the optimal configuration of Light and Heavy drones to maximize delivery capacity without exceeding a strict $10,000 budget.
- **Module 3: Path Planning (A* Search)**
  Calculates the optimal Manhattan distance flight path for drones, successfully navigating around dynamic No-Fly zones.
- **Module 4: Real-Time Disruption Simulator**
  Simulates mid-flight weather disruptions and triggers real-time A* detours so drones can safely reach their destination.
- **Module 5: Machine Learning Pipeline**
  - **Regression**: Trains a Random Forest Regressor on the Kaggle Bike Sharing dataset to forecast daily delivery demand.
  - **Classification**: Trains a Random Forest Classifier on synthetic drone telemetry to detect battery/speed anomalies with 100% accuracy.

## 🚀 How to Run

### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn matplotlib streamlit
```

### 2. The 20-Step CLI Simulation
To view the fully automated 20-step event log (as required by the assignment demo scenario):
```bash
python src/main.py
```
This prints a comprehensive "story" of the AI validating the grid, selecting a fleet, routing a drone, handling a mid-air disruption, and detecting an anomaly.

### 3. The Interactive Dashboard
To launch the visual Streamlit interface featuring a live, animated drone map and interactive AI controls:
```bash
streamlit run src/visualization.py
```
The dashboard allows you to:
- Drag a custom Start/Goal route.
- Trigger mid-flight disruptions to watch the drone detour.
- Test impossible edge cases against the AI.

## 🧪 Edge-Case Testing
We have included a rigorous test suite that attempts to break the AI modules (e.g. impossible budgets, routing into walls, starting in No-Fly zones). To verify system robustness:
```bash
python src/test_suite.py
```

## 📁 Repository Structure
- `data/` - Raw datasets (e.g., Bike Sharing dataset).
- `src/` - Core Python modules (`astar_planner.py`, `fleet_selector.py`, etc.).
- `report/` - Final project documentation.

---
*Developed for BSDS-SemesterProject-AI-SP2026.*
