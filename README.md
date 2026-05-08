# 🚁 AeroNet Lite — Autonomous Drone Delivery Simulator

> **BSDS-SemesterProject-AI-SP2026** | BS Data Science · AI Module  
> A full-stack AI simulation combining CSP, Genetic Algorithms, A* Search, Real-Time Replanning, and ML Forecasting on a 10×10 urban grid.

---

## 📸 Dashboard Preview

Launch the interactive SaaS dashboard with:
```bash
streamlit run src/visualization.py
```

---

## 🧠 System Architecture

```
aeronet_lite/
├── data/
│   └── raw/train.csv          ← Kaggle Bike Sharing Demand dataset
├── src/
│   ├── grid_model.py          ← Shared 10×10 city grid (dataclass)
│   ├── layout_validator.py    ← Module 1: CSP constraint checker
│   ├── fleet_selector.py      ← Module 2: Genetic Algorithm fleet optimizer
│   ├── astar_planner.py       ← Module 3: A* pathfinding engine
│   ├── delivery_simulator.py  ← Module 4: Real-time disruption handler
│   ├── ml_pipeline.py         ← Module 5: Regression + Classification
│   ├── visualization.py       ← SaaS Streamlit dashboard
│   ├── main.py                ← 20-step CLI simulation
│   └── test_suite.py          ← Automated test suite (9/9 passing)
├── report/                    ← Final project report
└── README.md
```

---

## 🌟 Modules

### Module 1 — Layout Validation (CSP)
Validates the city grid against 4 hard constraints:

| Rule | Constraint |
|------|-----------|
| **R1** | Industrial cells cannot be adjacent to School / Hospital |
| **R2** | Every Residential cell must be ≤ 3 Manhattan distance from a Hub |
| **R3** | Every Hub must have a Charging Pad within distance 2 |
| **R4** | Every Hospital must be a medical pickup point and reachable by a hub |

Reports all violations with exact coordinates and suggested fixes.

### Module 2 — Fleet Selection (Genetic Algorithm)
Evolves the optimal Light/Heavy drone mix under a fixed budget.

| Drone | Cost | Payload | Range |
|-------|------|---------|-------|
| Light | $1,000 | 5 kg | 12 cells |
| Heavy | $2,500 | 15 kg | 20 cells |

**Fitness:** `score = (0.75 × coverage%) − (0.25 × budget_used%)`  
Uses selection, single-point crossover, and random mutation over 50 generations.

### Module 3 — A* Path Planner
Finds the shortest safe route: **Hub → Pickup → Drop-off → Hub**.

- **State**: `(row, col)` grid coordinate  
- **Heuristic**: Manhattan distance (admissible for 4-directional movement)  
- **Blocked**: all cells where `no_fly = True`  
- Returns `(path, cost, status)` — fails gracefully with a message if no route exists.

### Module 4 — Real-Time Disruption Handler
At any simulation step, a cell can turn no-fly (weather / obstacle). The system:
1. Marks the cell in the shared grid model
2. Checks if the current drone path crosses it
3. Re-runs A* from the drone's current position to its remaining target
4. Logs the reroute event or marks delivery as failed

### Module 5 — Machine Learning Pipeline

**Demand Forecasting (Regression)**
- Dataset: [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
- Model: Random Forest Regressor (`n_estimators=50`)
- Features: `season, holiday, workingday, weather, temp, humidity, windspeed`
- Result: **MAE ≈ 108.70 units**

**Anomaly Detection (Classification)**
- Dataset: Synthetic drone telemetry (2,000 samples, 15% anomaly rate)
- Features: `battery_drop, speed, route_deviation`
- Model: Random Forest Classifier
- Result: **100% Accuracy** | Full confusion matrix reported

---

## 🚀 Quick Start

### 1. Setup
```bash
cd aeronet_lite
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn matplotlib streamlit
```

### 2. Download Dataset
Place the Kaggle Bike Sharing `train.csv` in `data/raw/train.csv`.

### 3. Run the 20-Step CLI Simulation
```bash
python src/main.py
```
Prints the full automated event log covering all 5 modules end-to-end.

### 4. Launch the Interactive Dashboard
```bash
streamlit run src/visualization.py
```

### 5. Run the Test Suite
```bash
python src/test_suite.py
```
Expected: **9/9 tests passing**.

---

## 📊 20-Step Simulation Event Log (Sample)

```
Step  1: Grid initialized. Running CSP layout validation...
Step  2: Layout analysis complete. R1 passed. R2/R3/R4 violations flagged.
Step  3: Running Genetic Algorithm for Fleet Selection...
Step  4: Fleet selected: 4 light drones, 2 heavy drones.
Step  5: Querying ML Regression model for demand forecast...
Step  6: Demand forecast complete. Predicted deliveries: ~191 units.
Step  7: Delivery Mission 1 assigned to Drone D1 (Light Drone).
Step  8: A* Planner computing route Hub(1,3) → Drop-off(7,5)...
Step  9: Route confirmed. Estimated cost: 8 moves.
Step 10: Drone D1 takes off along planned path.
Step 11: 💥 No-fly cell activated at (4,4)!
Step 12: Drone D1 path blocked. Initiating real-time A* replanning...
Step 13: Drone D1 rerouted successfully, avoiding (4,4).
Step 14: Drone D2 dispatched for Medical Delivery to Hospital (4,0).
Step 15: Telemetry streaming activated for active fleet.
Step 16: Processing telemetry through Random Forest Classifier...
Step 17: Classifier trained. Accuracy: 100.00%. D1 telemetry normal.
Step 18: ⚠️ ANOMALY DETECTED for Drone D2! battery_drop=25% flagged.
Step 19: Drone D2 initiating emergency landing protocol.
Step 20: Simulation complete. 1 completed, 1 delayed, 1 failed.
```

---

## ✅ Rubric Checklist

| Requirement | Status |
|-------------|--------|
| 10×10 grid visualization | ✅ |
| CSP layout validator with failed rule reporting | ✅ |
| Fleet selection result under budget | ✅ |
| A* route planner avoiding no-fly cells | ✅ |
| Rerouting demonstration after disruption | ✅ |
| Regression model with MAE metric | ✅ |
| Classifier with accuracy and confusion matrix | ✅ |
| 20-step simulation event log | ✅ |
| Automated test suite | ✅ 9/9 |
| Interactive visual dashboard | ✅ |

---

## 🗃️ Dataset Sources

| Purpose | Source |
|---------|--------|
| Demand forecasting | [Bike Sharing Demand — Kaggle](https://www.kaggle.com/c/bike-sharing-demand) |
| Anomaly detection | Synthetic telemetry (generated in `ml_pipeline.py`) |

---

*Developed for BSDS-SemesterProject-AI-SP2026 · BS Data Science AI Module*
