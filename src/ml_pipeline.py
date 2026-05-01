import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

class DemandForecaster:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        
    def run(self):
        print("\n" + "="*50)
        print("   AeroNet Lite: Demand Forecasting (Regression)")
        print("="*50)
        
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}. Skipping Demand Forecasting.")
            return
            
        # 1. Load Data (Kaggle Bike Sharing Demand)
        df = pd.read_csv(self.data_path)
        
        # 2. Select Features and Target
        # 'count' represents the demand. We predict it using weather/season data.
        features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
        target = 'count'
        
        X = df[features]
        y = df[target]
        
        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Train Model
        print("Training Random Forest Regressor on historical demand dataset...")
        self.model.fit(X_train, y_train)
        
        # 5. Evaluate
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"✅ Model Trained Successfully!")
        print(f"Mean Absolute Error (MAE): {mae:.2f} units")
        
        # Provide a sample prediction
        sample = X_test.iloc[[0]]
        pred = self.model.predict(sample)
        print(f"Sample Prediction -> Predicted Demand: {pred[0]:.0f}, Actual Demand: {y_test.iloc[0]}")

class AnomalyDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generates fake drone telemetry data as permitted by project rules."""
        np.random.seed(42)
        
        # Normal behavior ranges
        battery_drop = np.random.normal(loc=5, scale=2, size=num_samples) # % drop per flight sector
        speed = np.random.normal(loc=15, scale=3, size=num_samples)       # m/s
        route_deviation = np.random.normal(loc=2, scale=1, size=num_samples) # meters
        
        # Inject anomalies (~15% of data)
        anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.15), replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['battery', 'speed', 'deviation'])
            if anomaly_type == 'battery':
                battery_drop[idx] = np.random.uniform(15, 30) # Sudden huge battery drop
            elif anomaly_type == 'speed':
                speed[idx] = np.random.uniform(0, 5) # Sudden slow down (motor issue?)
            elif anomaly_type == 'deviation':
                route_deviation[idx] = np.random.uniform(20, 100) # Blown off course by wind
                
        df = pd.DataFrame({
            'battery_drop': battery_drop,
            'speed': speed,
            'route_deviation': route_deviation
        })
        
        # Simple if/else logic to assign the ground truth labels
        df['is_anomaly'] = ((df['battery_drop'] > 12) | 
                            (df['speed'] < 8) | 
                            (df['route_deviation'] > 15)).astype(int)
                            
        return df

    def run(self):
        print("\n" + "="*50)
        print("   AeroNet Lite: Anomaly Detection (Classification)")
        print("="*50)
        
        # 1. Generate Synthetic Data
        print("Generating synthetic drone telemetry data...")
        df = self.generate_synthetic_data(num_samples=2000)
        
        X = df[['battery_drop', 'speed', 'route_deviation']]
        y = df['is_anomaly']
        
        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Train Model
        print("Training Random Forest Classifier to detect in-flight anomalies...")
        self.model.fit(X_train, y_train)
        
        # 4. Evaluate (Accuracy & Confusion Matrix required by rubric)
        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        print(f"✅ Model Trained Successfully!")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Confusion Matrix:\n{cm}")
        print("\n(Confusion Matrix format: \n[True Negatives  False Positives]\n[False Negatives True Positives])")
        print("="*50 + "\n")

if __name__ == "__main__":
    # Demand Forecasting
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "train.csv")
    forecaster = DemandForecaster(data_path)
    forecaster.run()
    
    # Anomaly Detection
    detector = AnomalyDetector()
    detector.run()
