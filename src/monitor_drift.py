import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset

def check_data_drift() -> bool:
    # 1. Load reference data (Iris training baseline)
    iris = load_iris(as_frame=True)
    reference = iris.frame.sample(n=100, random_state=42)

    # 2. Simulate current production data with artificial drift on 2+ features
    current = iris.frame.sample(n=100, random_state=99).copy()
    
    cols = list(iris.feature_names)
    current[cols[0]] = current[cols[0]] * 1.8  # Drift feature 1 (sepal length)
    current[cols[2]] = current[cols[2]] * 1.8  # Drift feature 2 (petal length)

    # 3. Generate HTML report
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        
        if hasattr(report, "save_html"):
            report.save_html("drift_report.html")
        else:
            report.save("drift_report.html")
        print("Evidently drift report successfully saved to drift_report.html")
    except Exception as e:
        print(f"HTML Report rendering skipped: {e}")

    # 4. Kolmogorov-Smirnov test (p-value < 0.05)
    drifted_columns = 0
    for col in cols:
        p_val = ks_2samp(reference[col], current[col]).pvalue
        if p_val < 0.05:
            drifted_columns += 1

    # Declare dataset drift if 50%+ features drifted
    dataset_drift = drifted_columns >= (len(cols) / 2)
    print(f"Dataset Drift Detected: {dataset_drift} ({drifted_columns}/{len(cols)} features drifted)")
    
    return dataset_drift

if __name__ == "__main__":
    check_data_drift()