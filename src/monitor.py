from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def main() -> None:
    df = pd.read_parquet("data/ratings.parquet")
    ref = df[df["timestamp"] < df["timestamp"].quantile(0.5)].sample(10000)
    cur = df[df["timestamp"] >= df["timestamp"].quantile(0.5)].sample(10000)
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref, current_data=cur)
    rep.save_html("data/evidently_report.html")
    print("Report at data/evidently_report.html")

if __name__ == "__main__":
    main()
