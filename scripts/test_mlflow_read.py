import mlflow, os, traceback

uri = "sqlite:///" + os.path.abspath("mlflow.db").replace("\\", "/")
print("URI:", uri)
mlflow.set_tracking_uri(uri)
try:
    df = mlflow.search_runs(search_all_experiments=True, max_results=200)
    print("SUCCESS shape:", df.shape)
    print("exp ids:", sorted(df["experiment_id"].unique().tolist()) if "experiment_id" in df else "n/a")
    print("metric cols:", [c for c in df.columns if c.startswith("metrics.")][:25])
    print("param cols:", [c for c in df.columns if c.startswith("params.")][:25])
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:200])
    traceback.print_exc()
