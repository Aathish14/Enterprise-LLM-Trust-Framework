import sqlite3
import os

db = os.path.abspath("mlflow.db")
print("DB:", db, "exists:", os.path.exists(db), "size:", os.path.getsize(db))
con = sqlite3.connect(db)
cur = con.cursor()
print("tables:", [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")])
print("experiments:", list(cur.execute("SELECT experiment_id, name, lifecycle_stage FROM experiments")))
try:
    rows = list(cur.execute("SELECT COUNT(*) FROM runs WHERE experiment_id='2'"))
    print("runs exp2 count:", rows)
except Exception as e:
    print("runs exp2 err:", e)
try:
    rows = list(cur.execute("SELECT run_uuid, experiment_id, name, status, start_time FROM runs WHERE experiment_id='2' ORDER BY start_time"))
    for r in rows:
        print("RUN:", r)
except Exception as e:
    print("run list err:", e)
try:
    print("alembic_version:", list(cur.execute("SELECT * FROM alembic_version")))
except Exception as e:
    print("alembic err:", e)
# metric/param sample
print("metric cols:", [r[1] for r in cur.execute("PRAGMA table_info(latest_metrics)")] if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='latest_metrics'").fetchone() else "no latest_metrics")
print("latest_metrics rows:", list(cur.execute("SELECT * FROM latest_metrics LIMIT 5")) if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='latest_metrics'").fetchone() else "n/a")
con.close()
