import sqlite3
con = sqlite3.connect("mlflow.db")
cur = con.cursor()
cur.execute("UPDATE alembic_version SET version_num = 'acf3f17fdcc7'")
con.commit()
print("alembic_version now:", list(cur.execute("SELECT * FROM alembic_version")))
con.close()
