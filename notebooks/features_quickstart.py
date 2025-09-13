import duckdb
from pathlib import Path

con = duckdb.connect(database=":memory:")

# Load ratings.csv (from MovieLens-25M)
data_path = Path("data/raw/ml-25m/ratings.csv")
con.execute(
    f"""
    CREATE TABLE ratings AS
    SELECT * FROM read_csv_auto('{data_path}')
"""
)

# Run SQL feature pipeline
with open("sql/features.sql", "r") as f:
    sql_script = f.read()
con.execute(sql_script)

df = con.execute("SELECT * FROM interaction_features LIMIT 5").df()
print(df.head())
