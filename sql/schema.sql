-- sql/schema.sql (DuckDB/Postgres-compatible)
-- Basic relational tables: users, items, interactions

-- interactions read from ratings.csv; MovieLens has userId,movieId,rating,timestamp
CREATE TABLE interactions AS SELECT userId AS user_id, movieId AS item_id, rating, timestamp FROM read_csv_auto("../data/raw/ml-25m/ratings.csv");

-- Sessionize events per user (30 min gap)
WITH events AS (
  SELECT
    user_id,
    item_id,
    timestamp,
    LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_ts 
  FROM interactions 
)
SELECT
  user_id,
  item_id,
  timestamp,
  SUM(CASE WHEN prev_ts IS NULL OR timestamp - prev_ts > 1800 THEN 1 ELSE 0 END) OVER (PARTITION BY user_id ORDER BY timestamp) AS session_id 
FROM events;
