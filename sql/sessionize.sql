-- remove old table if exists
DROP TABLE IF EXISTS interactions;

-- load ratings into new table
CREATE TABLE interactions AS 
SELECT 
    userId AS user_id, 
    movieId AS item_id, 
    rating, 
    timestamp 
FROM read_csv_auto("../data/raw/ml-25m/ratings.csv");

WITH events AS (
  SELECT
    user_id,
    item_id,
    timestamp,
    -- previous event timestamp per user
    LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_ts  
  FROM interactions 
)
SELECT
  user_id,
  item_id,
  timestamp,
  -- start new session if gap > 30min, else continue same
  SUM(
    CASE 
      WHEN prev_ts IS NULL OR timestamp - prev_ts > 1800 THEN 1 ELSE 0  
    END
  ) OVER (PARTITION BY user_id ORDER BY timestamp) AS session_id  -- assign session ids
FROM events;
