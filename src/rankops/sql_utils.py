import pandas as pd


def create_session_table(conn) -> None:
    query = """
    CREATE TABLE sessions AS
    WITH user_events AS (
        SELECT
            userId,
            movieId,
            timestamp,
            ROW_NUMBER() OVER (
                PARTITION BY userId
                ORDER BY timestamp
            ) AS seq
        FROM ratings
    )
    SELECT
        userId,
        movieId,
        timestamp,
        seq,
        CASE
            WHEN LAG(timestamp) OVER (
                PARTITION BY userId
                ORDER BY timestamp
            ) IS NULL
            OR (
                timestamp - LAG(timestamp) OVER (
                    PARTITION BY userId
                    ORDER BY timestamp
                )
            ) > 1800
            THEN 1
            ELSE 0
        END AS new_session
    FROM user_events;
    """
    conn.execute(query)
    print("Session table created with sessionization logic.")


def sessionize_events(conn) -> pd.DataFrame:
    query = """
    SELECT
        userId,
        movieId,
        timestamp,
        SUM(new_session) OVER (
            PARTITION BY userId
            ORDER BY timestamp
            ROWS UNBOUNDED PRECEDING
        ) AS session_id
    FROM sessions;
    """
    result = conn.execute(query).fetchdf()
    return result


def create_user_metrics(conn) -> None:
    query = """
    CREATE TABLE user_metrics AS
    SELECT userId, COUNT(*) AS rating_count, AVG(rating) AS avg_rating
    FROM ratings
    GROUP BY userId;
    """
    conn.execute(query)
    print("User metrics table created.")


def create_user_recency(conn) -> None:
    query = """
    CREATE TABLE user_recency AS
    SELECT userId, MAX(timestamp) AS last_interaction
    FROM ratings
    GROUP BY userId;
    """
    conn.execute(query)
    print("User recency table created.")
