-- User-level features
CREATE OR REPLACE TABLE user_features AS
SELECT
  userId,
  AVG(rating) AS user_avg_rating,
  COUNT(*) AS user_rating_count
FROM ratings
GROUP BY userId;

-- Movie-level features
CREATE OR REPLACE TABLE movie_features AS
SELECT
  movieId,
  AVG(rating) AS movie_avg_rating,
  COUNT(*) AS movie_rating_count
FROM ratings
GROUP BY movieId;

-- Final joined features
CREATE OR REPLACE TABLE interaction_features AS
SELECT
  r.userId,
  r.movieId,
  r.rating,
  r.timestamp,
  u.user_avg_rating,
  u.user_rating_count,
  m.movie_avg_rating,
  m.movie_rating_count
FROM ratings r
LEFT JOIN user_features u ON r.userId = u.userId
LEFT JOIN movie_features m ON r.movieId = m.movieId;
