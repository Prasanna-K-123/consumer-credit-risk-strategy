WITH ranked AS (
  SELECT model, source_id,
    ROW_NUMBER() OVER (PARTITION BY model ORDER BY probability DESC, source_id ASC) AS position,
    COUNT(*) OVER (PARTITION BY model) AS population
  FROM predictions
)
SELECT r.source_id, r.model, c.fraction AS capacity_fraction,
       CASE WHEN r.position <= CAST(c.fraction * r.population AS INTEGER) THEN 1 ELSE 0 END AS flagged
FROM ranked r CROSS JOIN capacities c
ORDER BY r.model, c.fraction, r.source_id;
