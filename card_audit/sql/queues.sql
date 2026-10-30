WITH ranked AS (
  SELECT p.*, c.outcome,
    ROW_NUMBER() OVER (PARTITION BY model ORDER BY probability DESC, source_id ASC) AS position
  FROM predictions p JOIN customers c USING (source_id)
), totals AS (
  SELECT model, COUNT(*) AS rows, SUM(outcome) AS defaults FROM ranked GROUP BY model
), selections AS (
  SELECT r.*, cap.fraction AS capacity_fraction,
         CAST(cap.fraction * t.rows AS INTEGER) AS capacity_count,
         t.rows, t.defaults,
         CASE WHEN r.position <= CAST(cap.fraction * t.rows AS INTEGER) THEN 1 ELSE 0 END AS flagged
  FROM ranked r JOIN totals t USING (model) CROSS JOIN capacities cap
)
SELECT model, capacity_fraction, MAX(capacity_count) AS capacity_count,
       SUM(flagged) AS flagged,
       MIN(CASE WHEN flagged = 1 THEN probability END) AS cutoff_probability,
       SUM(CASE WHEN flagged = 1 THEN outcome ELSE 0 END) AS defaults_flagged,
       SUM(CASE WHEN flagged = 1 THEN 1 - outcome ELSE 0 END) AS nondefaults_flagged,
       MAX(defaults) - SUM(CASE WHEN flagged = 1 THEN outcome ELSE 0 END) AS defaults_unflagged,
       CASE WHEN MAX(defaults) > 0 THEN SUM(flagged * outcome) * 1.0 / MAX(defaults) ELSE NULL END AS capture_rate,
       CASE WHEN SUM(flagged) > 0 THEN SUM(flagged * outcome) * 1.0 / SUM(flagged) ELSE NULL END AS precision,
       SUM(flagged) * 1.0 / MAX(rows) AS flag_rate,
       5 * (MAX(defaults) - SUM(flagged * outcome)) + SUM(flagged * (1 - outcome)) AS cost_units
FROM selections GROUP BY model, capacity_fraction ORDER BY model, capacity_fraction;
