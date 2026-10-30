WITH binned AS (
  SELECT p.*, c.outcome,
         MIN(9, CAST(p.probability * 10 AS INTEGER)) AS risk_band
  FROM predictions p JOIN customers c USING (source_id)
)
SELECT model, risk_band, risk_band / 10.0 AS bin_left,
       (risk_band + 1) / 10.0 AS bin_right, COUNT(*) AS rows,
       SUM(outcome) AS defaults, AVG(probability) AS mean_probability,
       AVG(outcome * 1.0) AS default_rate
FROM binned GROUP BY model, risk_band ORDER BY model, risk_band;
