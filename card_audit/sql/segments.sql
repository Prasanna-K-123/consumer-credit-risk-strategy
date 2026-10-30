WITH grouped AS (
  SELECT c.*, 'sex_' || CAST(sex AS TEXT) AS segment FROM customers c
  UNION ALL
  SELECT c.*, CASE WHEN age < 30 THEN 'age_under30'
                  WHEN age < 50 THEN 'age_30to49' ELSE 'age_50plus' END
  FROM customers c
)
SELECT p.model, g.segment, COUNT(*) AS rows, SUM(g.outcome) AS defaults,
       AVG(g.outcome * 1.0) AS default_rate,
       AVG(p.probability) AS mean_probability,
       SUM(CASE WHEN p.probability >= t.threshold AND g.outcome = 0 THEN 1 ELSE 0 END) AS false_positive,
       SUM(CASE WHEN p.probability < t.threshold AND g.outcome = 1 THEN 1 ELSE 0 END) AS false_negative
FROM grouped g JOIN predictions p USING (source_id) JOIN policies t USING (model)
GROUP BY p.model, g.segment ORDER BY p.model, g.segment;
