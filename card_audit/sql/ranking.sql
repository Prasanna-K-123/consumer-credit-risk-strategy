-- Aggregate tied scores before integrating ROC or precision/recall.
WITH tied AS (
  SELECT p.model, p.probability,
         SUM(c.outcome) AS positives, SUM(1 - c.outcome) AS negatives
  FROM predictions p JOIN customers c USING (source_id)
  GROUP BY p.model, p.probability
), running AS (
  SELECT *,
    SUM(negatives) OVER (PARTITION BY model ORDER BY probability
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS negatives_below,
    SUM(positives) OVER (PARTITION BY model ORDER BY probability DESC
      ROWS UNBOUNDED PRECEDING) AS cumulative_positives,
    SUM(positives + negatives) OVER (PARTITION BY model ORDER BY probability DESC
      ROWS UNBOUNDED PRECEDING) AS cumulative_rows,
    SUM(positives) OVER (PARTITION BY model) AS total_positives,
    SUM(negatives) OVER (PARTITION BY model) AS total_negatives
  FROM tied
)
SELECT model,
  CASE WHEN MAX(total_positives) > 0 AND MAX(total_negatives) > 0 THEN
    SUM(positives * (COALESCE(negatives_below, 0) + 0.5 * negatives)) /
      (MAX(total_positives) * 1.0 * MAX(total_negatives)) ELSE NULL END AS auc,
  CASE WHEN MAX(total_positives) > 0 THEN
    SUM(positives * 1.0 * cumulative_positives / cumulative_rows) /
      MAX(total_positives) ELSE NULL END AS average_precision
FROM running GROUP BY model ORDER BY model;
