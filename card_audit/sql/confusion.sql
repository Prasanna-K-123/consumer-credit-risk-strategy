SELECT p.model, t.threshold, COUNT(*) AS rows, SUM(c.outcome) AS defaults,
       SUM(CASE WHEN p.probability >= t.threshold THEN 1 ELSE 0 END) AS flagged,
       SUM(CASE WHEN p.probability >= t.threshold AND c.outcome = 1 THEN 1 ELSE 0 END) AS true_positive,
       SUM(CASE WHEN p.probability >= t.threshold AND c.outcome = 0 THEN 1 ELSE 0 END) AS false_positive,
       SUM(CASE WHEN p.probability < t.threshold AND c.outcome = 1 THEN 1 ELSE 0 END) AS false_negative,
       SUM(CASE WHEN p.probability < t.threshold AND c.outcome = 0 THEN 1 ELSE 0 END) AS true_negative,
       SUM((p.probability - c.outcome) * (p.probability - c.outcome)) / COUNT(*) AS brier,
       -SUM(CASE WHEN c.outcome = 1 THEN LN(CLIP(p.probability))
                 ELSE LN(1.0 - CLIP(p.probability)) END) / COUNT(*) AS log_loss
FROM predictions p JOIN customers c USING (source_id)
JOIN policies t USING (model)
GROUP BY p.model, t.threshold ORDER BY p.model;
