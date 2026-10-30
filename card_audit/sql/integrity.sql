-- Every result must be zero. LEFT JOINs expose absent/extra customer keys.
SELECT 'cross_partition_feature_groups' AS control, COUNT(*) AS violations
FROM (SELECT feature_group FROM manifest GROUP BY feature_group
      HAVING COUNT(DISTINCT partition) > 1)
UNION ALL
SELECT 'missing_test_keys', COUNT(*) FROM manifest m
LEFT JOIN customers c USING (source_id)
WHERE m.partition = 'test' AND c.source_id IS NULL
UNION ALL
SELECT 'extra_or_wrong_partition_keys', COUNT(*) FROM customers c
LEFT JOIN manifest m USING (source_id)
WHERE m.source_id IS NULL OR m.partition <> 'test'
UNION ALL
SELECT 'prediction_group_disagreement', COUNT(*) FROM customers c
JOIN manifest m USING (source_id) WHERE c.feature_group <> m.feature_group
UNION ALL
SELECT 'malformed_probabilities', COUNT(*) FROM predictions
WHERE probability IS NULL OR probability < 0 OR probability > 1
UNION ALL
SELECT 'missing_model_customer_pairs', COUNT(*) FROM
    (SELECT c.source_id, p.model FROM customers c CROSS JOIN policies p
     EXCEPT SELECT source_id, model FROM predictions);
