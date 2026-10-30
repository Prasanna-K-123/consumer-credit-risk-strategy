SELECT partition, COUNT(*) AS rows, COUNT(DISTINCT feature_group) AS feature_groups
FROM manifest
GROUP BY partition
ORDER BY CASE partition
    WHEN 'train' THEN 1 WHEN 'calibration' THEN 2
    WHEN 'policy_validation' THEN 3 WHEN 'test' THEN 4 END;
