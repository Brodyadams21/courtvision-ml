SELECT 'CREATE DATABASE courtvision_mlflow'
WHERE NOT EXISTS (
    SELECT FROM pg_database
    WHERE datname = 'courtvision_mlflow'
)\gexec
