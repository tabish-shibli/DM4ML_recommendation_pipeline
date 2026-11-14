# Pipeline Run Summary

- Finished: 2026-11-13T16:22:20+00:00
- Runner: prefect
- Run id: 6c43968c-88c1-4e19-b72e-772e00ec2ccd

## Stages

| stage | seconds |
|---|---|
| ingest | 51.0 |
| validate | 37.5 |
| prepare | 9.4 |
| transform | 22.8 |
| features | 24.4 |
| train | 120.8 |
| evaluate | 6.1 |

## Data quality gate: PASSED

- users: 6,040 valid of 6,070 (30 rejected)
- items: 3,883 valid of 3,896 (13 rejected)
- interactions: 999,659 valid of 1,002,409 (2,750 rejected)

## Split

- Cutoff 2000-12-02T14:59:38+00:00: 799,412 train / 199,847 test rows
- Warm test users 1,182, cold-start 640

## Test metrics (K = 10)

| model | precision_at_10 | recall_at_10 | ndcg_at_10 | coverage_at_10 | diversity_at_10 | rmse | mae |
|---|---|---|---|---|---|---|---|
| popularity | 0.1959 | 0.0521 | 0.2120 | 0.0369 | 0.6551 | 0.9611 | 0.7611 |
| matrix_factorization | 0.0380 | 0.0131 | 0.0330 | 0.0890 | 0.6225 | 0.8886 | 0.6955 |
| hybrid | 0.2316 | 0.0663 | 0.2486 | 0.0374 | 0.6772 |  |  |
