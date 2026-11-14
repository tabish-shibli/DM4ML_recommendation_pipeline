# Data Validation Report

Generated 2026-11-13T16:43:28+00:00 from MovieLens 1M (archive md5 `c4d9eecfca2ab87c1945afe126590906`).
Dirty-feed simulation: on.

## Quality gate: PASSED

A table fails the gate if more than 5% of its rows are rejected or if fewer valid rows remain than the configured minimum.


## Rows per table

| table | total | valid | rejected | warned | rejected_share |
|---|---|---|---|---|---|
| users | 6070 | 6040 | 30 | 145 | 0.494% |
| items | 3896 | 3883 | 13 | 78 | 0.334% |
| interactions | 1002409 | 999659 | 2750 | 800 | 0.274% |

Rejected rows are in `data/errors/<table>_rejected.csv` with an `error_reasons` column. Rows that were kept but need cleaning are in `data/errors/<table>_warnings.csv`.

## Check results

| table | check | severity | description | rows_failed | pass_rate | detail |
|---|---|---|---|---|---|---|
| interactions | invalid_type | critical | Value cannot be parsed into the expected type | 200 | 99.9800% |  |
| interactions | item_exists | critical | Referential integrity: item_id must exist in the items table | 150 | 99.9850% |  |
| interactions | missing_value | critical | Required field is empty | 350 | 99.9651% |  |
| interactions | no_exact_duplicates | critical | Row is an exact copy of an earlier row | 1500 | 99.8504% |  |
| interactions | rating_valid_value | critical | Rating must be a whole number from 1 to 5 | 200 | 99.9800% |  |
| interactions | timestamp_in_window | critical | Timestamp must fall between platform launch and ingestion time | 200 | 99.9800% |  |
| interactions | user_exists | critical | Referential integrity: user_id must exist in the users table | 150 | 99.9850% |  |
| interactions | unique_user_item_pair | warning | Same user rated the same movie more than once (latest kept in preparation) | 800 | 99.9202% | 800 rows in 400 pairs |
| items | invalid_type | critical | Value cannot be parsed into the expected type | 0 | 100.0000% |  |
| items | missing_value | critical | Required field is empty | 3 | 99.9230% |  |
| items | unique_item_id | critical | item_id appears more than once | 10 | 99.7433% |  |
| items | genres_canonical | warning | Genre list should use the canonical spelling and separator | 50 | 98.7166% |  |
| items | genres_present | warning | Genre list is empty | 20 | 99.4867% |  |
| items | title_present | warning | Title is empty | 8 | 99.7947% |  |
| users | invalid_type | critical | Value cannot be parsed into the expected type | 0 | 100.0000% |  |
| users | missing_value | critical | Required field is empty | 5 | 99.9176% |  |
| users | unique_user_id | critical | user_id appears more than once | 25 | 99.5881% |  |
| users | age_code_valid | warning | Age must be one of the MovieLens age codes | 15 | 99.7529% |  |
| users | gender_valid | warning | Gender should be exactly M or F | 100 | 98.3526% |  |
| users | occupation_valid | warning | Occupation code must be between 0 and 20 | 0 | 100.0000% |  |
| users | zip_present | warning | Zip code is empty | 30 | 99.5058% |  |

## Simulated defects versus what validation caught

Every simulated defect is logged at ingestion. This table checks the validator against that log.

| table | defect_types | check | defects_injected | caught | match |
|---|---|---|---|---|---|
| interactions | non_numeric_rating, unparseable_timestamp | invalid_type | 200 | 200 | yes |
| interactions | orphan_item | item_exists | 150 | 150 | yes |
| interactions | missing_key, missing_rating | missing_value | 350 | 350 | yes |
| interactions | exact_duplicate | no_exact_duplicates | 1500 | 1500 | yes |
| interactions | out_of_range_rating | rating_valid_value | 200 | 200 | yes |
| interactions | future_timestamp, prehistoric_timestamp | timestamp_in_window | 200 | 200 | yes |
| interactions | rerated_pair | unique_user_item_pair | 400 | 400 | yes |
| interactions | orphan_user | user_exists | 150 | 150 | yes |
| items | messy_genres | genres_canonical | 50 | 50 | yes |
| items | missing_genres | genres_present | 20 | 20 | yes |
| items | missing_item_id | missing_value | 3 | 3 | yes |
| items | missing_title | title_present | 8 | 8 | yes |
| items | exact_duplicate | unique_item_id | 10 | 10 | yes |
| users | invalid_age | age_code_valid | 15 | 15 | yes |
| users | messy_gender, missing_gender | gender_valid | 100 | 100 | yes |
| users | missing_user_id | missing_value | 5 | 5 | yes |
| users | exact_duplicate | unique_user_id | 25 | 25 | yes |
| users | missing_zip | zip_present | 30 | 30 | yes |

## Profile of the validated interactions

- **interactions**: 999659
- **distinct_users**: 6040
- **distinct_items**: 3706
- **matrix_density**: 0.044659054763530055
- **rating_distribution**: {1: 56243, 2: 107552, 3: 261010, 4: 348700, 5: 226154}
- **mean_rating**: 3.581168178348817
- **first_event**: 2000-04-25T23:05:32+00:00
- **last_event**: 2003-02-28T17:49:50+00:00
- **ratings_per_user_median**: 96.0
- **ratings_per_item_median**: 123.5

## Sample of rejected interaction rows

| landing_row | user_id | item_id | rating | timestamp | event_type | error_reasons |
|---|---|---|---|---|---|---|
| 1189 | 12023 | 1018 | 4 | 971841852 | rating | user_exists |
| 1192 | 11041 | 2736 | 3 | 974820169 | rating | user_exists |
| 1616 | 3274 | 1375 | 3,5 | 979767165 | rating | invalid_type |
| 3908 | 4001 | 2154 | -1 | 965584545 | rating | rating_valid_value |
| 3948 | 4779 | 3037 |  | 963019214 | rating | missing_value |
| 3973 | 3483 | 1620 | 3 | 2077317856 | rating | timestamp_in_window |
| 4262 | 2559 | 780 | 4 | TBD | rating | invalid_type |
| 5528 | 5523 | 6808 | 4 | 959644977 | rating | item_exists |
| 7934 | 4728 | 1754 | 10 | 982382516 | rating | rating_valid_value |
| 7962 | 3574 | 8620 | 4 | 966731997 | rating | item_exists |
| 9599 | 5317 | 1784 |  | 972347364 | rating | missing_value |
| 10316 | 1170 | 377 | 4 | 0 | rating | timestamp_in_window |
| 20575 | 2611 | 3702 | 4 | 973801645 | rating | no_exact_duplicates |
| 22753 | 3401 | 1282 | 3 | 980126903 | rating | no_exact_duplicates |
