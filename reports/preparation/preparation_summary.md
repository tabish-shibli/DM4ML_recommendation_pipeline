# Preparation Summary

| table | rows in | rows out |
|---|---|---|
| users | 6,040 | 6,040 |
| items | 3,883 | 3,883 |
| interactions | 999,659 | 999,259 |

| fix | rows affected |
|---|---|
| users.gender_standardised | 60 |
| users.gender_unknown | 40 |
| users.age_unknown | 15 |
| users.occupation_unknown | 0 |
| users.zip_unknown | 30 |
| items.title_filled | 8 |
| items.release_year_imputed | 8 |
| items.article_moved | 915 |
| items.genres_reformatted | 50 |
| items.genre_tokens_dropped | 0 |
| items.genres_missing | 20 |
| interactions.rerates_collapsed | 400 |
