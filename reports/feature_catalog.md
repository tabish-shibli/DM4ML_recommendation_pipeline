# Feature Catalog

Generated from `configs/feature_registry.yaml` (version 1). Statistics are from the `training` snapshot, computed as of 2000-12-02T14:59:38+00:00 using only interactions on or before that time.

Offline store: `data/features/<snapshot>/*.parquet`. Online store: `data/features/online_store.db` (SQLite).

## user_profile

Stored in: `user_features`

| feature | entity | dtype | description | logic | source | null % | stats (min / mean / max or top) |
|---|---|---|---|---|---|---|---|
| user_gender | user | string | Self-reported gender (M, F, or U when missing) | standardised from users.gender | users | 0.0 | M (71%) |
| user_age_group | user | string | Age bracket label from the MovieLens age code | map age code to label, "unknown" when invalid | users | 0.0 | 25-34 (35%) |
| user_occupation | user | string | Occupation name | map occupation code to name | users | 0.0 | college/grad student (13%) |
| user_activity_count | user | int | Number of ratings the user gave in the training window | count(*) per user | interactions | 0.0 | 2.000 / 148.039 / 1,848 |
| user_avg_rating | user | float | Mean rating given by the user | mean(rating) per user | interactions | 0.0 | 1.015 / 3.705 / 5.000 |
| user_rating_std | user | float | Spread of the user's ratings (0 for a single rating) | std(rating) per user, filled with 0 | interactions | 0.0 | 0.000 / 1.011 / 2.121 |
| user_positive_share | user | float | Share of the user's ratings that are 4 or 5 | mean(rating >= 4) per user | interactions | 0.0 | 0.000 / 0.624 / 1.000 |
| user_days_since_last | user | float | Days between the user's last rating and the reference time | (reference_time - max(event_time)) in days | interactions | 0.0 | 0.000 / 74.583 / 220.565 |
| user_tenure_days | user | float | Days between the user's first and last rating | (max(event_time) - min(event_time)) in days | interactions | 0.0 | 0.000 / 10.652 / 214.008 |
| user_preferred_genre | user | string | Genre the user rated most often (ties broken by mean rating) | argmax of genre counts over the user's rated movies | interactions + items | 0.0 | Drama (37%) |
| user_genre_entropy | user | float | How spread the user's taste is across genres, 0 (one genre) to 1 (even) | normalised Shannon entropy of the user's genre distribution | interactions + items | 0.0 | 0.329 / 0.781 / 0.936 |
| user_mainstreamness | user | float | Mean popularity percentile of the movies the user rated | mean(item_popularity_pct) over the user's rated movies | interactions + item_profile | 0.0 | 0.477 / 0.851 / 0.988 |

## user_genre_affinity

Stored in: `user_genre_affinity`

| feature | entity | dtype | description | logic | source | null % | stats (min / mean / max or top) |
|---|---|---|---|---|---|---|---|
| affinity_<genre> (18 columns) | user | float | Share of the user's ratings that fall in each of the 18 genres (one column per genre, rows sum to 1) | genre counts / total genre tags per user | interactions + items | 0.0 | 0.000 / 0.056 / 0.640 |

## item_profile

Stored in: `item_features`

| feature | entity | dtype | description | logic | source | null % | stats (min / mean / max or top) |
|---|---|---|---|---|---|---|---|
| item_popularity | item | int | Number of ratings the movie received in the training window | count(*) per item | interactions | 0.0 | 0.000 / 205.875 / 2,900 |
| item_popularity_pct | item | float | Popularity percentile within the catalogue (0 to 1) | rank(item_popularity, pct=True) | interactions | 0.0 | 0.029 / 0.500 / 1.000 |
| item_avg_rating | item | float | Raw mean rating | mean(rating) per item | interactions | 5.7 | 1.000 / 3.238 / 5.000 |
| item_bayesian_rating | item | float | Mean rating shrunk towards the global mean, so movies with few ratings are not over-trusted | (sum(rating) + m * global_mean) / (count + m), m = bayesian_prior_weight | interactions | 0.0 | 1.773 / 3.408 / 4.549 |
| item_rating_std | item | float | Spread of the movie's ratings | std(rating) per item, filled with 0 | interactions | 0.0 | 0.000 / 0.927 / 2.828 |
| item_days_since_last | item | float | Days between the movie's last rating and the reference time | (reference_time - max(event_time)) in days | interactions | 5.7 | 0.000 / 5.782 / 219.467 |
| item_trending_count | item | int | Ratings received in the last N days before the reference time | count(*) where event_time > reference_time - trending_window_days | interactions | 0.0 | 0.000 / 76.194 / 1,196 |
| item_release_year | item | int | Release year parsed from the title | regex on the "(YYYY)" suffix of the title | items | 0.0 | 1,919 / 1,986 / 2,000 |
| item_age_years | item | int | Years between release and the reference time | reference_year - item_release_year | items | 0.0 | 0.000 / 13.893 / 81.000 |
| item_n_genres | item | int | Number of genres the movie is tagged with | len(genres) | items | 0.0 | 0.000 / 1.643 / 6.000 |

## item_genres

Stored in: `item_genres`

| feature | entity | dtype | description | logic | source | null % | stats (min / mean / max or top) |
|---|---|---|---|---|---|---|---|
| genre_<genre> (18 columns) | item | int | Multi-hot genre flags, one column per genre | 1 if the movie carries the genre else 0 | items | 0.0 | 0.000 / 0.091 / 1.000 |

## interaction_context

Stored in: `interaction_features`

| feature | entity | dtype | description | logic | source | null % | stats (min / mean / max or top) |
|---|---|---|---|---|---|---|---|
| ix_days_before_reference | user_item | float | Age of the rating at the reference time, in days | (reference_time - event_time) in days | interactions | 0.0 | 0.000 / 85.355 / 220.663 |
| ix_recency_weight | user_item | float | Exponential time-decay weight with a configurable half-life | 0.5 ** (ix_days_before_reference / recency_half_life_days) | interactions | 0.0 | 0.183 / 0.587 / 1.000 |
| ix_rating_centered | user_item | float | Rating minus the user's own mean, removes personal rating scale | rating - user_avg_rating | interactions + user_profile | 0.0 | -3.891 / 0.000 / 3.696 |
| ix_is_positive | user_item | int | 1 if the rating is at or above the positive threshold | rating >= positive_rating_threshold | interactions | 0.0 | 0.000 / 0.579 / 1.000 |

## on_demand

Stored in: `computed at request time`

| feature | entity | dtype | description | logic | source | null % | stats (min / mean / max or top) |
|---|---|---|---|---|---|---|---|
| ui_genre_match | user_item | float | How well a candidate movie's genres match the user's genre affinity | dot(affinity_vector, genre_vector) / item_n_genres | user_genre_affinity + item_genres |  | computed per request |

