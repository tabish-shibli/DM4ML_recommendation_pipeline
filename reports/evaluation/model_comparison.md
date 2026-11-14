# Model Evaluation Results

Test period: ratings after the time cutoff. Evaluated users: 1,138 warm users (seen in training, at least one test rating >= 4). Relevant item = test rating >= 4. Movies already rated in training are never recommended.

## Ranking metrics at K = 10

| model | precision_at_10 | recall_at_10 | ndcg_at_10 | hit_rate_at_10 | coverage_at_10 | diversity_at_10 | personalization_at_10 | novelty_at_10 | head_share_at_10 |
|---|---|---|---|---|---|---|---|---|---|
| popularity | 0.1959 | 0.0521 | 0.2120 | 0.6239 | 0.0369 | 0.6551 | 0.6855 | 1.4131 | 1.0000 |
| matrix_factorization | 0.0380 | 0.0131 | 0.0330 | 0.2531 | 0.0890 | 0.6225 | 0.6942 | 8.8999 | 0.1244 |
| hybrid | 0.2316 | 0.0663 | 0.2486 | 0.6722 | 0.0374 | 0.6772 | 0.7217 | 1.5831 | 0.9993 |

`coverage` is the share of recommendable movies that appear in anyone's list. `diversity` is the mean pairwise genre dissimilarity inside a list. `personalization` is 1 minus the average overlap between two users' lists. `novelty` is the mean self-information (-log2 of the share of users who rated the movie). `head_share_at_10` is the share of recommendations drawn from the 10% most-rated movies.

## Ranking metrics at K = 5

| model | precision_at_5 | recall_at_5 | ndcg_at_5 | hit_rate_at_5 |
|---|---|---|---|---|
| popularity | 0.2069 | 0.0294 | 0.2204 | 0.4851 |
| matrix_factorization | 0.0253 | 0.0039 | 0.0218 | 0.1028 |
| hybrid | 0.2525 | 0.0370 | 0.2614 | 0.5510 |

## Rating prediction (all scorable test pairs)

| model | rmse | mae |
|---|---|---|
| popularity | 0.9611 | 0.7611 |
| matrix_factorization | 0.8886 | 0.6955 |

The hybrid ranks with a blended score, so it has no rating prediction of its own.

## Is the difference real? Paired bootstrap on per-user NDCG

| comparison | metric | mean_difference | ci95_low | ci95_high | significant |
|---|---|---|---|---|---|
| matrix_factorization - popularity | ndcg_at_10 | -0.1790 | -0.1930 | -0.1652 | yes |
| hybrid - popularity | ndcg_at_10 | 0.0365 | 0.0287 | 0.0445 | yes |

## Cold-start users

640 test users have no training history. All three models fall back to the popularity list for them: Precision@10 = 0.3667, Recall@10 = 0.0617, NDCG@10 = 0.3927.

## Example recommendations (top 5, * = the user rated it 4+ in the test period)

| user_id | model | top_5 |
|---|---|---|
| 3436 | popularity | Jurassic Park; Back to the Future; Men in Black; E.T. the Extra-Terrestrial; Star Wars: Episode I - The Phantom Menace |
| 3436 | matrix_factorization | Foreign Student; Der Bewegte Mann; Boys Life 2; The Godfather; The Shawshank Redemption |
| 3436 | hybrid | The Shawshank Redemption; The Godfather; One Flew Over the Cuckoo's Nest; Pulp Fiction; The Godfather: Part II |
| 953 | popularity | Star Wars: Episode VI - Return of the Jedi; Terminator 2: Judgment Day *; Saving Private Ryan *; Jurassic Park *; Back to the Future |
| 953 | matrix_factorization | Boys Life 2; Julien Donkey-Boy; Mamma Roma; Lamerica; Sanjuro |
| 953 | hybrid | The Shawshank Redemption *; The Silence of the Lambs *; Schindler's List; The Godfather *; Raiders of the Lost Ark |
| 4463 | popularity | American Beauty; Star Wars: Episode IV - A New Hope; Star Wars: Episode V - The Empire Strikes Back; Star Wars: Episode VI - Return of the Jedi; Terminator 2: Judgment Day |
| 4463 | matrix_factorization | See the Sea (Regarde la mer); Anne Frank Remembered; Marcello Mastroianni: I Remember Yes, I Remember; Nobody Loves Me (Keiner liebt mich); Hana-bi |
| 4463 | hybrid | American Beauty; Schindler's List; The Shawshank Redemption; Fargo *; The Silence of the Lambs |

