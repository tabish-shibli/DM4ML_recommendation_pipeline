"""Data contracts for the three landing tables, written as Pandera schemas.

Two severities:
  critical - the record cannot be used at all (no key, unknown movie, rating
             of 7, ...). It is moved to the error dataset.
  warning  - the record is usable but a non-key field needs fixing. It stays,
             and the preparation stage standardises or imputes the field.

Keeping user and item problems at warning level is deliberate: rejecting a
user because their zip code is blank would orphan every rating they made.
"""
from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

CANONICAL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

# Human-readable description of every check, used in the validation report.
CHECK_CATALOG = {
    # generic, applied before Pandera while the data is still text
    "missing_value": ("critical", "Required field is empty"),
    "invalid_type": ("critical", "Value cannot be parsed into the expected type"),
    # interactions
    "rating_valid_value": ("critical", "Rating must be a whole number from 1 to 5"),
    "timestamp_in_window": ("critical", "Timestamp must fall between platform launch and ingestion time"),
    "user_exists": ("critical", "Referential integrity: user_id must exist in the users table"),
    "item_exists": ("critical", "Referential integrity: item_id must exist in the items table"),
    "no_exact_duplicates": ("critical", "Row is an exact copy of an earlier row"),
    "unique_user_item_pair": ("warning", "Same user rated the same movie more than once (latest kept in preparation)"),
    # users
    "unique_user_id": ("critical", "user_id appears more than once"),
    "gender_valid": ("warning", "Gender should be exactly M or F"),
    "age_code_valid": ("warning", "Age must be one of the MovieLens age codes"),
    "occupation_valid": ("warning", "Occupation code must be between 0 and 20"),
    "zip_present": ("warning", "Zip code is empty"),
    # items
    "unique_item_id": ("critical", "item_id appears more than once"),
    "title_present": ("warning", "Title is empty"),
    "title_has_year": ("warning", "Title should end with the release year in brackets"),
    "genres_present": ("warning", "Genre list is empty"),
    "genres_canonical": ("warning", "Genre list should use the canonical spelling and separator"),
}


def _no_exact_duplicates(df: pd.DataFrame) -> pd.Series:
    return ~df.duplicated(keep="first")


def _first_occurrence(col: str):
    def check(df: pd.DataFrame) -> pd.Series:
        return ~(df[col].notna() & df.duplicated(subset=[col], keep="first"))
    return check


def _genres_canonical(s: pd.Series) -> pd.Series:
    allowed = set(CANONICAL_GENRES)
    return s.map(lambda g: all(part in allowed for part in str(g).split("|")) if g else True)


def interactions_critical(valid_users: set[int], valid_items: set[int], ts_min: int, ts_max: int) -> DataFrameSchema:
    return DataFrameSchema(
        {
            "user_id": Column("Int64", nullable=True, checks=Check.isin(valid_users, name="user_exists")),
            "item_id": Column("Int64", nullable=True, checks=Check.isin(valid_items, name="item_exists")),
            "rating": Column("Float64", nullable=True, checks=Check.isin([1, 2, 3, 4, 5], name="rating_valid_value")),
            "timestamp": Column("Int64", nullable=True,
                                checks=Check.in_range(ts_min, ts_max, name="timestamp_in_window")),
        },
        checks=[Check(_no_exact_duplicates, name="no_exact_duplicates")],
        name="interactions_critical",
        coerce=True,
    )


def users_critical() -> DataFrameSchema:
    return DataFrameSchema(
        {"user_id": Column("Int64", nullable=True)},
        checks=[Check(_first_occurrence("user_id"), name="unique_user_id")],
        name="users_critical",
        coerce=True,
    )


def users_warning(age_codes: list[int], occ_range: tuple[int, int]) -> DataFrameSchema:
    return DataFrameSchema(
        {
            "gender": Column(str, nullable=True, checks=Check.isin(["M", "F"], name="gender_valid")),
            "age": Column("Int64", nullable=True, checks=Check.isin(age_codes, name="age_code_valid")),
            "occupation": Column("Int64", nullable=True,
                                 checks=Check.in_range(occ_range[0], occ_range[1], name="occupation_valid")),
            "zip_code": Column(str, nullable=True, checks=Check.str_length(min_value=1, name="zip_present")),
        },
        name="users_warning",
        coerce=True,
    )


def items_critical() -> DataFrameSchema:
    return DataFrameSchema(
        {"item_id": Column("Int64", nullable=True)},
        checks=[Check(_first_occurrence("item_id"), name="unique_item_id")],
        name="items_critical",
        coerce=True,
    )


def items_warning() -> DataFrameSchema:
    return DataFrameSchema(
        {
            "title": Column(str, nullable=True, checks=[
                Check.str_length(min_value=1, name="title_present"),
                Check.str_matches(r"^$|.*\(\d{4}\)\s*$", name="title_has_year"),
            ]),
            "genres": Column(str, nullable=True, checks=[
                Check.str_length(min_value=1, name="genres_present"),
                Check(_genres_canonical, name="genres_canonical"),
            ]),
        },
        name="items_warning",
        coerce=True,
    )
