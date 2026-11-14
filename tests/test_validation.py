from datetime import datetime, timezone

import pandas as pd

from src.validation.validate import validate_interactions, validate_items, validate_users

VCFG = {"platform_launch": "2000-01-01", "allowed_age_codes": [1, 18, 25, 35, 45, 50, 56],
        "occupation_range": [0, 20]}
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _frame(rows, cols):
    return pd.DataFrame(rows, columns=cols, dtype=str)


def failed(result, check):
    return set(result.flags[check][result.flags[check]].index) if check in result.flags else set()


def test_interaction_checks_catch_each_defect():
    rows = [
        ["1", "10", "4", "970000000"],     # 0 fine
        ["1", "10", "4", "970000000"],     # 1 exact duplicate of 0
        ["", "10", "4", "970000001"],      # 2 missing user
        ["1", "11", "five", "970000002"],  # 3 rating not a number
        ["1", "12", "7", "970000003"],     # 4 rating out of range
        ["1", "13", "3", "TBD"],           # 5 timestamp not a number
        ["1", "14", "3", "0"],             # 6 before platform launch
        ["999", "10", "3", "970000004"],   # 7 unknown user
        ["1", "999", "3", "970000005"],    # 8 unknown movie
        ["1", "15", "2", "970000006"],     # 9 same pair as row 10 -> warning only
        ["1", "15", "5", "970000999"],     # 10
    ]
    raw = _frame(rows, ["user_id", "item_id", "rating", "timestamp"])
    res = validate_interactions(raw, valid_users={1}, valid_items={10, 11, 12, 13, 14, 15}, vcfg=VCFG, ingested_at=NOW)
    assert failed(res, "no_exact_duplicates") == {1}
    assert failed(res, "missing_value") == {2}
    assert failed(res, "invalid_type") == {3, 5}
    assert failed(res, "rating_valid_value") == {4}
    assert failed(res, "timestamp_in_window") == {6}
    assert failed(res, "user_exists") == {7}
    assert failed(res, "item_exists") == {8}
    assert set(res.mask("critical")[res.mask("critical")].index) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert failed(res, "unique_user_item_pair") == {9, 10}


def test_user_problems_are_warnings_not_rejections():
    rows = [["1", "M", "25", "4", "12345"], ["2", "male", "99", "4", ""], ["2", "F", "18", "1", "54321"],
            ["", "F", "18", "1", "11111"]]
    res = validate_users(_frame(rows, ["user_id", "gender", "age", "occupation", "zip_code"]), VCFG)
    critical = set(res.mask("critical")[res.mask("critical")].index)
    assert critical == {2, 3}                       # duplicate id, missing id
    assert failed(res, "gender_valid") == {1}
    assert failed(res, "age_code_valid") == {1}
    assert failed(res, "zip_present") == {1}
    assert 1 not in critical                        # untidy but still usable


def test_item_genre_format_is_flagged():
    rows = [["1", "Toy Story (1995)", "Animation|Children's|Comedy"], ["2", "Heat (1995)", "action | crime"],
            ["3", "", ""]]
    res = validate_items(_frame(rows, ["item_id", "title", "genres"]))
    assert failed(res, "genres_canonical") == {1}
    assert failed(res, "title_present") == {2}
    assert failed(res, "genres_present") == {2}
    assert not res.mask("critical").any()
