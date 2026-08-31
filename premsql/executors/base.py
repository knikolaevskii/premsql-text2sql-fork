import collections
import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    if len(cols) != len(set(cols)):
        duplicates = [
            item for item, count in collections.Counter(cols).items() if count > 1
        ]
        for dup in duplicates:
            indices = [i for i, x in enumerate(cols) if x == dup]
            for i in indices:
                cols[i] = f"{dup}_{i}"
        df.columns = cols
    return df


def normalize_table(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """
    Normalizes a dataframe by:
    1. removing all duplicate rows
    2. sorting columns in alphabetical order
    3. sorting rows using values from first column to last (unless the
       question itself asks for a specific order, in which case row order
       is part of what's being graded and must not be touched)
    4. resetting index
    """
    df = df.drop_duplicates()
    sorted_df = df.reindex(sorted(df.columns), axis=1)

    pattern = re.compile(r"\b(order|sort|arrange)\b", re.IGNORECASE)
    has_order_by = bool(re.search(pattern, question.lower()))

    if not has_order_by:
        sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))

    sorted_df = deduplicate_columns(sorted_df)
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df


def subset_df(df_sub: pd.DataFrame, df_super: pd.DataFrame, question: str) -> bool:
    """
    Checks whether df_sub's columns and values are all present in df_super,
    i.e. the predicted query returned everything the gold query asked for
    plus (optionally) extra columns. Used to give partial credit when a
    prediction's result set contains the correct answer but isn't an exact
    match to the gold query's exact column set.
    """
    if df_sub.empty:
        return False

    df_super_temp = df_super.copy(deep=True)
    matched_columns = []
    df_sub = deduplicate_columns(df_sub)
    df_super_temp = deduplicate_columns(df_super_temp)

    for col_sub_name in df_sub.columns:
        col_match = False
        for col_super_name in df_super_temp.columns:
            col_sub = df_sub[col_sub_name].sort_values().reset_index(drop=True)
            col_super = df_super_temp[col_super_name].sort_values().reset_index(drop=True)

            try:
                assert_series_equal(col_sub, col_super, check_dtype=False, check_names=False)
                col_match = True
                matched_columns.append(col_super_name)
                df_super_temp = df_super_temp.drop(columns=[col_super_name])
                break
            except AssertionError:
                continue

        if not col_match:
            return False

    df_sub_normalized = normalize_table(df_sub, question)
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(df_super_matched, question)

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False


class BaseExecutor(ABC):

    @abstractmethod
    def execute_sql(self, sql: str, dsn_or_db_path: str) -> dict:
        return {"result": None, "result_df": None, "execution_time": None, "error": None}

    def match_sqls(
        self, predicted_sql: str, gold_sql: str, dsn_or_db_path: str, question: Optional[str] = None
    ) -> dict:
        """
        Matches predicted vs. gold SQL and reports both exact match and
        subset match:

            {"result": int, "subset_match": int, "error": str | None}

        subset_match is 1 when the gold result set is contained in the
        predicted result set (see subset_df) even if it isn't an exact
        match — e.g. the model selected extra columns beyond what was
        asked. This gives evaluation a partial-credit signal distinct from
        outright failure, on top of upstream's exact-match-only result.
        """
        prediction = self.execute_sql(sql=predicted_sql, dsn_or_db_path=dsn_or_db_path)
        gold = self.execute_sql(sql=gold_sql, dsn_or_db_path=dsn_or_db_path)

        if prediction["error"]:
            return {"result": 0, "subset_match": 0, "error": prediction["error"]}

        is_match = set(prediction["result"]) == set(gold["result"])
        if is_match:
            return {"result": int(is_match), "subset_match": 1, "error": None}

        question_str = question if question is not None else ""
        subset = subset_df(gold["result_df"], prediction["result_df"], question_str)

        if subset:
            return {"result": 0, "subset_match": 1, "error": None}
        return {"result": 0, "subset_match": 0, "error": "Table mismatch"}

    def clean_abnormal(self, input: list[float]) -> list[float]:
        input_array = np.asarray(input)
        mean = np.mean(input_array)
        std = np.std(input_array)
        return [x for x in input_array if mean - 3 * std < x < mean + 3 * std]

    def iterated_execution(
        self,
        predicted_sql: str,
        gold_sql: str,
        dsn_or_db_path: str,
        num_iterations: int,
    ) -> dict:
        is_match = self.match_sqls(
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            dsn_or_db_path=dsn_or_db_path,
        )

        if is_match["result"] == 1:
            diff_list = [
                self.execute_sql(sql=predicted_sql, dsn_or_db_path=dsn_or_db_path)["execution_time"]
                / self.execute_sql(sql=gold_sql, dsn_or_db_path=dsn_or_db_path)["execution_time"]
                for _ in range(num_iterations)
            ]
            processed_diff_list = self.clean_abnormal(diff_list)
            return {
                "result": sum(processed_diff_list) / len(processed_diff_list),
                "subset_match": 1,
                "error": None,
            }
        elif is_match["subset_match"] == 1:
            # Partial credit for VES too: the query is directionally right
            # (subset match) even though it isn't the exact gold query, so
            # it isn't scored as a hard zero.
            return {"result": 0.5, "subset_match": 1, "error": None}
        else:
            return {"result": 0, "subset_match": 0, "error": is_match["error"]}
