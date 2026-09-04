import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from func_timeout import func_timeout
from sqlalchemy import create_engine

from premsql.executors.base import BaseExecutor
from premsql.logger import setup_console_logger

logger = setup_console_logger(name="[POSTGRES-EXEC]")


class PostgresExecutor(BaseExecutor):
    """
    Executes SQL against a PostgreSQL database.

    Credentials are read from the environment:
        POSTGRES_HOST (default: localhost)
        POSTGRES_PORT (default: 5432)
        POSTGRES_USER
        POSTGRES_PASSWORD

    `dsn_or_db_path` is part of BaseExecutor's interface and carries each
    row's db_path. It is only used as a fallback: if the environment
    doesn't supply user/password and dsn_or_db_path points at an existing
    JSON file of the same keys (lowercased: host/port/user/password), that
    file is read instead. Loading a .env file is the caller's job — this
    only reads os.environ.

    The target database is resolved per row, because a benchmark's questions
    can span many databases — Defog's 210 questions cover 11 of them, so a
    single fixed database could only ever evaluate a fraction of the set.
    Resolution order for each call:

      1. dsn_or_db_path is already a postgresql:// URL — connect to it as-is
      2. db_name was passed to the constructor — use that for every row
      3. otherwise take the database name from dsn_or_db_path's filename stem,
         which is how the dataset classes lay out db_id (".../<db_id>/<db_id>
         .sqlite"). The file itself need not exist; only its name is used.

    `db_name` is therefore optional, and only needed to force every row at one
    database.
    """

    def __init__(self, db_name: Optional[str] = None, query_timeout: float = 10.0) -> None:
        self.db_name = db_name
        self.query_timeout = query_timeout

    def _resolve_db_name(self, dsn_or_db_path: Optional[str]) -> str:
        if self.db_name:
            return self.db_name
        if dsn_or_db_path:
            stem = Path(dsn_or_db_path).stem
            if stem:
                return stem
        raise ValueError(
            "Could not determine which Postgres database to query: pass "
            "db_name, or give a dsn_or_db_path whose filename identifies the "
            "database."
        )

    def _load_credentials(self, dsn_or_db_path: Optional[str]) -> Dict[str, Any]:
        creds = {
            "host": os.environ.get("POSTGRES_HOST", "localhost"),
            "port": os.environ.get("POSTGRES_PORT", 5432),
            "user": os.environ.get("POSTGRES_USER"),
            "password": os.environ.get("POSTGRES_PASSWORD"),
        }
        # Password deliberately not required: a local server using trust or
        # peer authentication (the default for a Homebrew/initdb install) has
        # no password to give, and demanding one turns a working setup into a
        # confusing failure.
        if creds["user"]:
            return creds

        # dsn_or_db_path carries each row's db_path, which for most datasets is
        # a .sqlite file rather than a credentials file. Only treat it as
        # credentials when it actually parses as JSON, otherwise fall through
        # to the error below rather than surfacing a confusing decode failure.
        if dsn_or_db_path and Path(dsn_or_db_path).is_file():
            try:
                with open(dsn_or_db_path, "r") as f:
                    from_file = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                from_file = None
            if isinstance(from_file, dict) and "user" in from_file:
                logger.info(f"Postgres credentials read from file: {dsn_or_db_path}")
                return {**creds, **from_file}

        raise ValueError(
            "Postgres credentials not found. Set POSTGRES_USER and "
            "POSTGRES_PASSWORD (plus POSTGRES_HOST / POSTGRES_PORT if not "
            "localhost:5432), or pass a path to a JSON credentials file."
        )

    def execute_sql(self, sql: str, dsn_or_db_path: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        result = None
        df = pd.DataFrame()
        error = None

        try:
            if dsn_or_db_path and str(dsn_or_db_path).startswith("postgresql://"):
                db_url = dsn_or_db_path
            else:
                creds = self._load_credentials(dsn_or_db_path)
                db_name = self._resolve_db_name(dsn_or_db_path)
                db_url = (
                    f"postgresql://{creds['user']}:{creds['password']}"
                    f"@{creds['host']}:{creds['port']}/{db_name}"
                )
            engine = create_engine(db_url)

            with engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                df = func_timeout(self.query_timeout, pd.read_sql_query, args=(sql, conn))
                result = list(df.itertuples(index=False, name=None))

        except Exception as e:
            logger.error(f"Postgres execution failed: {e}")
            error = str(e)

        return {
            "result": result,
            "result_df": df,
            "error": error,
            "execution_time": time.time() - start_time,
        }
