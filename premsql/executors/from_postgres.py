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

    `db_name` is per-run configuration rather than a credential, so it's a
    constructor argument.
    """

    def __init__(self, db_name: str, query_timeout: float = 10.0) -> None:
        self.db_name = db_name
        self.query_timeout = query_timeout

    def _load_credentials(self, dsn_or_db_path: Optional[str]) -> Dict[str, Any]:
        creds = {
            "host": os.environ.get("POSTGRES_HOST", "localhost"),
            "port": os.environ.get("POSTGRES_PORT", 5432),
            "user": os.environ.get("POSTGRES_USER"),
            "password": os.environ.get("POSTGRES_PASSWORD"),
        }
        if creds["user"] and creds["password"]:
            return creds

        if dsn_or_db_path and Path(dsn_or_db_path).is_file():
            with open(dsn_or_db_path, "r") as f:
                from_file = json.load(f)
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
            creds = self._load_credentials(dsn_or_db_path)
            db_url = (
                f"postgresql://{creds['user']}:{creds['password']}"
                f"@{creds['host']}:{creds['port']}/{self.db_name}"
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
