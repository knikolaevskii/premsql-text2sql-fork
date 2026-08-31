import json
import time
from typing import Any, Dict

import pandas as pd
from func_timeout import func_timeout
from sqlalchemy import create_engine

from premsql.executors.base import BaseExecutor
from premsql.logger import setup_console_logger

logger = setup_console_logger(name="[POSTGRES-EXEC]")


class PostgresExecutor(BaseExecutor):
    """
    Executes SQL against a PostgreSQL database. `dsn_or_db_path` passed to
    execute_sql is the path to a JSON credentials file:

        {"host": ..., "port": ..., "user": ..., "password": ...}

    `db_name` is fixed per-executor instance since a single evaluation run
    targets one database.
    """

    def __init__(self, db_name: str, query_timeout: float = 10.0) -> None:
        self.db_name = db_name
        self.query_timeout = query_timeout

    def execute_sql(self, sql: str, dsn_or_db_path: str) -> Dict[str, Any]:
        start_time = time.time()
        result = None
        df = pd.DataFrame()
        error = None

        try:
            with open(dsn_or_db_path, "r") as f:
                creds = json.load(f)

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
