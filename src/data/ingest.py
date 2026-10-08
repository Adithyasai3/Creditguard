"""Ingest raw CSVs from data/raw/ into PostgreSQL tables.

Run:  python -m src.data.ingest
Uses psycopg2 COPY for speed. Idempotent — truncates and reloads.
Tables are created from the CSV headers (all columns TEXT, cast to proper
types downstream in the feature-engineering SQL).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import psycopg2
from psycopg2 import sql

from src.utils.config import DATABASE_URL, DATA_RAW, REQUIRED_TABLES


def _parse_dsn(url: str) -> dict:
    """Parse a postgresql://user:pass@host:port/dbname URL."""
    from urllib.parse import urlparse

    p = urlparse(url)
    return {
        "user": p.username,
        "password": p.password,
        "host": p.hostname,
        "port": p.port or 5432,
        "dbname": p.path.lstrip("/"),
    }


def _connect():
    return psycopg2.connect(**_parse_dsn(DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")))


def _table_columns(csv_path: Path) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _create_table(conn, table: str, columns: list[str]) -> None:
    quoted = ", ".join(f'"{c}" TEXT' for c in columns)
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS creditguard;")
        cur.execute(f'DROP TABLE IF EXISTS creditguard."{table}";')
        cur.execute(f'CREATE TABLE creditguard."{table}" ({quoted});')
    conn.commit()


def _copy_csv(conn, csv_path: Path, table: str) -> int:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    cols = ", ".join(f'"{c}"' for c in header)
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for row in rows:
        buf.write(",".join(str(v) if v != "" else "\\N" for v in row) + "\n")
    buf.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            sql.SQL(f'COPY creditguard."{table}" ({cols}) FROM STDIN WITH (FORMAT csv, NULL \'\\N\')').as_string(conn),
            buf,
        )
    conn.commit()
    return len(rows)


def ingest() -> None:
    conn = _connect()
    try:
        for table in REQUIRED_TABLES:
            csv_path = DATA_RAW / f"{table}.csv"
            if not csv_path.exists():
                print(f"SKIP (missing): {csv_path.name}")
                continue
            columns = _table_columns(csv_path)
            _create_table(conn, table, columns)
            n = _copy_csv(conn, csv_path, table)
            print(f"loaded {table}: {n:,} rows")
    finally:
        conn.close()
    print("Ingest complete.")


if __name__ == "__main__":
    ingest()