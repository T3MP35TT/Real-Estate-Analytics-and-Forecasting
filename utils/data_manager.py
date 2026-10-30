from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import ARTIFACT_DIR, DATA_PATH, HPI_PATH, REPORT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "managed_data.db"

DATASETS = {
    "Property Listings": {
        "path": DATA_PATH,
        "table": "property_listings",
        "editable": True,
        "description": "Property listing source records. Changes are stored in the managed database until explicitly exported.",
    },
    "NHB HPI Historical Data": {
        "path": HPI_PATH,
        "table": "hpi_historical",
        "editable": True,
        "description": "Original wide-format NHB RESIDEX city-by-period source data.",
    },
    "NHB HPI Forecast Data": {
        "path": ARTIFACT_DIR / "hpi_forecast_by_city.csv",
        "table": "hpi_forecast",
        "editable": False,
        "description": "Generated model output. Read-only here; regenerate using scripts/train.py.",
    },
    "Forecast CV Results": {
        "path": REPORT_DIR / "hpi_forecast_cv_results.csv",
        "table": "forecast_cv_results",
        "editable": False,
        "description": "Forecast model comparison report. Read-only generated evaluation output.",
    },
    "Forecast Holdout Predictions": {
        "path": REPORT_DIR / "hpi_forecast_holdout_predictions.csv",
        "table": "forecast_holdout_predictions",
        "editable": False,
        "description": "Temporal holdout evaluation output. Read-only generated report.",
    },
}

@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _safe_table(table: str) -> str:
    if table not in {v["table"] for v in DATASETS.values()}:
        raise ValueError("Unknown dataset table.")
    return table

def _load_source(dataset_name: str) -> pd.DataFrame:
    cfg = DATASETS[dataset_name]
    path = Path(cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Configured source file not found: {path}")
    return pd.read_csv(path)

def initialize_dataset(dataset_name: str, force_reload: bool = False) -> tuple[int, bool]:
    cfg = DATASETS[dataset_name]
    table = _safe_table(cfg["table"])
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] if exists else 0
        if exists and count and not force_reload:
            return int(count), False
        frame = _load_source(dataset_name)
        if frame.empty:
            raise ValueError("The selected source dataset is empty.")
        if force_reload and exists:
            conn.execute(f'DROP TABLE "{table}"')
        frame.to_sql(table, conn, if_exists="replace", index=False)
        _audit(conn, dataset_name, "INITIALIZE" if not exists else "RELOAD", None,
               {"rows": len(frame), "source": str(cfg["path"])})
        return len(frame), True

def read_dataset(dataset_name: str, search: str = "", limit: int = 1000) -> tuple[pd.DataFrame, int]:
    table = _safe_table(DATASETS[dataset_name]["table"])
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            initialize_dataset(dataset_name)
        total = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        if search.strip():
            cols = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            clauses = " OR ".join(f'CAST("{c}" AS TEXT) LIKE ?' for c in cols)
            params = [f"%{search.strip()}%"] * len(cols)
            frame = pd.read_sql_query(
                f'SELECT rowid AS "_record_id", * FROM "{table}" WHERE {clauses} LIMIT ?',
                conn, params=[*params, int(limit)]
            )
        else:
            frame = pd.read_sql_query(
                f'SELECT rowid AS "_record_id", * FROM "{table}" LIMIT ?',
                conn, params=[int(limit)]
            )
    return frame, total

def _audit(conn, dataset: str, action: str, record_id: int | None, details: dict[str, Any]):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS change_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            changed_at TEXT NOT NULL,
            dataset TEXT NOT NULL,
            action TEXT NOT NULL,
            record_id INTEGER,
            details_json TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO change_audit(changed_at,dataset,action,record_id,details_json) VALUES(?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), dataset, action,
         record_id, json.dumps(details, default=str))
    )

def _table_columns(conn, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]

def create_record(dataset_name: str, values: dict[str, Any]):
    cfg = DATASETS[dataset_name]
    if not cfg["editable"]:
        raise PermissionError("This generated dataset is read-only.")
    table = _safe_table(cfg["table"])
    with _connect() as conn:
        columns = _table_columns(conn, table)
        clean = {k: v for k, v in values.items() if k in columns}
        if not clean:
            raise ValueError("No valid fields were supplied.")
        cols = list(clean)
        marks = ",".join("?" for _ in cols)
        quoted = ",".join(f'"{c}"' for c in cols)
        cur = conn.execute(
            f'INSERT INTO "{table}" ({quoted}) VALUES ({marks})',
            [clean[c] for c in cols]
        )
        rid = int(cur.lastrowid)
        _audit(conn, dataset_name, "CREATE", rid, clean)

def update_record(dataset_name: str, record_id: int, values: dict[str, Any]):
    cfg = DATASETS[dataset_name]
    if not cfg["editable"]:
        raise PermissionError("This generated dataset is read-only.")
    table = _safe_table(cfg["table"])
    with _connect() as conn:
        columns = _table_columns(conn, table)
        clean = {k: v for k, v in values.items() if k in columns}
        if not clean:
            raise ValueError("No valid fields were supplied.")
        assignments = ", ".join(f'"{c}"=?' for c in clean)
        conn.execute(
            f'UPDATE "{table}" SET {assignments} WHERE rowid=?',
            [*clean.values(), int(record_id)]
        )
        _audit(conn, dataset_name, "UPDATE", int(record_id), clean)

def delete_record(dataset_name: str, record_id: int):
    cfg = DATASETS[dataset_name]
    if not cfg["editable"]:
        raise PermissionError("This generated dataset is read-only.")
    table = _safe_table(cfg["table"])
    with _connect() as conn:
        old = conn.execute(f'SELECT * FROM "{table}" WHERE rowid=?', (int(record_id),)).fetchone()
        if old is None:
            raise KeyError(f"Record {record_id} no longer exists.")
        conn.execute(f'DELETE FROM "{table}" WHERE rowid=?', (int(record_id),))
        _audit(conn, dataset_name, "DELETE", int(record_id), {"previous_values": old})

def import_csv(dataset_name: str, uploaded_file, append: bool = True) -> int:
    cfg = DATASETS[dataset_name]
    if not cfg["editable"]:
        raise PermissionError("This generated dataset is read-only.")
    incoming = pd.read_csv(uploaded_file)
    if incoming.empty:
        raise ValueError("The uploaded CSV contains no records.")
    table = _safe_table(cfg["table"])
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if append and exists:
            current_cols = _table_columns(conn, table)
            if list(incoming.columns) != current_cols:
                raise ValueError("CSV columns must exactly match the selected dataset schema to append safely.")
            incoming.to_sql(table, conn, if_exists="append", index=False)
        else:
            incoming.to_sql(table, conn, if_exists="replace", index=False)
        _audit(conn, dataset_name, "IMPORT_APPEND" if append and exists else "IMPORT_REPLACE",
               None, {"rows": len(incoming), "columns": list(incoming.columns)})
    return len(incoming)

def export_dataset(dataset_name: str) -> pd.DataFrame:
    table = _safe_table(DATASETS[dataset_name]["table"])
    with _connect() as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)

def audit_log(dataset_name: str | None = None, limit: int = 200) -> pd.DataFrame:
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='change_audit'"
        ).fetchone()
        if not exists:
            return pd.DataFrame(columns=["changed_at","dataset","action","record_id","details_json"])
        if dataset_name:
            return pd.read_sql_query(
                "SELECT * FROM change_audit WHERE dataset=? ORDER BY audit_id DESC LIMIT ?",
                conn, params=[dataset_name, limit]
            )
        return pd.read_sql_query(
            "SELECT * FROM change_audit ORDER BY audit_id DESC LIMIT ?", conn, params=[limit]
        )

def replace_from_editor(dataset_name: str, edited: pd.DataFrame):
    cfg = DATASETS[dataset_name]
    if not cfg["editable"]:
        raise PermissionError("This generated dataset is read-only.")
    table = _safe_table(cfg["table"])
    clean = edited.drop(columns=["_record_id"], errors="ignore")
    with _connect() as conn:
        current_cols = _table_columns(conn, table)
        if list(clean.columns) != current_cols:
            raise ValueError("The edited table schema changed; reload before saving.")
        # Use explicit row IDs to update records and create new rows.
        existing = set(
            row[0] for row in conn.execute(f'SELECT rowid FROM "{table}"').fetchall()
        )
        seen = set()
        for _, row in edited.iterrows():
            rid = row.get("_record_id")
            vals = {c: (None if pd.isna(row[c]) else row[c]) for c in current_cols}
            if pd.isna(rid):
                cols = list(vals)
                conn.execute(
                    f'INSERT INTO "{table}" ({",".join(f""" "{c}" """ for c in cols)}) VALUES ({",".join("?" for _ in cols)})',
                    list(vals.values())
                )
                _audit(conn, dataset_name, "CREATE", None, vals)
            else:
                rid = int(rid)
                if rid not in existing:
                    continue
                seen.add(rid)
                assignments = ", ".join(f'"{c}"=?' for c in current_cols)
                conn.execute(
                    f'UPDATE "{table}" SET {assignments} WHERE rowid=?',
                    [*vals.values(), rid]
                )
                _audit(conn, dataset_name, "UPDATE", rid, vals)
