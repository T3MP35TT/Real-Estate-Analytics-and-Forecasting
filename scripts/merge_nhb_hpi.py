"""Combine city-wise NHB RESIDEX exports into a city-by-quarter wide CSV.

The portal's .xls downloads are HTML tables. This script uses only Python's
standard library, so it does not require Excel, xlrd, openpyxl, or lxml.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "nhb"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "nhb_hpi_combined.csv"

CITY_STATE = {
    "Ahmedabad": "Gujarat", "Bengaluru": "Karnataka", "Bhiwadi": "Rajasthan",
    "Bhopal": "Madhya Pradesh", "Bhubaneswar": "Odisha",
    "Bidhan Nagar": "West Bengal", "Chakan": "Maharashtra",
    "Chandigarh": "Chandigarh / Punjab / Haryana", "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Dehradun": "Uttarakhand", "Delhi": "Delhi",
    "Faridabad": "Haryana", "Gandhinagar": "Gujarat", "Ghaziabad": "Uttar Pradesh",
    "Greater Noida": "Uttar Pradesh", "Gurugram": "Haryana", "Guwahati": "Assam",
    "Howrah": "West Bengal", "Hyderabad": "Telangana", "Indore": "Madhya Pradesh",
    "Jaipur": "Rajasthan", "Kalyan Dombivali": "Maharashtra", "Kanpur": "Uttar Pradesh",
    "Kochi": "Kerala", "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh",
    "Ludhiana": "Punjab", "Meerut": "Uttar Pradesh", "Mira Bhayander": "Maharashtra",
    "Mumbai": "Maharashtra", "Nagpur": "Maharashtra", "Nashik": "Maharashtra",
    "Navi Mumbai": "Maharashtra", "New Town Kolkata": "West Bengal",
    "Noida": "Uttar Pradesh", "Panvel": "Maharashtra", "Patna": "Bihar",
    "Pimpri Chinchwad": "Maharashtra", "Pune": "Maharashtra",
    "Raipur": "Chhattisgarh", "Rajkot": "Gujarat", "Ranchi": "Jharkhand",
    "Surat": "Gujarat", "Thane": "Maharashtra",
    "Thiruvananthapuram": "Kerala", "Vadodara": "Gujarat",
    "Vasai Virar": "Maharashtra", "Vijayawada": "Andhra Pradesh",
    "Vizag": "Andhra Pradesh",
}

OUTPUT_COLUMNS = [
    "City", "State", "Series_Type", "Quarter", "Quarter_End", "HPI_Assessment_Index",
    "Total_Received_Records", "Usable_Record_Count", "Usable_Records_Percent",
    "Composite_Price_Rs_Per_SqFt", "Price_Rs_Per_SqFt_Upto_646",
    "Price_Rs_Per_SqFt_646_to_1184", "Price_Rs_Per_SqFt_Over_1184",
    "Base_Year", "Source_View", "Source_File",
]
MEASURE_COLUMNS = OUTPUT_COLUMNS[5:13]


class TableRows(HTMLParser):
    """Read rows/cells from the HTML table returned with a .xls extension."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None


def _number(value):
    value = (value or "").replace(",", "").strip()
    if not value or value.lower() in {"n/a", "na", "-", "--"}:
        return ""
    try:
        return float(value)
    except ValueError:
        return ""


def _parse_file(path):
    city_match = re.match(r"(.+?)-City HPI Data (Current|Previous)-Q", path.stem)
    if not city_match:
        raise ValueError("filename should start '<City>-City HPI Data Current-Q' or 'Previous-Q'")
    city, view = city_match.groups()
    state = CITY_STATE.get(city)
    if state is None:
        raise ValueError(f"no state mapping configured for city {city!r}")

    parser = TableRows()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    rows = parser.rows
    if not rows:
        raise ValueError("no HTML table rows found")

    detailed = any("Total Received Records" in cell for cell in rows[0])
    has_hpi = any("HPI@Assessment Prices" in cell for cell in rows[0])
    if not detailed and not has_hpi:
        raise ValueError(f"unrecognized column headers: {rows[0]}")

    records = []
    for cells in rows[2 if detailed else 1:]:
        if len(cells) < 2 or not re.match(r"^[A-Za-z]{3} \d{4}$", cells[0]):
            continue
        try:
            q_date = datetime.strptime(cells[0], "%b %Y")
            q_end = q_date.date().replace(
                day=calendar.monthrange(q_date.year, q_date.month)[1]
            ).isoformat()
        except ValueError:
            continue
        row = dict.fromkeys(OUTPUT_COLUMNS, "")
        row.update({
            "City": city,
            "State": state,
            "Series_Type": "HPI and carpet-area prices" if detailed else "HPI@Assessment Prices",
            "Quarter": cells[0],
            "Quarter_End": q_end,
            "Base_Year": "2018",
            "Source_View": f"{view}-Q",
            "Source_File": path.name,
        })
        if detailed:
            if len(cells) < 9:
                raise ValueError(f"expected 9 data cells, found {len(cells)} in {cells[0]}")
            row.update({
                "Total_Received_Records": _number(cells[1]),
                "Usable_Record_Count": _number(cells[2]),
                "Usable_Records_Percent": _number(cells[3]),
                "HPI_Assessment_Index": _number(cells[4]),
                "Composite_Price_Rs_Per_SqFt": _number(cells[5]),
                "Price_Rs_Per_SqFt_Upto_646": _number(cells[6]),
                "Price_Rs_Per_SqFt_646_to_1184": _number(cells[7]),
                "Price_Rs_Per_SqFt_Over_1184": _number(cells[8]),
            })
        else:
            row["HPI_Assessment_Index"] = _number(cells[1])
        records.append(row)
    if not records:
        raise ValueError("no quarterly observations found")
    return records


def _merge_duplicates(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["City"], row["Series_Type"], row["Source_View"], row["Quarter_End"])].append(row)

    merged, duplicate_count, conflicts = [], 0, 0
    for key, group in sorted(groups.items()):
        duplicate_count += len(group) - 1
        # Prefer the copy containing the most actual measurements; use its
        # source filename as a deterministic tie-breaker.
        group.sort(key=lambda row: (-sum(row[c] != "" for c in MEASURE_COLUMNS), row["Source_File"]))
        chosen = group[0].copy()
        for other in group[1:]:
            for column in MEASURE_COLUMNS:
                left, right = chosen[column], other[column]
                if left == "" and right != "":
                    chosen[column] = right
                elif left != "" and right != "" and left != right:
                    conflicts += 1
                    print(
                    f"WARNING: conflicting {column} for {key[0]} / {key[1]} / {key[2]} / {key[3]}: "
                        f"kept {left!r} from {chosen['Source_File']}; ignored {right!r} "
                        f"from {other['Source_File']}",
                        file=sys.stderr,
                    )
        merged.append(chosen)
    return merged, duplicate_count, conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help=f"directory containing NHB .xls exports (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"merged CSV path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.xls"))
    if not files:
        parser.error(f"no .xls files found in {args.input_dir}")

    records, failures = [], []
    for path in files:
        try:
            records.extend(_parse_file(path))
        except (OSError, ValueError) as exc:
            failures.append((path.name, str(exc)))
    if failures:
        for name, error in failures:
            print(f"ERROR: {name}: {error}", file=sys.stderr)
        parser.error(f"{len(failures)} input file(s) could not be parsed; no CSV written")

    combined, duplicates, conflicts = _merge_duplicates(records)
    # Current-Q is the selected snapshot for this one-row-per-city output.
    current = [row for row in combined if row["Source_View"] == "Current-Q"]
    if not current:
        parser.error("no Current-Q observations found; cannot create the city-wide table")

    periods = sorted({(row["Quarter_End"], row["Quarter"]) for row in current})
    period_headers = [quarter for _, quarter in periods]
    city_values = defaultdict(dict)
    for row in current:
        city, period = row["City"], row["Quarter"]
        value = row["HPI_Assessment_Index"]
        if period in city_values[city] and city_values[city][period] != value:
            parser.error(
                f"conflicting Current-Q HPI values for {city} / {period}; "
                "refusing to choose one silently"
            )
        city_values[city][period] = value

    headers = ["City", *period_headers]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        for city in sorted(city_values):
            writer.writerow([city, *(city_values[city].get(period, "")
                                     for period in period_headers)])

    print(f"Input files: {len(files)}")
    print(f"Parsed observations: {len(records)}")
    print(f"Duplicate city-series-view-quarter rows collapsed: {duplicates}")
    print(f"Conflicting measurements encountered: {conflicts}")
    print(f"Source view: Current-Q")
    print(f"Cities: {len(city_values)}")
    print(f"Quarter columns: {len(period_headers)} ({period_headers[0]} to {period_headers[-1]})")
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
