"""
Data file importer for cube times data.
Supports CSV and JSON file formats.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


class ImporterError(Exception):
    """Custom exception for cube times processing errors."""

    # FIXME: Remove "pass"
    pass


def read_json_file(file_path: str) -> tuple[list[dict], list[str]]:
    """Read the JSON file and return a list of dictionaries from all sessions."""
    # FIXME: Use a TypedDict or dataclass(es) for the return type.
    if not os.path.exists(file_path):
        raise ImporterError(f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as jsonfile:
            json_data = json.load(jsonfile)
    except json.JSONDecodeError as e:
        raise ImporterError(f"Error parsing JSON file: {e}")

    # Validate JSON structure - should be a dictionary with session data
    if not isinstance(json_data, dict):
        raise ImporterError("JSON file must contain a dictionary with session data")

    data = []
    fieldnames = ["Time", "Date", "Scramble"]

    # Find all session keys (session1, session2, etc.)
    session_keys = [key for key in json_data.keys() if key.startswith("session")]

    if not session_keys:
        raise ImporterError("No session data found in JSON file")

    solve_number = 0  # For error reporting

    # Process all sessions
    for session_key in sorted(session_keys):  # Sort to ensure consistent order
        session_data = json_data[session_key]

        if not isinstance(session_data, list):
            raise ImporterError(
                f"Session '{session_key}' must contain a list of solves"
            )

        for solve_record in session_data:
            solve_number += 1

            # Validate solve record structure
            if not isinstance(solve_record, list) or len(solve_record) < 4:
                raise ImporterError(
                    # FIXME: Should report the session number and solve number within the session.
                    f"Solve {solve_number}: expected list with at least 4 elements [times, scramble, comment, date]"
                )

            times, scramble, comment, date_seconds = solve_record[:4]

            # Extract time information
            if not isinstance(times, list) or len(times) != 2:
                raise ImporterError(
                    # FIXME: Should report the session number and solve number within the session.
                    f"Solve {solve_number}: 'times' must be a list with exactly 2 elements"
                )

            penalty, time_ms = times

            # Handle different penalty values
            if penalty == -1:
                # DNF (Did Not Finish)
                solve_time = -1
            elif penalty == 0:
                # Normal solve
                solve_time = time_ms / 1000.0  # Convert milliseconds to seconds
            else:
                # Penalty (usually 2000ms = 2 seconds)
                solve_time = (
                    time_ms + penalty
                ) / 1000.0  # Add penalty and convert to seconds

            # Validate and convert date
            if date_seconds is None:
                raise ImporterError(f"Solve {solve_number}: missing date")

            try:
                # Convert seconds since epoch to timezone-aware datetime in UTC
                parsed_date = datetime.fromtimestamp(date_seconds, tz=ZoneInfo("UTC"))
            except (ValueError, TypeError) as e:
                raise ImporterError(
                    f"Solve {solve_number}: error parsing date '{date_seconds}' - {e}"
                )

            # Create row dictionary similar to CSV format
            row = {
                "Time": str(solve_time),
                # FIXME: Just store the date object and omit the string representation.
                "Date": parsed_date.isoformat(),
                "Scramble": scramble or "",
                "parsed_date": parsed_date,
            }

            data.append(row)

    return data, fieldnames


def read_csv_file(file_path: str) -> tuple[list[dict], list[str]]:
    """Read the CSV file and return a list of dictionaries."""
    # FIXME: Use a TypedDict or dataclass(es) for the return type.

    if not os.path.exists(file_path):
        raise ImporterError(f"File not found: {file_path}")

    data = []
    with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        # Check if required columns exist
        required_columns = ["Time", "Date"]
        fieldnames = list(reader.fieldnames or [])
        if not all(col in fieldnames for col in required_columns):
            missing_columns = [col for col in required_columns if col not in fieldnames]
            raise ImporterError(f"Missing required columns: {missing_columns}")

        line_number = 1  # Start at 1 for header
        for row in reader:
            line_number += 1

            # Parse and validate the date
            date_str = row.get("Date", "")
            if not date_str:
                raise ImporterError(f"Empty Date field on line {line_number}")

            try:
                # Parse the date with timezone awareness
                parsed_date = datetime.fromisoformat(date_str)

                # Ensure the datetime is timezone-aware (UTC if no timezone specified)
                if parsed_date.tzinfo is None:
                    from zoneinfo import ZoneInfo

                    parsed_date = parsed_date.replace(tzinfo=ZoneInfo("UTC"))

                # FIXME: Just store the date object and omit the string representation.
                row["parsed_date"] = parsed_date
            except ValueError as e:
                raise ImporterError(
                    f"Error parsing date on line {line_number}: '{date_str}' - {e}"
                )

            data.append(row)

    return data, fieldnames


def read_file(file_path: str) -> tuple[list[dict], list[str]]:
    """Read a file and return a list of dictionaries based on file extension."""
    # FIXME: Use a TypedDict or dataclass(es) for the return type.

    if not os.path.exists(file_path):
        raise ImporterError(f"File not found: {file_path}")

    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".csv":
        return read_csv_file(file_path)
    elif file_extension == ".json":
        return read_json_file(file_path)
    else:
        raise ImporterError(
            f"Unsupported file format: {file_extension}. Supported formats: .csv, .json"
        )
