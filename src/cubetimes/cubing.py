"""
Cubing-specific functions for calculating averages and analyzing solve times.
"""

from enum import Enum
from typing import TypedDict


class StatisticKind(Enum):
    SINGLE = ("Single", 1)
    AO5 = ("ao5", 5)
    AO12 = ("ao12", 12)
    AO100 = ("ao100", 100)

    def __init__(self, display_name: str, count: int):
        self.display_name = display_name
        self.count = count


class StatisticResult(TypedDict):
    kind: StatisticKind
    latest: float | None
    best: float | None
    best_position: int | None


class SummaryResult(TypedDict):
    total_solves: int
    average_time: float
    best_single: float
    worst_single: float


class AnalysisResult(TypedDict):
    statistics: list[StatisticResult]
    summary: SummaryResult


class AnalysisError(TypedDict):
    error: str


def calculate_average(times: list[float], count: int) -> float | None:
    """
    Calculate average of N solves using standard cubing rules.
    For ao5: drop best and worst, average the middle 3
    For ao12: drop best and worst, average the middle 10
    For ao100: drop best 5 and worst 5, average the middle 90

    DNF times (-1) are handled according to cubing rules:
    - For count <= 12: maximum 1 DNF allowed
    - For count > 12: maximum 5% of count (rounded down) DNFs allowed
    - If too many DNFs, the average is DNF
    - DNFs are treated as worst times and dropped when possible
    """
    if len(times) < count:
        return None

    # Take the most recent N times
    recent_times = times[:count]

    if count == 1:
        # For single, return -1 if it's a DNF
        return recent_times[0] if recent_times[0] != -1 else None

    # Count DNFs in the recent times
    dnf_count = sum(1 for t in recent_times if t == -1)

    # FIXME: max_dnfs and trim_count are the same value (1 or 5 %).
    # FIXME: This could be simplified by using a single variable.

    # Determine maximum allowed DNFs based on count
    if count <= 12:
        max_dnfs = 1
    else:
        max_dnfs = int(count * 0.05)  # 5% rounded down

    # If too many DNFs, the average is DNF (return None to indicate this)
    if dnf_count > max_dnfs:
        return None

    if count <= 3:
        # For small counts, just return the average (excluding DNFs)
        valid_times = [t for t in recent_times if t != -1]
        if not valid_times:
            return None
        return sum(valid_times) / len(valid_times)
    else:
        # Standard cubing average: remove best and worst, average the rest
        # DNFs are treated as worst times
        sorted_times = sorted(
            recent_times, key=lambda x: float("inf") if x == -1 else x
        )

        # For larger averages (count > 12), remove 5% from each end
        # For smaller averages, remove 1 from each end
        if count > 12:
            # Calculate 5% trim (rounded down)
            trim_count = int(count * 0.05)
            # Remove trim_count from each end
            middle_times = (
                sorted_times[trim_count:-trim_count] if trim_count > 0 else sorted_times
            )
        else:
            # Remove first (best) and last (worst) - traditional method
            middle_times = sorted_times[1:-1]

        # FIXME: After checking for max_dnfs, there should be no DNFs left
        # FIXME: in middle_times.

        # Filter out any remaining DNFs
        valid_middle_times = [t for t in middle_times if t != -1]

        if not valid_middle_times:
            return None

        return sum(valid_middle_times) / len(valid_middle_times)


def find_best_average(
    times: list[float], count: int
) -> tuple[float | None, int | None]:
    # FIXME: Better use a named tuple or dataclass for the return type.
    """
    Find the best average of N solves in the entire dataset.
    Returns (best_average, solve_number) where solve_number is 1-indexed.
    Properly handles DNF times according to cubing rules.
    """
    if len(times) < count:
        return None, None

    best_avg = None
    best_position = None

    # Check every possible consecutive sequence of N solves
    for i in range(len(times) - count + 1):
        sequence = times[i : i + count]
        avg = calculate_average(sequence, count)

        # Only consider valid averages (not DNF)
        if avg is not None and (best_avg is None or avg < best_avg):
            best_avg = avg
            best_position = (
                i + count
            )  # Position of the last solve in the sequence (1-indexed)

    return best_avg, best_position


def analyse_times(data: list[dict]) -> AnalysisResult | AnalysisError:
    """
    Analyse cube times and return statistics as a dictionary.
    Properly handles DNF times (-1) according to cubing rules.
    """
    # FIXME: Why not throw an exception instead of returning an error dict?

    # Extract times and convert to float, chronologically ordered (newest first)
    times = []
    for row in data:
        try:
            time_value = float(row.get("Time", 0))
            times.append(time_value)
        except (ValueError, TypeError):
            # FIXME: If this happens, no value is added to times.
            # FIXME: This will result in invalid indexes later.
            # FIXME: Better raise and fail?
            continue

    if not times:
        # FIXME: Raise an error?
        return AnalysisError(error="No valid times found for analysis.")

    # Calculate statistics for ao5, ao12, ao100
    statistics: list[StatisticResult] = []

    for stat_kind in StatisticKind:
        # Latest average
        latest_avg = calculate_average(times, stat_kind.count)

        # Best average
        best_avg, best_position = find_best_average(times, stat_kind.count)

        # Adjust position to count from oldest solve (solve #1) instead of newest
        # Since data is in descending order (newest first), we need to convert:
        # position_from_oldest = total_solves - position_from_newest + 1
        adjusted_best_position = None
        if best_position is not None:
            adjusted_best_position = len(times) - best_position + 1

        statistics.append(
            StatisticResult(
                kind=stat_kind,
                latest=latest_avg,
                best=best_avg,
                best_position=adjusted_best_position,
            )
        )

    # Calculate summary statistics (excluding DNFs for averages)
    valid_times = [t for t in times if t != -1]

    if not valid_times:
        # FIXME: Raise an error?
        return AnalysisError(error="No valid (non-DNF) times found for analysis.")

    # Return results
    return AnalysisResult(
        statistics=statistics,
        summary=SummaryResult(
            total_solves=len(times),
            average_time=sum(valid_times) / len(valid_times),
            best_single=min(valid_times),
            worst_single=max(valid_times),
        ),
    )
