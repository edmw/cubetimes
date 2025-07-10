"""
Plotting functionality for cube times using matplotlib.
"""

import logging
from enum import Enum
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .cubing import StatisticKind, analyse_times


class PlotColors(Enum):
    """
    Color constants for plotting cube times with professional color palette.

    Each color is defined as RGB values normalized to [0,1] range for matplotlib.
    Colors are chosen for good contrast and visual appeal in dark mode theme.
    """

    DATA_POINTS = (55 / 255, 158 / 255, 253 / 255)  # Bright Blue
    DNF_MARKERS = (255 / 255, 95 / 255, 89 / 255)  # Coral Red
    AO5_LINE = (193 / 255, 147 / 255, 128 / 255)  # Sandy Brown
    AO12_LINE = (150 / 255, 117 / 255, 107 / 255)  # Brownish Gray
    AO100_LINE = (69 / 255, 29 / 255, 25 / 255)  # Dark Brown
    ROLLING_AVERAGE = (249 / 255, 167 / 255, 70 / 255)  # Golden Orange
    SIGMOID_CURVE = (245 / 255, 206 / 255, 74 / 255)  # Pale Gold
    REFERENCE_LINE = (110 / 255, 204 / 255, 82 / 255)  # Lime Green

    # Reserved colors for future use
    LAVENDER = (209 / 255, 137 / 255, 226 / 255)  # Light Purple
    MAGENTA = (225 / 255, 36 / 255, 229 / 255)  # Bright Magenta
    CYAN = (70 / 255, 226 / 255, 226 / 255)  # Bright Cyan


def sigmoid_function(x, a, b, c, d):
    """Sigmoid function: a / (1 + exp(-b * (x - c))) + d"""
    return a / (1 + np.exp(-b * (x - c))) + d


def fit_sigmoid_simple(x, y):
    """Simple sigmoid fitting using basic parameter estimation"""
    try:
        # Normalize x to [0, 1] range for better fitting
        # FIXME: Why is this not used adterwards? Use it or remove it.
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x

        # Estimate initial parameters
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min

        # Simple parameter estimation
        a = -y_range  # Negative range for decreasing curve
        b = 10  # Steepness
        c = 0.5  # Middle point
        d = y_max  # Offset

        # Generate sigmoid curve
        x_smooth = np.linspace(0, 1, len(x) * 5)
        y_smooth = sigmoid_function(x_smooth, a, b, c, d)

        # Convert back to original x scale
        x_smooth_orig = x.min() + x_smooth * (x.max() - x.min())

        return x_smooth_orig, y_smooth
    except Exception:
        return None, None


class PlotterError(Exception):
    """Exception raised for plotting-related errors."""

    # FIXME: Remove "pass"
    pass


def plot_times_over_date(
    data: list[dict],
    output_file: str | None = None,
    rolling_average: int = 0,
    sigmoid_curve: bool = False,
    subx_seconds: float = 60.0,
    by_solve_number: bool = False,
) -> Figure:
    """
    Plot cube times over date or solve number using matplotlib.

    Args:
        data: List of dictionaries containing cube time data with 'Time' and 'parsed_date' keys
        output_file: Optional path to save the plot to a file
        rolling_average: If > 0, add a rolling average line with this window size
        sigmoid_curve: If True, add a sigmoid curve fit to the data
        subx_seconds: Value in seconds for horizontal reference line (default: 60.0)
        by_solve_number: If True, plot by solve number instead of date (default: False)

    Returns:
        matplotlib Figure object

    Raises:
        PlotterError: If there's an error creating or saving the plot
    """
    if not data:
        raise PlotterError("No data to plot")

    try:
        # FIXME: Break into smaller functions for better readability.
        # FIXME: Or - maybe even better - break into separate modules.

        # Extract dates/solve numbers and times
        dates = []
        times = []
        dnf_dates = []  # Separate list for DNF dates/solve numbers to plot differently
        solve_numbers = []
        dnf_solve_numbers = []

        for i, row in enumerate(data):
            if "parsed_date" not in row or "Time" not in row:
                continue

            try:
                time_value = float(row["Time"])
                date_value = row["parsed_date"]
                solve_number = i + 1  # 1-indexed solve number

                if time_value == -1:
                    # DNF time - store separately for different visualization
                    if by_solve_number:
                        dnf_solve_numbers.append(solve_number)
                    else:
                        dnf_dates.append(date_value)
                else:
                    if by_solve_number:
                        solve_numbers.append(solve_number)
                    else:
                        dates.append(date_value)
                    times.append(time_value)
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping invalid data row: {e}")
                continue

        if not times and not (dnf_dates or dnf_solve_numbers):
            raise PlotterError("No valid time/date data found")

        # Use solve numbers or dates for x-axis
        x_values = solve_numbers if by_solve_number else dates
        dnf_x_values = dnf_solve_numbers if by_solve_number else dnf_dates

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

        # Set dark mode styling
        # FIXME: Make ALL colors into constants.
        fig.patch.set_facecolor("#2E2E2E")  # Dark gray figure background
        ax.set_facecolor("white")  # White plot area background

        # Set dark mode for axes, labels, and text
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.tick_params(colors="white", which="both")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        # Plot valid times
        if x_values and times:
            # Adjust marker size and alpha based on whether rolling average is shown
            markersize = 2 if rolling_average > 0 else 3
            alpha = 0.5 if rolling_average > 0 else 0.7
            label = "Individual Times" if rolling_average > 0 else "Solve times"

            ax.plot(
                x_values,
                times,
                "o",
                markersize=markersize,
                alpha=alpha,
                color=PlotColors.DATA_POINTS.value,
                label=label,
            )

            # Add rolling average if requested
            if rolling_average > 0:
                # Calculate rolling average (excluding DNFs)
                rolling_avg = []
                for i in range(len(times)):
                    start_idx = max(0, i - rolling_average + 1)
                    window_times = times[start_idx : i + 1]
                    rolling_avg.append(sum(window_times) / len(window_times))

                # Plot rolling average line with heavy smoothing
                if len(x_values) >= 3:  # Need at least 3 points for interpolation
                    if by_solve_number:
                        # For solve numbers, use direct numeric values
                        x_numeric = np.array(x_values)
                    else:
                        # Convert dates to numeric values for interpolation
                        x_numeric = np.array([mdates.date2num(d) for d in x_values])

                    rolling_avg_array = np.array(rolling_avg)

                    # Create many more points for very smooth curve (10x more points)
                    x_smooth = np.linspace(
                        x_numeric.min(), x_numeric.max(), len(x_values) * 10
                    )
                    rolling_avg_smooth = np.interp(
                        x_smooth, x_numeric, rolling_avg_array
                    )

                    # Apply additional smoothing using a simple moving average on the interpolated data
                    if len(rolling_avg_smooth) > 5:
                        # Apply a smoothing window to the interpolated data
                        window_size = max(
                            3, len(rolling_avg_smooth) // 20
                        )  # Adaptive window size
                        smoothed_values = []
                        for i in range(len(rolling_avg_smooth)):
                            start_idx = max(0, i - window_size // 2)
                            end_idx = min(
                                len(rolling_avg_smooth), i + window_size // 2 + 1
                            )
                            window_values = rolling_avg_smooth[start_idx:end_idx]
                            smoothed_values.append(np.mean(window_values))
                        rolling_avg_smooth = np.array(smoothed_values)

                    # Convert back to appropriate x-axis values for plotting
                    if by_solve_number:
                        x_smooth_plot = x_smooth
                    else:
                        x_smooth_plot = mdates.num2date(x_smooth)

                    ax.plot(
                        x_smooth_plot,
                        rolling_avg_smooth,
                        "-",
                        linewidth=1.5,
                        color=PlotColors.ROLLING_AVERAGE.value,
                        alpha=0.8,
                        label=f"Rolling Average (n={rolling_average})",
                    )
                else:
                    # Fallback for very few data points
                    ax.plot(
                        x_values,
                        rolling_avg,
                        "-",
                        linewidth=1.5,
                        color=PlotColors.ROLLING_AVERAGE.value,
                        alpha=0.8,
                        label=f"Rolling Average (n={rolling_average})",
                    )

        # Add sigmoid curve if requested
        if sigmoid_curve and x_values and times and len(x_values) >= 5:
            if by_solve_number:
                # Use solve numbers directly
                x_numeric = np.array(x_values)
            else:
                # Convert dates to numeric values for fitting
                x_numeric = np.array([mdates.date2num(d) for d in x_values])

            times_array = np.array(times)

            # Fit sigmoid curve
            x_smooth, y_smooth = fit_sigmoid_simple(x_numeric, times_array)

            if x_smooth is not None and y_smooth is not None:
                if by_solve_number:
                    x_sigmoid_plot = x_smooth
                else:
                    # Convert back to datetime objects
                    x_sigmoid_plot = mdates.num2date(x_smooth)

                ax.plot(
                    x_sigmoid_plot,
                    y_smooth,
                    "-",
                    linewidth=1.5,
                    color=PlotColors.SIGMOID_CURVE.value,
                    alpha=0.7,
                    label="Sigmoid Trend",
                )

        # Plot DNF times as red X markers on the x-axis
        if dnf_x_values:
            # Position DNF markers at y=0 (on the x-axis)
            dnf_heights = [0] * len(dnf_x_values)
            ax.scatter(
                dnf_x_values,
                dnf_heights,
                marker="x",
                s=80,
                color=PlotColors.DNF_MARKERS.value,
                alpha=0.9,
                label="DNF",
                zorder=10,  # Ensure DNF markers appear on top
            )

        # Add vertical lines for best averages
        # We need the data sorted in descending order for analysis (newest first)
        analysis_data = sorted(data, key=lambda row: row["parsed_date"], reverse=True)
        analysis_results = analyse_times(analysis_data)

        if not isinstance(analysis_results, dict) or "error" not in analysis_results:
            # Analysis succeeded, add vertical lines for best averages
            for stat in analysis_results["statistics"]:
                if (
                    stat["kind"]
                    in [StatisticKind.AO5, StatisticKind.AO12, StatisticKind.AO100]
                    and stat["best_position"] is not None
                ):
                    # Convert position back to index in ascending order data
                    # stat["best_position"] is counted from oldest (1-indexed)
                    # data is sorted ascending (oldest first), so position directly maps to index
                    position_index = stat["best_position"] - 1  # Convert to 0-indexed

                    if 0 <= position_index < len(data):
                        if by_solve_number:
                            # Use solve number (1-indexed) for x-coordinate
                            x_coordinate = stat["best_position"]
                        else:
                            # Use date for x-coordinate
                            x_coordinate = data[position_index]["parsed_date"]

                        # Choose colors for different averages
                        color_map = {
                            StatisticKind.AO5: (PlotColors.AO5_LINE.value, "Best ao5"),
                            StatisticKind.AO12: (
                                PlotColors.AO12_LINE.value,
                                "Best ao12",
                            ),
                            StatisticKind.AO100: (
                                PlotColors.AO100_LINE.value,
                                "Best ao100",
                            ),
                        }

                        if stat["kind"] in color_map:
                            color, label = color_map[stat["kind"]]
                            ax.axvline(
                                x=x_coordinate,
                                color=color,
                                linestyle="--",
                                alpha=0.7,
                                linewidth=2,
                                label=f"{label} (solve #{stat['best_position']})",
                            )

        # Add horizontal line at specified seconds for reference
        ax.axhline(
            y=subx_seconds,
            color=PlotColors.REFERENCE_LINE.value,
            linestyle=":",
            alpha=0.6,
            linewidth=2,
            label=f"{subx_seconds:.0f} seconds",
        )

        # Formatting
        if by_solve_number:
            ax.set_xlabel("Solve Number", fontsize=12)
        else:
            ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Solve Time (seconds)", fontsize=12)

        # Set title based on what features are shown
        title_parts = ["Cube Solve Times"]
        if by_solve_number:
            title_parts.append("by Solve Number")
        else:
            title_parts.append("Over Time")
        if rolling_average > 0:
            title_parts.append(f"with {rolling_average}-Solve Rolling Average")
        if sigmoid_curve:
            title_parts.append(
                "and Sigmoid Trend" if rolling_average > 0 else "with Sigmoid Trend"
            )

        title = " ".join(title_parts)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, color="#2E2E2E")  # Dark grid lines on white background

        # Add legend if there are DNFs or vertical lines
        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=True,
            fancybox=False,
            shadow=False,
            borderpad=0.5,
            columnspacing=0.8,
            handletextpad=0.5,
        )
        legend.get_frame().set_facecolor("white")  # Same as diagram background
        legend.get_frame().set_edgecolor("#2E2E2E")  # Dark legend border
        legend.get_frame().set_linewidth(1)  # Thin border
        legend.get_frame().set_alpha(1.0)  # No transparency
        for text in legend.get_texts():
            text.set_color("#2E2E2E")  # Dark legend text

        # Format x-axis based on plot type
        if not by_solve_number:
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            # Rotate date labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add some statistics as text (excluding DNFs)
        if times:
            avg_time = sum(times) / len(times)
            best_time = min(times)
            worst_time = max(times)
            total_solves = len(times) + len(dnf_x_values)
            dnf_count = len(dnf_x_values)

            stats_text = f"Average: {avg_time:.2f}s | Best: {best_time:.2f}s | Worst: {worst_time:.2f}s | Solves: {total_solves}"
            if dnf_count > 0:
                stats_text += f" | DNFs: {dnf_count}"

            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                color="white",
                bbox=dict(
                    # boxstyle="round",
                    facecolor="#2E2E2E",
                    alpha=0.8,
                    edgecolor="white",
                ),
            )

        # Tight layout to prevent label cutoff
        plt.tight_layout()

        # FIXME: Separate saving logic from plotting.
        # FIXME: Maybe handle save in main.py.
        # Save to file if specified
        if output_file:
            try:
                # Ensure directory exists
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save with high DPI for quality
                fig.savefig(output_file, dpi=300, bbox_inches="tight")
                logging.info(f"Plot saved to {output_file}")
            except Exception as e:
                raise PlotterError(f"Failed to save plot to {output_file}: {e}")

        return fig

    except Exception as e:
        if isinstance(e, PlotterError):
            raise
        raise PlotterError(f"Error creating plot: {e}")
