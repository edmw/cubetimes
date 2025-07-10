"""
CubeTimes - A comprehensive tool for analyzingand plotting cube solve times.
"""

# FIXME: Add pytest and tests for all modules.

import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table

from .cubing import AnalysisError, AnalysisResult, analyse_times
from .importer import ImporterError, read_file
from .plotter import PlotterError, plot_times_over_date


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze and plot cube solve times from data file"
    )

    parser.add_argument(
        "data_file",
        nargs="?",
        default="cubetimes.json",
        help="Path to the data file (default: cubetimes.json)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (default: disabled)",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="COMMAND"
    )

    # Show subcommand

    show_parser = subparsers.add_parser(
        "show",
        help="Display solve times in a table format",
    )
    show_parser.add_argument(
        "-o", "--output", help="Save table to file (default: print to stdout)"
    )
    show_parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (oldest first, default: newest first)",
    )

    # Analyse subcommand

    analyse_parser = subparsers.add_parser(
        "analyse", help="Analyze solve times and show statistics (ao5, ao12, ao100)"
    )

    # Plot subcommand

    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate plots of solve times",
    )
    plot_parser.add_argument(
        "-o", "--output", help="Save plot to file (default: show interactive plot)"
    )
    plot_parser.add_argument(
        "--rolling-average",
        type=int,
        metavar="N",
        help="Plot rolling average with window size N",
    )
    plot_parser.add_argument(
        "--sigmoid",
        action="store_true",
        help="Add sigmoid trend curve to the plot",
    )
    plot_parser.add_argument(
        "--subX",
        type=float,
        metavar="SECONDS",
        default=60.0,
        help="Add horizontal reference line at specified seconds (default: 60)",
    )
    plot_parser.add_argument(
        "-s",
        "--by-solve-number",
        action="store_true",
        help="Plot by solve number instead of date (x-axis shows solve progression)",
    )

    args = parser.parse_args()
    if args.command is None:
        args.command = "show"

    return args


def sort_by_date(data, ascending=False):
    """Sort the data by date column."""
    # Sort using the pre-parsed dates
    sorted_data = sorted(
        data, key=lambda row: row["parsed_date"], reverse=not ascending
    )

    return sorted_data


def display_data_table(data, fieldnames, output_file=None):

    table_fieldnames = [
        field for field in fieldnames if field not in ["parsed_date", "Comment"]
    ]

    console = Console()

    table = Table(
        show_header=True, header_style="bold magenta", title="Cube Solve Times"
    )
    for field in table_fieldnames:
        if field == "Time":
            table.add_column(field, justify="right", style="cyan")
        elif field == "Date":
            table.add_column(field, justify="left", style="green")
        else:
            table.add_column(field, justify="left")

    for row in data:
        row_values = []
        for field in table_fieldnames:
            value = row.get(field, "")
            if field == "Time":
                try:
                    time_value = float(value)
                    if time_value == -1:
                        value = "DNF"
                    else:
                        value = f"{time_value:.2f}"
                except (ValueError, TypeError):
                    pass
            elif field == "Date":
                try:
                    parsed_date = row["parsed_date"]
                    # use local time
                    local_date = parsed_date.astimezone()
                    value = local_date.strftime("%Y-%m-%d %H:%M")
                except (KeyError, AttributeError):
                    pass
            row_values.append(str(value))
        table.add_row(*row_values)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            file_console = Console(file=f, width=120)
            file_console.print(table)
        logging.info(f"Data saved to {output_file}")
    else:
        console.print(table)


def display_analysis_results(results: AnalysisResult | AnalysisError) -> None:

    console = Console()

    if "error" in results:
        print(results["error"])
        return

    table = Table(
        show_header=True, header_style="bold magenta", title="Cube Solve Times Analysis"
    )
    table.add_column("Statistic", style="cyan", width=15)
    table.add_column("Latest", justify="right", style="green", width=12)
    table.add_column("Best", justify="right", style="yellow", width=12)
    table.add_column("Best at solve", justify="right", style="dim", width=15)

    for stat in results["statistics"]:
        latest_str = f"{stat['latest']:.3f}s" if stat["latest"] is not None else "DNF"
        best_str = f"{stat['best']:.3f}s" if stat["best"] is not None else "DNF"
        position_str = (
            str(stat["best_position"]) if stat["best_position"] is not None else "N/A"
        )
        table.add_row(stat["kind"].display_name, latest_str, best_str, position_str)

    console.print(table)

    summary = results["summary"]
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"Total solves: {summary['total_solves']}")
    console.print(f"Average time: {summary['average_time']:.3f}s")
    console.print(f"Best single: {summary['best_single']:.3f}s")
    console.print(f"Worst single: {summary['worst_single']:.3f}s")


def main():
    args = parse_arguments()

    log_level = logging.INFO if args.verbose else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    try:
        logging.info(f"Reading data file: {args.data_file}")
        data, fieldnames = read_file(args.data_file)
        logging.info(f"Loaded {len(data)} records")

        if args.command == "show":
            ascending = getattr(args, "ascending", False)
            # FIXME: Extract sorting and logging into a common function.
            logging.info("Sorting by date...")
            sorted_data = sort_by_date(data, ascending=ascending)

            dates = [row["parsed_date"] for row in sorted_data]
            date_min = min(dates)
            date_max = max(dates)
            order = "ascending" if ascending else "descending"
            logging.info(f"Date range: {date_min} to {date_max}")
            logging.info(f"Sorted in {order} order")

            output_file = getattr(args, "output", None)
            display_data_table(sorted_data, fieldnames, output_file)

        elif args.command == "analyse":
            # FIXME: Extract sorting and logging into a common function.
            logging.info("Sorting by date...")
            sorted_data = sort_by_date(data, ascending=False)

            dates = [row["parsed_date"] for row in sorted_data]
            date_min = min(dates)
            date_max = max(dates)
            logging.info(f"Date range: {date_min} to {date_max}")
            logging.info("Sorted in descending order (newest first for analysis)")

            results = analyse_times(sorted_data)
            display_analysis_results(results)

        elif args.command == "plot":
            # FIXME: Extract sorting and logging into a common function.
            logging.info("Sorting by date...")
            sorted_data = sort_by_date(data, ascending=True)

            dates = [row["parsed_date"] for row in sorted_data]
            date_min = min(dates)
            date_max = max(dates)
            logging.info(f"Date range: {date_min} to {date_max}")
            logging.info("Sorted in ascending order (required for plotting)")

            try:
                output_file = getattr(args, "output", None)
                rolling_window = getattr(args, "rolling_average", 0) or 0
                sigmoid_enabled = getattr(args, "sigmoid", False)
                subx_seconds = getattr(args, "subX", 60.0)
                by_solve_number = getattr(args, "by_solve_number", False)

                fig = plot_times_over_date(
                    sorted_data,
                    output_file=output_file,
                    rolling_average=rolling_window,
                    sigmoid_curve=sigmoid_enabled,
                    subx_seconds=subx_seconds,
                    by_solve_number=by_solve_number,
                )

                if not output_file:
                    import matplotlib.pyplot as plt

                    plt.show()
                else:
                    import matplotlib.pyplot as plt

                    plt.close(fig)

            except PlotterError as e:
                logging.error(f"Plotting error: {e}")
                sys.exit(1)

    except ImporterError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
