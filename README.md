# CubeTimes

A Python tool for analyzing and plotting cube solve times.

## Features

- Read cube solve times from JSON (csTimer) files 
- **Analyze solve times with cubing statistics (average, ao5, ao12, ao100)**
- **Plot solve times over date or solve number**
  - **DNF (Did Not Finish) handling and visualization**
  - **Vertical lines marking best average achievements**
  - **Generate smooth rolling average curve for trend analysis**
  - **Add sigmoid curve fitting to visualize improvement trends**
  - **Configurable horizontal reference lines**
  - **Dark mode plotting with professional styling**
- Verbose logging option
- Save output to file or display in terminal or window

## Installation

```bash
# Install the package
uv pip install -e .

# Or run directly with uv
uv run cubetimes
```

## Usage

The tool uses subcommands for different operations:

```bash
# Show help
cubetimes -h

# Default: show table
cubetimes

# Show table explicitly
cubetimes show

# Show table with specific file and options
cubetimes my-times.json show --ascending --output sorted_times.txt

# Analyze times and show statistics
cubetimes analyse

# Generate plots
cubetimes plot
cubetimes plot --by-solve-number
cubetimes plot --subX 45
cubetimes plot --rolling-average 50
cubetimes plot --sigmoid
cubetimes plot --output my_progress.png

# Global options work with all subcommands
cubetimes --verbose analyse
```

## Analysis Features

The `analyse` command provides comprehensive cubing statistics:

- **Latest averages**: Most recent ao5, ao12, and ao100
- **Best averages**: Personal best averages with solve numbers
- **Summary statistics**: Total solves, overall average, best and worst single times
- **DNF handling**: Proper treatment of DNF (Did Not Finish) solves

Standard cubing averaging rules are applied (drop best and worst times for averages).

## Plotting Features

The `plot` command provides comprehensive visual analysis of your cubing progress:

### Core Features:
- **Timeline plots**: View solve times over date with scatter plot visualization
- **DNF visualization**: DNF solves shown as red X markers on the x-axis
- **Statistics overlay**: Key stats displayed on plots (average, best, worst, solve count)
- **Best average markers**: Vertical lines marking your best ao5, ao12, and ao100 achievements
- **Dark mode styling**: Professional dark theme with white plot area
- **File output**: Save high-quality plots as PNG files (300 DPI)
- **Interactive display**: View plots interactively when no output file specified

### Advanced Features:
- **Rolling averages**: Plot smooth rolling averages to see improvement trends
  - Configurable window size (e.g., `--rolling-average 25`)
  - Heavily smoothed curves for clean trend visualization
  - Adaptive marker sizing when rolling average is shown
- **Sigmoid curve fitting**: Add mathematical trend curves to visualize improvement patterns
  - Simple parameter estimation for best fit
  - Smooth curve interpolation
  - Option: `--sigmoid`
- **Configurable reference lines**: Add horizontal reference lines at any time
  - Default: 60 seconds (`--subX 60`)
  - Custom values: `--subX 45` for sub-45 reference
  - Useful for tracking specific time goals

Plotting examples:
```bash
# Basic timeline plot
cubetimes plot

# Basic plot over solve numbers
cubetimes plot --by-solve-number

# Rolling average with 25-solve window
cubetimes plot --rolling-average 25

# Sigmoid trend analysis
cubetimes plot --sigmoid

# Sub-30 goal tracking
cubetimes plot --subX 30

# Complete analysis
cubetimes plot --rolling-average 20 --sigmoid --subX 45 --output complete_analysis.png
```

## Data Format

The JSON file should follow the csTimer JSON format.

## Development

```bash
# Install development dependencies
uv sync

# Run the tool during development
uv run python -m cubetimes.main

# Run tests (when added)
uv run pytest
```
