# Stream Macroinvertebrate Image Analysis System

A Python application for exploratory analysis and classification of stream
macroinvertebrate images. Software Technology 1 (4483), Assignment 3,
Group 4.

Stages submitted: Stage 1 (Exploratory Data Analysis) + Stage 3 (Deployment)

## Features

- Recursive indexing of the image dataset into a structured pandas DataFrame
- Exploratory data analysis: class distribution, image size histograms,
  sample grid, summary statistics
- Baseline scikit-learn RandomForest classifier (used by the deployed app
  to make predictions)
- Tkinter desktop GUI for predicting the species of a new image
- Menu-driven console application as an alternative interface
- Command-line pipeline runner for end-to-end reproduction

## Requirements

- Python 3.12
- Packages listed in `requirements.txt`

## Installation

1. Clone the repository and enter the project folder.
2. Create and activate a virtual environment:
   - Windows: `py -3.12 -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3.12 -m venv .venv` then `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Download the Kaggle Stream Macroinvertebrates dataset and unzip it into
   `data/raw/` so each species has its own subfolder.

## How to run

- `python -m src.main` - runs the full pipeline (index, EDA, train, save model)
- `python -m src.app` - launches the Tkinter GUI for image prediction
- `python -m src.console_app` - launches the menu-driven console application

The GUI and console prediction require a trained model. Run
`python -m src.main` once before predicting, or use the train option in the
console menu.

## Project structure

src/        application code (config, services, entry points)
outputs/    generated EDA charts, model, and evaluation reports
docs/       test screenshots
data/raw/   dataset location (not included in the repository)

## Documentation

- `IMPLEMENTATION_SUMMARY.md` - design overview, findings, testing summary
- `MANUAL_TESTING.md` - manual test scenarios and results

## Authors

Group 4 - u3318477, u3334531, u3323255
