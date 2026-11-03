# ICC Cricket World Cup 2027 Predictor

An end-to-end cricket analytics project that uses historical ODI match data, machine learning, team ratings, and Monte Carlo simulation to estimate outcomes for a provisional 2027 ICC Cricket World Cup tournament format.

## Project Overview

This project explores how historical ODI performance and contextual factors can be used to estimate match outcomes and simulate a multi-stage international cricket tournament.

The project includes:

- Historical ODI data preparation and analysis
- Team-strength estimation using Elo ratings
- Machine learning match prediction
- Venue-aware prediction features
- Model evaluation and comparison
- Monte Carlo tournament simulation
- Interactive Streamlit dashboard

> **Important:** Tournament format, team composition, and qualifier assumptions are provisional. Predictions are probabilistic estimates, not official ICC forecasts or guaranteed outcomes.

## Dashboard

The interactive dashboard provides:

- Tournament overview
- Team qualification and progression probabilities
- Tournament simulation
- Match intelligence
- Match-wise win probabilities

## Project Structure

```text
ICC Cricket World Cup 2027 Predictor/
├── config/
├── data/
│   ├── processed/
│   └── raw/
├── docs/
├── models/
├── reports/
├── src/
│   └── world_cup_2027_dashboard.py
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-dashboard.txt
```

## Models and Evaluation

The project compares baseline machine-learning models, Elo-based ratings, and venue-aware models.

Evaluation includes:

- Accuracy
- ROC-AUC
- Log loss
- Brier score
- Rolling-origin evaluation
- Bootstrap uncertainty analysis

The current evaluation suggests that historical match data contains predictive signal, but ODI outcomes remain uncertain. Model performance should not be interpreted as highly accurate or guaranteed.

## Tournament Simulation

Monte Carlo simulation is used to estimate:

- Group-stage progression
- Super 7 qualification
- Semi-final qualification
- Final qualification
- Championship probability

The simulation runs repeatedly using model-based match probabilities and tournament-format assumptions.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/NishantKale-work/ICC_World_Cup_2027_Predictor.git
cd ICC_World_Cup_2027_Predictor
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run src/world_cup_2027_dashboard.py
```

## Data and Reproducibility

The repository contains project code, processed data, model artifacts, and simulation outputs where appropriate.

Raw source datasets may be excluded from version control because of file size or source-data restrictions. Refer to the project documentation for data assumptions and preparation details.

## Limitations

- Historical ODI data cannot fully capture future team composition, injuries, form, or tactical changes.
- Some tournament teams and format details are provisional assumptions.
- Venue effects are estimates based on historical data and may change.
- Simulation results depend on model probabilities and tournament assumptions.
- The simulator is an analytical project, not an official tournament prediction.

## Disclaimer

This project is for educational, analytical, and portfolio purposes. It is not affiliated with or endorsed by the ICC.

## Author

**Nishant Kale**

GitHub: [NishantKale-work](https://github.com/NishantKale-work)
