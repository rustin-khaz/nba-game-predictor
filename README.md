# 🏀 NBA Game Outcome Predictor

Predicting NBA game winners using machine learning and recent team performance data.

Built with Python, scikit-learn, XGBoost, and the NBA Stats API across **3 seasons** (2022-23, 2023-24, 2024-25) — **7,380 games** total.

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 65.4% | 0.709 |
| **Random Forest** | **65.8%** | **0.702** |
| XGBoost | 62.6% | 0.668 |

> For context, Vegas oddsmakers hover around 67-68% — making 65%+ accuracy on a clean ML pipeline genuinely competitive.

---

## Project Structure

```
nba-game-predictor/
├── notebooks/
│   ├── 01_data_collection.ipynb      # Pull & explore 3 seasons of NBA data
│   ├── 02_feature_engineering.ipynb  # Build rolling features, rest days, win streaks
│   ├── 03_modeling.ipynb             # Train & evaluate 3 ML models
│   └── 04_prediction_summary.ipynb   # Interactive game predictor
├── data/
│   └── cleaned/
│       ├── team_logs_all_seasons.csv  # 7,380 team-game rows
│       └── matchups_features.csv     # 3,638 matchup rows, 24 features
└── visuals/                           # All charts and plots
```

---

## Methodology

### 1. Data Collection
Game logs pulled from the [NBA Stats API](https://www.nba.com/stats) for the 2022-23, 2023-24, and 2024-25 regular seasons. Each row represents one team's performance in one game.

### 2. Feature Engineering
The core challenge: you can't use in-game stats to predict a game — that's data leakage. Instead, we build **rolling pre-game features** from each team's recent history:

- **Rolling averages** (last 5 and 10 games) for points, rebounds, assists, steals, blocks, turnovers, FG%, 3P%, FT%, and plus/minus
- **Opponent points allowed** — rolling average of points conceded (defensive proxy)
- **Rest days** — days since last game, capped at 7 (back-to-backs measurably hurt teams)
- **Win streak** — current streak going into the game (positive = wins, negative = losses)
- **Differential features** — every stat computed as home minus away, capturing relative team strength directly

Rolling windows reset at the start of each new season so end-of-season form doesn't bleed into the next year's opening games.

### 3. Modeling
One row per matchup with **24 differential features** as inputs. Target variable: `home_win` (1 = home team wins).

**Train/test split is time-based** — first 80% of games by date for training, last 20% for testing. This mirrors real-world prediction where you only use past data to predict future games.

Three classifiers compared:
- **Logistic Regression** — interpretable linear baseline with StandardScaler
- **Random Forest** — 300 trees, captures non-linear interactions
- **XGBoost** — gradient boosted trees, sequential error correction

### 4. Key Findings
- **Plus/minus differential** is the single strongest predictor of game outcomes
- **Home court advantage** is real — home teams win 54.4% of games in the dataset
- **Rest day advantage** measurably shifts win probability — well-rested teams outperform back-to-back teams
- **Win streak momentum** adds predictive signal on top of raw performance stats
- Models are better at catching home wins (81% recall) than road upsets (47% recall) — upsets are inherently hard to predict

---

## Visuals

### Team Win Percentage — 2024-25 Season
![Win Percentage](visuals/win_percentage.png)

### ROC Curves — All 3 Models
![ROC Curves](visuals/roc_curves.png)

### Feature Importance (Random Forest)
![Feature Importance](visuals/feature_importance.png)

### Confusion Matrices
![Confusion Matrices](visuals/confusion_matrices.png)

### Home Win Rate by Rest & Streak Advantage
![Home Advantage Analysis](visuals/home_advantage_analysis.png)

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/rustin-khaz/nba-game-predictor.git
cd nba-game-predictor

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost nba_api jupyter

# Launch Jupyter
jupyter notebook
```

Open the notebooks in order: `01` → `02` → `03` → `04`

### Predict a Game
In notebook 04, call `predict_game()` with any two team abbreviations:

```python
predict_game('OKC', 'CLE')
```

```
==================================================
   OKC (HOME)  vs  CLE (AWAY)
   OKC: avg 125.2 pts | streak +4
   CLE: avg 119.7 pts | streak -1
──────────────────────────────────────────────────
   Logistic Regression    OKC: 72.4%  CLE: 27.6%  → OKC
   Random Forest          OKC: 74.7%  CLE: 25.3%  → OKC
   XGBoost                OKC: 79.4%  CLE: 20.6%  → OKC
──────────────────────────────────────────────────
   CONSENSUS: OKC wins  (75.5% confidence)
==================================================
```

---

## Tech Stack

| Tool | Use |
|---|---|
| Python 3.12 | Core language |
| pandas / numpy | Data manipulation |
| nba_api | NBA Stats data source |
| scikit-learn | Logistic Regression, Random Forest, preprocessing |
| XGBoost | Gradient boosted classifier |
| matplotlib / seaborn | Visualizations |
| Jupyter | Interactive notebooks |

---

## Future Work
- Add Elo ratings as a running team strength metric
- Account for player availability (injuries, rest)
- Build a Streamlit web app for live predictions
- Compare model probabilities against Vegas betting lines

---

*Built by [Rustin Khazravi](https://github.com/rustin-khaz)*
