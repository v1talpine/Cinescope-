# CineScope — Movie Data Analytics

A Python data-science project with an interactive Streamlit dashboard. It explores movie genres, ratings, runtime, budget, revenue and votes, and includes a baseline machine-learning model to estimate ratings.

## Important data note
The app initially runs on **synthetic sample data** so it works immediately without internet or a downloaded dataset. These records are fake and must not be used to make claims about real movies. For a final academic submission, upload a real, properly attributed dataset.

## Run locally (Windows)

1. Install Python 3.10 or newer from https://www.python.org/downloads/ and enable **Add Python to PATH** during installation.
2. Extract this project folder.
3. Open Command Prompt in the project folder.
4. Create a virtual environment (recommended):

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install packages:

   ```bat
   python -m pip install -r requirements.txt
   ```

6. Launch the dashboard:

   ```bat
   streamlit run app.py
   ```

7. A browser tab should open automatically. If not, copy the local URL printed in the terminal (usually `http://localhost:8501`).

## Dataset format

The app accepts a CSV. It works best with these columns:

- `title`
- `year`
- `genre`
- `runtime_min`
- `budget_m_usd`
- `revenue_m_usd`
- `rating` (0–10)
- `vote_count`

Some common alternative names are mapped automatically, including `movie_title`, `release_year`, `genres`, `runtime`, `imdb_rating`, `imdb_score`, `budget`, `revenue`, `votes`, and `num_votes`.

Columns not found are left missing; charts/model features that need them may be unavailable. For multi-genre rows, the app uses the first listed genre.

## Suggested real dataset
Use a public movie dataset from IMDb datasets (https://developer.imdb.com/non-commercial-datasets/) or TMDB (https://developer.themoviedb.org/). Check the specific dataset's licence/terms and attribution requirements. Data sources may not provide budget or revenue fields. Do not scrape sites against their terms.

## Project features
- Overview metrics and charts
- Interactive genre/year/rating filters
- Runtime-versus-rating and budget-versus-rating plots
- Download filtered/cleaned data
- Random Forest regression model with MAE, RMSE and R² metrics
- Correlation matrix and automatically generated genre observations
- Methodology and limitations section

## Academic honesty / interpretation
Explain which dataset you used, what cleaning was done, why the train/test split is needed, and what MAE/RMSE/R² mean. Do not describe synthetic data as real. A predictive relationship does not prove causation.
