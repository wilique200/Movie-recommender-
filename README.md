# Title-based Movie Recommender

Type a movie title and get similar movies with short reasons why they were recommended.

## Quick start

1. Place your downloaded ZIP file in the repo:
   - Path: `data/imdb_movies.zip`
   - The ZIP should contain CSV files (one or many genre CSVs).

2. Or run the app and upload the ZIP via the sidebar.

3. Install dependencies:
   pip install -r requirements.txt

4. Run locally:
   streamlit run app.py

## Notes
- The app uses TF‑IDF on `genre` + `description` (or `overview`) to compute similarity.
- If your CSVs use `movie_name` instead of `title`, the app will handle that automatically.
- Keep large datasets out of the repo; include the ZIP locally or upload at runtime.
