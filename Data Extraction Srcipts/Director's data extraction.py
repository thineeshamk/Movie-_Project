import requests
import pandas as pd
from datetime import datetime
import time

TMDB_API_KEY = "add_yours" #I hide my API from here if any who download this script and want to run he/she have to create new API key and put that in here
OMDB_API_KEY = "add_yours" #I hide my API from here if any who download this script and want to run he/she have to create new API key and put that in here

BASE_URL = "https://api.themoviedb.org/3"
MAX_YEAR = 2009
TODAY = datetime.today().date()
EXCLUDED_GENRES = {"Documentary", "TV Movie"}

# Caches
director_id_cache = {}
director_rating_cache = {}

def get_tmdb_person_id(director_name):
    if director_name in director_id_cache:
        return director_id_cache[director_name]

    url = f"{BASE_URL}/search/person"
    params = {"api_key": TMDB_API_KEY, "query": director_name}
    response = requests.get(url, params=params)
    data = response.json()

    try:
        director_id = data["results"][0]["id"]
        director_id_cache[director_name] = director_id
        return director_id
    except (IndexError, KeyError):
        return None

def get_director_movies(director_id):
    url = f"{BASE_URL}/person/{director_id}/movie_credits"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    movies = []

    for crew_member in data.get("crew", []):
        if crew_member.get("job") != "Director":
            continue

        movie_id = crew_member["id"]
        release_date = crew_member.get("release_date", "")
        if not release_date:
            continue

        try:
            release_date_obj = datetime.strptime(release_date, "%Y-%m-%d").date()
        except:
            continue

        if release_date_obj > TODAY or release_date_obj.year > MAX_YEAR:
            continue

        detail_url = f"{BASE_URL}/movie/{movie_id}"
        detail_params = {"api_key": TMDB_API_KEY}
        detail_resp = requests.get(detail_url, params=detail_params)

        if detail_resp.status_code != 200:
            continue

        detail = detail_resp.json()
        runtime = detail.get("runtime", 0)

        if runtime and runtime < 40:
            continue

        countries = [c['iso_3166_1'] for c in detail.get("production_countries", [])]
        if "US" not in countries:
            continue

        genres = {g['name'] for g in detail.get("genres", [])}
        if genres & EXCLUDED_GENRES:
            continue

        imdb_id = detail.get("imdb_id")
        if imdb_id:
            movies.append(imdb_id)

    return list(set(movies))

def get_average_imdb_rating(director_name):
    if director_name in director_rating_cache:
        return director_rating_cache[director_name]

    director_id = get_tmdb_person_id(director_name)
    if not director_id:
        return None

    imdb_ids = get_director_movies(director_id)
    ratings = []

    for imdb_id in imdb_ids:
        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        try:
            response = requests.get(url)
            data = response.json()
            rating = data.get("imdbRating")

            if rating and rating != "N/A":
                ratings.append(float(rating))

            time.sleep(0.1)
        except:
            continue

    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
    director_rating_cache[director_name] = avg_rating
    return avg_rating

# Load Dataset
file_path = "directors_input.xlsx"        # This input is mine you can change this as you want
df = pd.read_excel(file_path)

avg_ratings = []

for idx, row in df.iterrows():
    director = row["first_director"]
    avg_rating = get_average_imdb_rating(director)
    avg_ratings.append(avg_rating)

df["Director Avg IMDb Rating"] = avg_ratings

output_file = "directors_with_average_ratings.xlsx"   # This output path is mine you can change this according to your output path
df.to_excel(output_file, index=False)
