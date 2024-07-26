import pandas as pd

def movie_id_to_link_vectorized(df: pd.DataFrame) -> dict:
    df['imdbId'] = df['imdbId'].astype(str)
    links = "www.imdb.com/title/tt0" + df['imdbId']
    imdb_links = dict(zip(df['movieId'], links))
    print(f"imdb links: {imdb_links}")
    return imdb_links
