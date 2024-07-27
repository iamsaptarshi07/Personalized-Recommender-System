import torch
import pandas as pd
from model_def import NeuCF
from pathlib import Path

from utils.data_processing import movie_id_to_link_vectorized

def load_model(model_path: Path, df: pd.DataFrame, device):

    config = {
        'num_users': df['userId'].nunique(),
        'num_items': df['movieId'].nunique(),
        'embedding_dim': 32,
        'hidden_layers': [64, 32, 16, 8],
        'learning_rate': 0.001,
        'num_epochs': 50,
        'batch_size': 256,
        'dropout_rate': 0.2
    }


    model = torch.compile(NeuCF(config), backend="aot_eager").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def recommend_movies(model, user_id, all_movie_ids, df, device, top_n=10):
    assert 0 <= user_id < model.num_user, f"User ID {user_id} is out of bounds. Use between 0 and {model.num_user}."

    model.eval()
    user_tensor = torch.tensor([user_id] * len(all_movie_ids), dtype=torch.int32).to(device)
    item_tensor = torch.tensor(all_movie_ids, dtype=torch.int32).to(device)

    with torch.inference_mode():
        probabilities = model(user_tensor, item_tensor).cpu().numpy().flatten()

    movie_probabilities = list(zip(all_movie_ids, probabilities))
    recommended_movies = sorted(movie_probabilities, key=lambda x: x[1], reverse=True)

    top_movie_ids = [movie_id for movie_id, _ in recommended_movies[:top_n]]
    movie_id_to_title = df.set_index('movieId')['title'].to_dict()
    movie_id_to_link = movie_id_to_link_vectorized(df)

    recommended_movie_titles = dict([(movie_id_to_title[movie_id], movie_id_to_link[movie_id])for movie_id in top_movie_ids])

    return recommended_movie_titles
