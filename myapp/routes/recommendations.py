from fastapi import APIRouter, HTTPException
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
import torch
import os

from utils.model import recommend_movies, load_model

router = APIRouter()

DF_PATH = Path(__file__).resolve().parent.parent / 'resources' / 'useful_df'
MODEL_PATH = Path(__file__).resolve().parent.parent / 'resources' / 'model.pt'

device = ("cuda" if torch.cuda.is_available() else "cpu")
useful_df = pd.read_csv(DF_PATH)
trained_model = load_model(model_path=MODEL_PATH, df=useful_df, device=device)

class RecommendationRequest(BaseModel):
    user_id: int = 0
    top_n: int

@router.post("/")
def get_recommendation(request: RecommendationRequest):
    user_id = request.user_id
    all_movie_ids = useful_df['movieId'].values

    if not (0 <= user_id < useful_df['userId'].nunique()):
        raise HTTPException(status_code=400, detail="User ID is out of bounds")

    try:
        recommendations = recommend_movies(
            model=trained_model,
            user_id=user_id,
            all_movie_ids=all_movie_ids,
            df=useful_df,
            device=device,
            top_n=request.top_n
        )
        if not recommendations:
            raise HTTPException(status_code=404,detail="No Recommendation Found")

        return {f"recommendations for user: {user_id}": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail="str{e}")
