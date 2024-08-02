## Building a MovieLens Recommender System Using Neural Collaborative Filtering

<img src="https://github.com/user-attachments/assets/95f040c9-048e-4da7-b019-cddbbb44802e" width="40%" align='right'/>
This project implements a Neural Collaborative Filtering (NeuCF) model in PyTorch for movie recommendations using the MovieLens dataset.

Key features include:

- **Model Architecture:** Combines user and item embeddings with fully connected hidden layers to predict user ratings for movies.
- **Training & Evaluation:** Includes mechanisms for early stopping to prevent overfitting and ensure optimal model performance.
- **Recommendations:** Provides a function to generate top N movie recommendations for a given user, outputting IMDb links for easy access to movie details.
- **Preprocessing:** Ensures no user rates the same movie more than once by handling duplicate entries effectively.
- **Bayesian Average:** Utilizes a Bayesian average approach to handle rating data, ensuring more robust and stable predictions by accounting for variability in the number of ratings per movie.

The model predicts the probability that a user will like a particular movie and uses these probabilities to recommend the most suitable movies to each user.

We'll be using these packages to do our analysis:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [pytorch](https://pytorch.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://www.docker.com/)

In this repo, you'll find one notebook:

1. [notebook](neucf.ipynb): this contains all the codes for model training as well as recommendation
2. [RESTful API using FastAPI](myapp): This folder contains code for FastAPI application used to recommend movies.

## DEMO
<video width="600" controls>
  <source src="https://github.com/user-attachments/assets/da4e711f-1f64-4018-996b-9b8165ff77f4.webm" type="video/webm">
</video>
