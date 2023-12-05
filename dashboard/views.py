from django.shortcuts import render
import os.path
import boto3
import sys
import os
import pandas as pd
import csv
import io
from tensorflow.keras.models import load_model
import boto3
import sys
import os
import pandas as pd
import csv
import io
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from typing import Dict, Text
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import requests
# Create your views here.

def home(request):
    return render(request,'dashboard/home.html')


def predict(request):
    predictions = ''
   
    if request.method == 'GET':
        user_id = request.GET.get('userid')
        #current os path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # join the path with 'df.csv' to get path
        path = os.path.join(BASE_DIR, 'df.csv')
        df = pd.read_csv(path)
        path = os.path.join(BASE_DIR,'ratings_small.csv')
        ratings_df = pd.read_csv(path)
        ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
        ratings_df.drop('timestamp', axis=1, inplace=True)

        ratings_df = ratings_df.merge(df[['id', 'original_title', 'genres', 'overview']], left_on='movieId',right_on='id', how='left')
        ratings_df = ratings_df[~ratings_df['id'].isna()]
        ratings_df.drop('id', axis=1, inplace=True)
        ratings_df.reset_index(drop=True, inplace=True)

        movies_df = df[['id', 'original_title']]
        movies_df.rename(columns={'id':'movieId'}, inplace=True)

        ratings_df['userId'] = ratings_df['userId'].astype(str)

        ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df[['userId', 'original_title', 'rating']]))
        movies = tf.data.Dataset.from_tensor_slices(dict(movies_df[['original_title']]))

        ratings = ratings.map(lambda x: {
            "original_title": x["original_title"],
            "userId": x["userId"],
            "rating": float(x["rating"])
        })

        movies = movies.map(lambda x: x["original_title"])
        movie_titles = movies.batch(1_000)
        user_ids = ratings.batch(1_000).map(lambda x: x["userId"])

        unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
        unique_user_ids = np.unique(np.concatenate(list(user_ids)))

        class MovieModel(tfrs.models.Model):

            def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
                # We take the loss weights in the constructor: this allows us to instantiate
                # several model objects with different loss weights.

                super().__init__()

                embedding_dimension = 64

                # User and movie models.
                self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_movie_titles, mask_token=None),
                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
                ])
                self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                ])

                # A small model to take in user and movie embeddings and predict ratings.
                # We can make this as complicated as we want as long as we output a scalar
                # as our prediction.
                self.rating_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(256, activation="relu"),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(1),
                ])

                # The tasks.
                self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()],
                )
                self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
                    metrics=tfrs.metrics.FactorizedTopK(
                        candidates=movies.batch(128).map(self.movie_model)
                    )
                )

                # The loss weights.
                self.rating_weight = rating_weight
                self.retrieval_weight = retrieval_weight

            def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
                # We pick out the user features and pass them into the user model.
                user_embeddings = self.user_model(features["userId"])
                # And pick out the movie features and pass them into the movie model.
                movie_embeddings = self.movie_model(features["original_title"])
                
                return (
                    user_embeddings,
                    movie_embeddings,
                    # We apply the multi-layered rating model to a concatentation of
                    # user and movie embeddings.
                    self.rating_model(
                        tf.concat([user_embeddings, movie_embeddings], axis=1)
                    ),
                )

            def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

                ratings = features.pop("rating")

                user_embeddings, movie_embeddings, rating_predictions = self(features)

                # We compute the loss for each task.
                rating_loss = self.rating_task(
                    labels=ratings,
                    predictions=rating_predictions,
                )
                retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

                # And combine them using the loss weights.
                return (self.rating_weight * rating_loss
                        + self.retrieval_weight * retrieval_loss)

    # Create an instance of the model
    model = MovieModel(rating_weight=1.0, retrieval_weight=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    # Load only the weights
    #join movie_model_weights with current os path
    path = os.path.join(BASE_DIR, 'movie_model_weights')
    model.load_weights(path)
    

       
    def predict_movie(user, top_n=3):
    # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        # recommends movies out of the entire movies dataset.
        index.index_from_dataset(
        tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
        )

        # Get recommendations.
        recommendations_dict = {}
        _, titles = index(tf.constant([str(user)]))
        
        print('Top {} recommendations for user {}:\n'.format(top_n, user))
        for i, title in enumerate(titles[0, :top_n].numpy()):
            recommendation_index = i + 1
            recommendation_title = title.decode("utf-8")
            recommendations_dict[recommendation_index] = recommendation_title
        print(recommendations_dict)
        return recommendations_dict
     
    recommendations_dict = predict_movie(user_id)
    return render(request,'dashboard/predictions.html', {'user_id': user_id, 'recommendations_dict': recommendations_dict})


def random(request):
    function_endpoint = 'https://gtwlna3bg5.execute-api.us-west-2.amazonaws.com/default/SimpleFunction'
    response = requests.get(function_endpoint)
    if response.status_code == 200:
        print("Response from Lambda:", response.text)
        response2 = response.text
    else:
        print("Invocation failed:", response.text)
    context = {
        'response_text': response2  # Passing response_text to the template
    }
    print(context)
    return render(request,'dashboard/random.html', context)
   

    
