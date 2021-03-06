# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback
import logging
import numpy as np
from models.recommender_net import RecommenderNet 
import flask

prefix = '/opt/program/'
model_path = os.path.join(prefix, 'models')

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = RecommenderNet.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='Ok\n', status=status, mimetype='application/json')

@app.route('/train', methods=['POST'])
def train():
    # Training should happen here

    return flask.jsonify({
        "error": False,
        "message": "It is all good man"
    })

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    try :
        # Get the target products
        products = flask.request.json["products"]

        # TODO: we need to get all the candidates for the target products        
        recommended_products = RecommenderNet.predict(products[0])

        print("Candidates for product 1 :", recommended_products)

        return flask.jsonify(recommended_products)

    except Exception as e:
        logging.error(traceback.format_exc())
        return flask.jsonify({
            "error": "true",
            "message": "Cannot recommend products."
        })