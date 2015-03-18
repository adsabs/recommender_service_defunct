import os
from flask import Blueprint
from flask import Flask, g
from views import blueprint, Recommender
from flask.ext.restful import Api
from flask.ext.discoverer import Discoverer
from client import Client
from utils.database import db

def create_app(blueprint_only=False):
  app = Flask(__name__, static_folder=None)

  app.url_map.strict_slashes = False
  app.config.from_pyfile('config.py')
  try:
    app.config.from_pyfile('local_config.py')
  except IOError:
    pass
  app.client = Client(app.config['CLIENT'])

  api = Api(blueprint)
  api.add_resource(Recommender, '/<string:bibcode>')

  if blueprint_only:
    return blueprint

  app.register_blueprint(blueprint)
  db.init_app(app)

  discoverer = Discoverer(app)

  return app

if __name__ == "__main__":
  app = create_app()
  app.run()
