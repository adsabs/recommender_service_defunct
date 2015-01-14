import os
from flask import Blueprint
from flask import Flask, g
from views import blueprint, Resources, Recommender
from flask.ext.restful import Api
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
  api.add_resource(Resources, '/resources')
  api.add_resource(Recommender, '/<string:bibcode>')

  if blueprint_only:
    return blueprint

  app.register_blueprint(blueprint)
  db.init_app(app)
  return app

if __name__ == "__main__":
  app.run()
