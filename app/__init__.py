from flask import Flask
import os

app = Flask(__name__)
config = dict(os.environ)

from app import routes