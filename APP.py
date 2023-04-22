from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow import keras
from PIL import Image
import numpy as np
import io