import os

import cv2
from PIL import  Image
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import flagSettings
from dataManagement import get_data_set
from linearEvaluation import linear_evaluation_model
from modelFunctions import build_simCLR_model, plot_linear_evaluation_accuracy
X=[]
Z = []
IMG_SIZE = 32
FLOWER_DAISY_DIR = '../../datasets/flowers/flowers/daisy'
FLOWER_SUNFLOWER_DIR = '../../datasets/flowers/flowers/sunflower'
FLOWER_TULIP_DIR = '../../datasets/flowers/flowers/tulip'
FLOWER_DANDI_DIR = '../../datasets/flowers/flowers/dandelion'
FLOWER_ROSE_DIR = '../../datasets/flowers/flowers/rose'


def assign_label(img, flower_type):
    return flower_type


def make_train_data(flower_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, flower_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(np.array(img))
        Z.append(str(label))


make_train_data('Daisy', FLOWER_DAISY_DIR)
make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
make_train_data('Tulip',FLOWER_TULIP_DIR)
make_train_data('Dandelion',FLOWER_DANDI_DIR)
make_train_data('Rose',FLOWER_ROSE_DIR)

le=LabelEncoder()
Y=le.fit_transform(Z)


x_train,x_test,y_train,y_test=train_test_split(np.array(X),np.array(Y),test_size=0.20,random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.10,random_state=1)


model = build_simCLR_model(encoder_network="resnet-18", projection_head_mode="nonlinear")
model.load_weights("../saved_models/resnet-18")

sk_learn_model, val_accuracy, test_acc = linear_evaluation_model(model, (x_train, y_train), (x_val, y_val), (x_test, y_test), "nonlinear")
plot_linear_evaluation_accuracy(val_accuracy, should_save_figure=True, file_name="linear_evaluation/")