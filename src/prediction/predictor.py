import os

import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

from src.constants.eurosat import INDEX_TO_CLASS
from src.data.transforms import validation_transform
from src.helpers.config import get_settings

settings = get_settings()


class Predictor:

    def __init__(self):

        model_path = os.path.join(
            settings.MODEL_SAVE_DIR,
            settings.MODEL_FILE_NAME,
        )

        if not os.path.exists(model_path):
            model_path = hf_hub_download(
                repo_id="Mowael1/efficientnetb3-eurosat",
                filename="efficientnetb3.keras",
            )

        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, image: np.ndarray):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = validation_transform(image)

        image = tf.expand_dims(
            image,
            axis=0,
        )

        return image

    def predict(self, image: np.ndarray):

        image = self.preprocess_image(image)

        predictions = self.model.predict(
            image,
            verbose=0,
        )

        predicted_index = np.argmax(predictions[0])

        confidence = float(
            predictions[0][predicted_index]
        )

        predicted_class = INDEX_TO_CLASS[predicted_index]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
        }