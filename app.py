from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

model = load_model('solar_farm_model_best.h5')

# model.make_predict_function()

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]
def preprocess_image(image_path, target_size=(150, 150)):
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Rescale the image (if the model was trained with rescaled images)
    img_array = img_array / 255.0
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Make a prediction
    prediction = model.predict(preprocessed_image)[0][0]
    # Determine the label based on the prediction
    label = 'Solar Farm' if prediction > 0.5 else 'No Solar Farm'
    return prediction, label


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        if img:
            # Create a secure filename and save the image in the static folder
            img_path = os.path.join('static', img.filename)
            print(img_path)
            img.save(img_path)

            prediction, label = predict_image(img_path)

            return render_template("index.html", label=label, prediction=prediction, img_path=img_path)
        else:
            return "No image found", 400
# return render_template("index.html", prediction = prediction, img_path = img_path)


if __name__ =='__main__':
	app.run(debug = True)