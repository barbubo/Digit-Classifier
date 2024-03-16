import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
dataset_path = 'dataset.csv'
image_size = (28, 28)  # add 3 if RGB image


def load():
    data = pd.read_csv(dataset_path)
    pixels = data['Pixels'].tolist()
    labels = data['Labels'].tolist()
    width, height = 28, 28
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        a = face
        face = np.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    pixels = faces
    faces = np.expand_dims(faces, -1)
    return pixels, labels

pixels, labels = load()
pixels = pixels.reshape(500, -1)

train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10
x_train, x_aux, y_train, y_aux = train_test_split(pixels, labels, test_size=(1 - train_ratio))
x_val, x_test, y_val, y_test = train_test_split(pixels, labels, test_size=(1 - val_ratio))
model = RandomForestClassifier(n_estimators=300, random_state=15)
model.fit(x_train, y_train)
y_pred = model.predict(x_val)
print("Validation accuracy")
print(model.score(x_val, y_val))
print(accuracy_score(y_pred, y_val))
print("\nTest accuracy")
y_pred2 = model.predict(x_test)
print(model.score(x_test, y_test))
print(accuracy_score(y_pred2, y_test))
y_predicted = model.predict(x_test[5].reshape(1, -1))
pixel = x_test[5]
label = y_predicted
pixel = pixel.reshape((28,28))
plt.title(f'Predicted digit is {label}'.format(label=label))
plt.imshow(pixel, cmap='gray')
plt.show()