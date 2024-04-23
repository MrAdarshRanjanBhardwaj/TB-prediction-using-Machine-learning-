import numpy  as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
import os

Categories = ["Normal","Tuberculosis"]
flat_data_arr = []
target_arr = []

datadir = "TB_Chest_Radiography_Database"


for i in Categories:
    print(f"loading categories : {i}")
    path = os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img))
        img_resized = resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f"loaded category {i} ")


flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(flat_data)
df["Target"] = target


X = df.drop('Target',axis=1)
Y = df[['Target']]


from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y)

X_train = X_train[:700]
y_train = y_train[:700]
X_test = X_test[:700]
y_test = y_test[:700]


svc = svm.SVC(probability=True)
svc.fit(X_train, y_train.values.ravel())
print(f" SUPPORT VECTOR MACHINE ACCURACY {svc.score(X_test,y_test)}")

from PIL import Image

# Read the image
image_path = "TB_Chest_Radiography_Database/Normal/Normal-1.png"
image = Image.open(image_path)

# Preprocess the image
resized_image = image.resize((150, 150))  # Resize the image to match the input size used during training
normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values to the range [0, 1]

# Convert the image to a format suitable for your model
flat_image = normalized_image.flatten()  # Flatten the image array

# Make predictions using your trained model
prediction = svc.predict([flat_image])  # Assuming 'svc' is your trained SVM classifier

# Interpret the prediction
if prediction[0] == 0:
    print("The image does not depict tuberculosis (TB)")
else:
    print("The image depicts tuberculosis (TB)")

knn = KNeighborsClassifier()
knn.fit(X_train, y_train.values.ravel())
print(f" KNN  ACCURACY {knn.score(X_test,y_test)}")



