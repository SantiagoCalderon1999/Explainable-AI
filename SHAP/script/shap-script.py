import os
import cv2
import shap
import numpy as np
import tensorflow as tf

folder_path = 'c:/repos/Explainable-AI/SHAP/script/positive/image/'

image_files = [f for f in os.listdir(folder_path) if (f.lower().endswith('.tif') and f.lower().startswith('tcga'))]
img_array = np.empty([len(image_files),256,256,3]);
counter = 0
for filename in image_files:
    countar = counter + 1
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path)

    if image is not None:
        image = image / 255.0 # we need to normalise (but see what happens if you don't)
        image = np.expand_dims(image, axis=0)
        img_array[counter] = image

    else:
        print(f"Failed to load image: {filename}")
img_array = img_array.astype('float32')    
print("Number of images: " + str(len(image_files)))

shap.initjs()

explainers = []
mask = 'blur(64,64)'

shap_values = {}

shap_values_array= []

classes = ["No tumour", "Tumour"]

path_to_model = "c:/repos/Explainable-AI/SHAP/clf-resnet-weights.hdf5"
model = tf.keras.models.load_model(path_to_model)

print(str(img_array[0].shape))
img_xd = []
img_xd.append(img_array[0])
for img in img_xd:
    masker = shap.maskers.Image('blur(64,64)', shape=img.shape)
    explainer=shap.Explainer(model, masker, output_names=classes, seed = 42)
    shap_values_array.append(explainer(np.expand_dims(img, axis = 0), max_evals = 2000, batch_size = 50, outputs=shap.Explanation.argsort.flip[:1]))      