number_of_images = 1

import os
import cv2
import shap
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from numba import jit, cuda 
from timeit import default_timer as timer

folder_path = 'positive/image/'

image_files = [f for f in os.listdir(folder_path) if (f.lower().endswith('.tif') and f.lower().startswith('tcga'))]
img_array = np.empty([len(image_files),256,256,3])
img_name = []
for counter, filename in enumerate(image_files):
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path)

    if image is not None:
        image = image / 255.0 # we need to normalise (but see what happens if you don't)
        image = np.expand_dims(image, axis=0)
        img_array[counter] = image
        img_name.append(filename)
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

path_to_model = "clf-resnet-weights.hdf5"
model = tf.keras.models.load_model(path_to_model)
img_0 = []
img_0.append(img_array[0])

dirname = 'results'
dirname2 = 'minimum_explanation/countours'
dirname3 = 'minimum_explanation/no-countours'
if(not os.path.isdir(dirname)):
    os.mkdir(dirname)
if(not os.path.isdir(dirname2)):
    os.mkdir(dirname2)
if(not os.path.isdir(dirname3)):
    os.mkdir(dirname3)

def perform_shap_analysis(img_name, mask, classes, model, img_array, dirname):
    for count, img in enumerate(img_array):
        print("Starting image" + str(count) + ": " + str(img_name[count]))
        if (count == number_of_images):
            break
        masker = shap.maskers.Image(mask, shape=img.shape)
        explainer=shap.Explainer(model, masker, output_names=classes, seed = 42)
        plt.clf()
        results = explainer(np.expand_dims(img, axis = 0), max_evals = 10, batch_size = 50, outputs=shap.Explanation.argsort.flip[:1])
        shap.image_plot(results, show = False)
        plt.savefig(os.path.join(dirname, img_name[count]), dpi=500, bbox_inches='tight')
        average = (results.data[0, :, :, 0] + results.data[0, :, :, 1] + results.data[0, :, :, 2]) / 3
        levels = np.flip(np.unique(average))
        masks = np.empty([256,256,3])
        counter = 0
        for level in levels:
            counter = counter + 1
            pixels = np.where(average == level)
            masks[pixels[0], pixels[1], :] = True
            xd = np.where(masks, img_array[0], 0)
            tf.config.run_functions_eagerly(False)
            pre = model.predict(np.expand_dims(xd, axis=0), verbose=0)
            argmax = np.argmax(pre, axis = 1)
            if (argmax == 1 and pre[0][argmax] > 0.5):
                break
        xd = cv2.cvtColor(xd, cv2.COLOR_BGR2GRAY)
        xd  = (xd*255).astype(np.uint8)

        _, xd = cv2.threshold(xd, 0, 255, cv2.THRESH_BINARY)
        countours,_ = cv2.findContours(xd,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        plt.clf()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(xd.astype(np.uint8))
        ax.axis("off")
        plt.savefig(os.path.join(dirname3, img_name[count]), dpi=500, bbox_inches='tight')
        plt.close()  # Close the figure to release resources
        # draw contours
        cv2.drawContours(xd, countours, -1, (0,255,0), 0)
        plt.clf()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(xd.astype(np.uint8))
        ax.axis("off")
        plt.savefig(os.path.join(dirname2, img_name[count]), dpi=500, bbox_inches='tight')
        plt.close()  # Close the figure to release resources
        print("Finished image" + str(count) + ": " + str(img_name[count]))
        
start = timer() 
perform_shap_analysis(np.array(img_name), mask, np.array(classes), model, np.array(img_array), dirname)
print("with jit:", timer()-start)