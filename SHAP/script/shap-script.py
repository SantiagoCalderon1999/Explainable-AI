import psutil
from multiprocessing import Pool
import multiprocessing
import os
import cv2
import shap
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from multiprocessing import Pool
import logging
import pandas as pd

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('shap')

# Get the total RAM in bytes
total_ram = psutil.virtual_memory().total

# Get the available RAM in bytes
available_ram = psutil.virtual_memory().available

# Convert bytes to gigabytes (GB)
total_ram_gb = total_ram / (1024 ** 3)
available_ram_gb = available_ram / (1024 ** 3)

logger.info(f"Total RAM: {total_ram_gb:.2f} GB")
logger.info(f"Available RAM: {available_ram_gb:.2f} GB")

# Get the current CPU frequency in Hertz
cpu_frequency = psutil.cpu_freq()

# Convert Hertz to gigahertz (GHz) for a more human-readable format
cpu_frequency_ghz = cpu_frequency.current

logger.info(f"Current CPU Frequency: {cpu_frequency_ghz/1e3:.2f} GHz")

logger.info(f"Current core count: {multiprocessing.cpu_count()}")

mask = 'blur(64,64)'
classes = ["No tumour", "Tumour"]
results_dir = 'results'
min_expl_dir = 'minimal_explanation'
shap_array = []

def perform_parallel_shap_analysis(img_dict):
    pool = Pool(processes=120)
    pool.map(perform_individual_shap_analysis, img_dict.items())

def perform_individual_shap_analysis(img_item):
    name, img = img_item
    logger.info("Starting image " + str(name))

    path_to_model = "clf-resnet-weights.hdf5"
    model = tf.keras.models.load_model(path_to_model)

    # Get and plot shap results
    masker = shap.maskers.Image(mask, shape=img.shape)
    explainer=shap.Explainer(model, masker, output_names=classes, seed = 42)
    plt.clf()
    results = explainer(np.expand_dims(img, axis = 0), max_evals = 2000, batch_size = 50, outputs=shap.Explanation.argsort.flip[:1])
    shap.image_plot(results, show = False)
    plt.savefig(os.path.join(results_dir, name), dpi=500, bbox_inches='tight')
    plt.close()

    #Save results
    shap_array.append({'Name': name, 'Shap_values': results.data[0]})
    pd.DataFrame(shap_array).to_csv('shap_values.csv')
    # Extract minimal explanation
    average = (results.data[0, :, :, 0] + results.data[0, :, :, 1] + results.data[0, :, :, 2]) / 3
    levels = np.flip(np.unique(average))
    masks = np.empty([256,256,3])
    logger.info("Start extracting minimal explanation for "+ str(name))
    for count, level in enumerate(levels):
        logger.info("Level number: " + str(count))
        pixels = np.where(average == level)
        masks[pixels[0], pixels[1], :] = True
        min_expl = np.where(masks, img, 0)
        pre = model.predict(np.expand_dims(min_expl, axis=0), verbose=0)
        argmax = np.argmax(pre, axis = 1)
        logger.info("Pred: " + str(pre))
        if (argmax == 1 and pre[0][argmax] > 0.5):
            break
    logger.info("Finished extracting minimal explanation for "+ str(name))

    # Plot minimal explanation
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow((min_expl * 255).astype(np.uint8))
    ax.axis("off")
    plt.savefig(os.path.join(min_expl_dir, name), dpi=500, bbox_inches='tight')
    plt.close()

    logger.info("Finished image " + str(name))
    logger.info("----------------------------------------")

def create_directory_if_not_exists(dirname):
    if(not os.path.isdir(dirname)):
        os.mkdir(dirname)

if __name__ == '__main__':

    folder_path = 'positive/image/'

    image_files = [f for f in os.listdir(folder_path) if (f.lower().endswith('.tif') and f.lower().startswith('tcga'))]
    img_array = np.empty([len(image_files),256,256,3])
    img_name = []
    for counter, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is not None:
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            img_array[counter] = image
            img_name.append(filename)
        else:
            print(f"Failed to load image: {filename}")
    img_array = img_array.astype('float32')
    logger.info("Number of images: " + str(len(image_files)))
    img_dict = {name: img for name, img in zip(img_name, img_array)}
    shap.initjs()

    # Create required folders
    create_directory_if_not_exists(results_dir)
    create_directory_if_not_exists(min_expl_dir)

    start = timer()
    perform_parallel_shap_analysis(img_dict)
    logger.info("Time taken: ", timer()-start)
    logger.info("----------------------------------------")