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
from csv import writer, reader

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
min_expl_dir = 'minimal_explanation_only_shap_values'
shap_array = []
    
def perform_parallel_shap_analysis(results):
    pool = Pool(processes=1)
    pool.map(perform_individual_shap_analysis, results.items())

def perform_individual_shap_analysis(result):
    name, shap_values = result
    logger.info("Starting image " + str(name))

    path_to_model = "clf-resnet-weights.hdf5"
    model = tf.keras.models.load_model(path_to_model)
    
    # Extract minimal explanation
    average = (shap_values[:, :, :, 0] + shap_values[:, :, :, 1] + shap_values[:, :, :, 2]) / 3
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
    min_expl = min_expl_dir + '/'
    
    # Create required folders
    create_directory_if_not_exists(results_dir)
    create_directory_if_not_exists(min_expl_dir)

    image_files = [f for f in os.listdir(folder_path) if (f.lower().endswith('.tif') and f.lower().startswith('tcga'))]
    finished_image_files = [f for f in os.listdir(min_expl) if (f.lower().endswith('.tif') and f.lower().startswith('tcga'))]
    results_image_files = [f for f in os.listdir(results_dir) if (f.lower().endswith('.tif') and f.lower().startswith('tcga'))]
    
    image_files = [x for x in image_files if (x not in finished_image_files) or (x not in results_image_files)]
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
    # Get shap shap_values_without_first_row
    with open('shap_values_without_first_row.csv', newline='') as csvfile:
        results = list(reader(csvfile))
    import re
    shap_array = []
    for result in results:    
        numeric_values = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", result[0])]
        # Create a NumPy array from the numeric values
        shap_array.append(np.array(numeric_values).reshape(256, 256, 3))
        
    shap_dict = {name: shap_val for name, shap_val in zip(numeric_values, shap_array)}
    start = timer()
    perform_parallel_shap_analysis(results)
    logger.info("Time taken: " + str(timer()-start))
    logger.info("----------------------------------------")