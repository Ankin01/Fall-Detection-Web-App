import streamlit as st
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from IPython import get_ipython
from numpy import asarray

import codee


from PIL import Image  


@st.cache
def load_image(image_file):
	imag = Image.open(image_file,'r')
	return imag

def main():
	st.title("Fall / No-Fall Detection in an Image")

	st.write("Please upload an image to detect a fall instance / falling person ")

	image_file = st.file_uploader("Upload images", type=["png","jpeg","jpg"])

	if image_file is not None:
		#st.write(type(image_file))
		#st.write(dir(image_file))

		#file_details ={"filename":image_file.name,"filetype":image_file.type,"filesize":image_file.size,"directory":image_file.__dir__}
		#st.write(file_details)

		st.header("Your uploaded image")

		st.image(load_image(image_file),width=400,height=400)


		ROOT_DIR = os.path.abspath("../")

		#st.write(ROOT_DIR)


		sys.path.append(ROOT_DIR)  
		


		MODEL_DIR = os.path.join(ROOT_DIR, "ProjectTwo")

		WEIGHTS_PATH = os.path.join(MODEL_DIR,"mask_rcnn_fall_0010.h5")


		#WEIGHTS_PATH = "/home/andmerc/Thesis/Project/ProjectTwo/mask_rcnn_fall_0010.h5" 
		#"/home/andmerc/Thesis/Project/ProjectTwo/mask_rcnn_fall_0010.h5" 

		config = codee.BasicConfig()

		DATASET_DIR = os.path.join(MODEL_DIR,"Data")

		#DATASET_DIR = "/home/andmerc/Thesis/Data"

		class InferenceConfig(config.__class__):
		    GPU_COUNT = 1
		    IMAGES_PER_GPU = 1

		config = InferenceConfig()
		#config.display()

		DEVICE = "/cpu:0"  

		TEST_MODE = "inference"

		def get_ax(rows=1, cols=1, size=16):
		    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
		    return ax

		dataset = codee.FallDataset()
		dataset.load_custom(DATASET_DIR, "val")


		dataset.prepare()

		print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

		with tf.device(DEVICE):
		    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
		                              config=config)

		model.load_weights(WEIGHTS_PATH, by_name=True)

		
		
		image = load_image(image_file)
		
		image = np.array(image)

		if image.shape[2] > 3:
			image = image[:,:,:3]

		st.header("Processing...")

		progress_bar = st.progress(0)

		for completion_rate in range(100):
			time.sleep(0.6)
			progress_bar.progress(completion_rate + 1)

		pathOut = os.path.join(MODEL_DIR,"Output")

		results = model.detect([image], verbose=1)
		#print(results)
		r = results[0]
		x = r['class_ids']

		#print(type(x))
		# Display results

		ax = get_ax(1)
		r = results[0]

		save_image_name = "final"

		finalpath = pathOut +"/" + save_image_name +".jpg"


		visualize.save_image(image,save_image_name ,r['rois'], r['masks'], r['class_ids'],r['scores'],dataset.class_names,save_dir=pathOut)

		if not x.size:
			st.header("No Fall Instance Detected on the image")
			st.image(load_image(image_file),height=400,width=400)
		else:
			st.header("Fall Instance Detected on the image")
			final_img = io.imread(finalpath)

			st.image(final_img, height=400, width=400)

if __name__ =='__main__':
	main() 