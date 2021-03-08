import os
import sys
import h5py
import json
import datetime
import numpy as np
import skimage.draw
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

localdirectory = os.path.abspath("../")

# Get the required library
sys.path.append(localdirectory)
# To find local version of the library
import mrcnn.visualize
from mrcnn.config import Config
from mrcnn import model as modellib, utils


coco_weights= os.path.join(localdirectory, "mask_rcnn_coco.h5")

logdirectory = os.path.join(localdirectory, "logs")

class BasicConfig(Config):
    NAME = "fall"
    IMAGES_PER_CPU = 1
    NUM_CLASSES = 1 + 1  # Background + fall
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9
class FallDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        
        self.add_class("fall", 1, "fall")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        fallannotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        
        annotations = list(fallannotations.values())  #Dictionary keys

        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "fall",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            dimone,dimtwo = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[dimone, dimtwo, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Image path"""
        info = self.image_info[image_id]
        if info["source"] == "fall":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
def train(model):
    dataset_train = FallDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # validation dataset
    dataset_val = FallDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
def color_mask(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    if mask.shape[0] > 0:
        fallmask = np.where(mask, image, gray).astype(np.uint8)
    else:
        fallmask = gray
    return fallmask


def detect_and_create_mask(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video
    if image_path:
        print("Running on {}".format(args.image))
        image = skimage.io.imread(args.image)
        r = model.detect([image], verbose=1)[0]
        fallmask = color_mask(image, r['masks'])
        file_name = "fallmask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, fallmask)
    elif video_path:
        import cv2
        #video
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "fallmask_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color mask
                fallmask = color_mask(image, r['masks'])
                # RGB -> BGR to save image to video
                fallmask = fallmask[..., ::-1]
                # Add image to video writer
                vwriter.write(fallmask)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'masky'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=logdirectory,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the mask effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the mask effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "masky":
        assert args.image or args.video,\
               "Provide --image or --video to apply color mask"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations highlighted
    if args.command == "train":
        config = BasicConfig()
    else:
        class InferenceConfig(BasicConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = coco_weights
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "masky":
        detect_and_create_mask(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))











