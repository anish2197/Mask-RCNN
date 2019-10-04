"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import skimage.draw
import time


import math, json, os, sys
import pickle
import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image as keras_image
from glob import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions



# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
save_images = True
MODEL_TYPE = "binary"
PLC = False
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

NUM_EPOCHS = 100

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two list_filenames = list(set(df["filename"]))images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 10
    STEPS_PER_EPOCH = 20
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5
    #DETECTION_NMS_THRESHOLD = 0.9
    #DETECTION_MAX_INSTANCES = 10
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1280

    VALIDATION_STEPS = 20

############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        
            
        if MODEL_TYPE == "9class" :
            self.add_class("balloon", 1, "handling")
            self.add_class("balloon", 2, "moulding")
            self.add_class("balloon", 3, "scratch")
            self.add_class("balloon", 4, "bubble")
            self.add_class("balloon", 5, "lint")
            self.add_class("balloon", 6, "overspray")
            self.add_class("balloon", 7, "patch")
            self.add_class("balloon", 8, "others")
        elif MODEL_TYPE == "binary":
            self.add_class("balloon", 1, "damage")

        # Train or validation dataset?
        CSV_PATH_TRAIN = dataset_dir+"/train_labels.csv"
        CSV_PATH_VAL = dataset_dir+"/val_labels.csv"
        assert subset in ["train", "val"]
        if subset == "train":
            df = pd.read_csv(CSV_PATH_TRAIN)
        elif subset == "val":
            df = pd.read_csv(CSV_PATH_VAL)
        dataset_dir = os.path.join(dataset_dir, subset)
        list_filenames = list(set(df["filename"]))
        for l in list_filenames:

            temp_df = df[df["filename"] == l]
            polygons = []
            names = []
            names = []
            for i,rows in temp_df.iterrows():
                temp_poly = {}
                temp_name = {}
                temp_poly["names"] = "polygon"
                temp_poly['all_points_x'] = [rows['xmin'],rows['xmax'],rows['xmax'],rows['xmin']]
                temp_poly['all_points_y'] = [rows['ymin'],rows['ymin'],rows['ymax'],rows['ymax']]
                temp_name["name"] = rows['class']
                temp_name['image_quality'] = {
                    'frontal': True,
                    'good_illumination': True,
                    'good': True
                }
                temp_name['type'] = 'unknown'
                polygons.append(temp_poly)
                names.append(temp_name)
            image_path = os.path.join(dataset_dir, l)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=l,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        #annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        #annotations = list(annotations.values())  # don't need the dict keys
        #df = pd.read_csv(CSV_PATH)
        #list_filenames = list(set(df["filename"]))

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        #annotations = [a for a in annotations if a['regions']]

        # Add image
        
        """
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
        """
    def load_balloon_eval(self, dataset_dir):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset?
        #CSV_PATH_TRAIN = dataset_dir+"/train_labels.csv"
        #CSV_PATH_VAL = dataset_dir+"/val_labels.csv"
        #assert subset in ["train", "val"]
        #if subset == "train":
        #    df = pd.read_csv(CSV_PATH_TRAIN)
        #elif subset == "val":
        #    df = pd.read_csv(CSV_PATH_VAL)
        #dataset_dir = os.path.join(dataset_dir, subset)BATCH_SIZE


        list_filenames = list(os.listdir(dataset_dir))

        print("NUMBER OF IMAGES :", len(list_filenames))
        #list_filenames = list(set(df["filename"]))
        for l in list_filenames:
            print(l)
            #temp_df = df[df["filename"] == l]
            """
            polygons = []
            names = []
            names = []
            for i,rows in temp_df.iterrows():
                temp_poly = {}
                temp_name = {}
                temp_poly["names"] = "polygon"
                temp_poly['all_points_x'] = [rows['xmin'],rows['xmax'],rows['xmax'],rows['xmin']]
                temp_poly['all_point
                s_y'] = [rows['ymin'],rows['ymin'],rows['ymax'],rows['ymax']]
                temp_name["name"] = rows['class']
                temp_name['image_quality'] = {
                    'frontal': True,
                    'good_illumination': True,
                    'good': True
                }
                temp_name['type'] = 'unknown'
                polygons.append(temp_poly)
                names.append(temp_name)
            """
            image_path = os.path.join(dataset_dir, l)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=l,  # use file name as a unique image id
                path=image_path,
                width=width, height=height)
            

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=NUM_EPOCHS,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    #gray = image
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask,gray, image).astype(np.uint8)
        #splash = np.where(mask, image, image).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
        #splash = image.astype(np.uint8)
    return splash

def evaluation(model, image_path=None, video_path=None):
    assert image_path or video_path
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0

    PLC = True
    PLC_THRESH = 0.5

    plc_model_dir = "/home/faurecia/FAQT-retinanet/Mask_RCNN/bounding boxes/nasnet.130.h5"
    plc_weights_dir = "/home/faurecia/FAQT-retinanet/Mask_RCNN/bounding boxes/nasnet.130.h5"

    plc_model = load_model(plc_model_dir)
    #plc_model = keras.applications.resnet.ResNet50()
    #plc_model.load_weights(plc_weights_dir)
    skipped_bad2 = 0
    skipped_bad1 = 0

    cnt_remove = 0
    cnt_bb_box = 0

    small_side_list = []
    big_side_list = []
    avg_score_good = 0
    # Image or video?

    list_filenames = list(os.listdir(image_path+"/good/"))
    
    #list_filenames = list(os.listdir("/home/faurecia/FAQT-retinanet/Mask_RCNN/val_images/small defects/"))
    print("**********FOR GOOD IMAGES : ")
    print("**********NUMER OF GOOD IMAGES : ", len(list_filenames))
    
    cnt = 0
    if image_path:

        for i in list_filenames:
        # Run model detection and generate the color splash effect
            cnt += 1
            print(cnt,"/",len(list_filenames))
            print("Running on {}".format(i))
            # Read image
            image = skimage.io.imread(image_path+"/good/"+i)
            # Detect objects
            start = time.time()
            r = model.detect([image], verbose=0)[0]
            #print("processing time: ", time.time() - start)
            
            num_preds = r["rois"].shape[0]
            final_num_preds =  num_preds
            if PLC == True:
                for j in range(num_preds):

                    #print(image.shape)
                    x1, y1, x2, y2 = r["rois"][j]
                    width = x2 - x1
                    height = y2 - y1
                    if(width <= 224 and height <=224 and width >= 10 and height >=10):
                        #print(r["scores"][j])
                        mid_x = int((x2+x1)/2)
                        mid_y = int((y2+y1)/2)

                        new_x1 = mid_x - 112
                        new_x2 = mid_x + 112
                        new_y1 = mid_y - 112
                        new_y2 = mid_y + 112

                        #print(x1,x2,y1,y2,mid_x,mid_y,new_x1,new_x2,new_y1,new_y2)

                        if (new_x1<0 or new_y1<0 or new_x2>2590 or new_y2>1942):
                            skipped_bad1 += 1
                            continue

                        cropped_img = image[new_x1:new_x2, new_y1:new_y2]
                        cnt_bb_box += 1
                        
                        if cropped_img.shape == (224,224,3):
                            x = keras_image.img_to_array(cropped_img)
                            x = x/255
                            x = np.expand_dims(x, axis =0)
                            preds = plc_model.predict(x)
                            pred_prob = preds[0][1]
                            pred = np.argmax(preds)
                            #import pdb; pdb.set_trace()
                            if pred == 1:
                                final_num_preds = final_num_preds - 1
                                print("bb_box removed")
                                cnt_remove += 1
                            else:
                                skimage.io.imsave("/home/faurecia/FAQT-retinanet/Mask_RCNN/val_output/gloss_bb/false_positives/"+str(j)+str(i), cropped_img)
                            #skimage.io.imsave("/home/faurecia/FAQT-retinanet/Mask_RCNN/bounding boxes/gloss/test/"+str(j)+str(i), cropped_img)
                        else :
                            print("bb_box shape mismatch")

                        
                        #skimage.io.imsave("/home/faurecia/FAQT-retinanet/Mask_RCNN/bounding boxes/gloss/test/"+str(j)+str(i), cropped_img)
                    else:
                        skipped_bad2 += 1
            print("**********NUMBER OF PREDS : ",final_num_preds)

            if final_num_preds > 0 :
                false_positives += 1

                if (save_images == True):
                    splash = color_splash(image, r['masks'])
                    # Save output
                    file_name = i
                    skimage.io.imsave("/home/faurecia/FAQT-retinanet/Mask_RCNN/val_output/NEW_VAL/fp/" + i, splash)

            elif final_num_preds <= 0:
                print("NUM_PREDS <= 0")
                true_negatives += 1
            # Color splash
            
            bb_box_acc = (cnt_remove/cnt_bb_box)

        print("**********FALSE POSITIVES :", false_positives)
    
    list_filenames = list(os.listdir(image_path+"/bad"))
    print("**********FOR BAD IMAGES : ")
    print("**********NUMER OF BAD IMAGES :", len(list_filenames))
    skipped_bad2 = 0
    skipped_bad1 = 0
    cnt = 0
    if image_path:


        for i in list_filenames:
        # Run model detection and generate the color splash effect
            cnt += 1
            print(cnt,"/",len(list_filenames))
            print("Running on {}".format(i))
            # Read image
            image = skimage.io.imread(image_path+"/bad/"+i)
            # Detect objects
            start = time.time()
            r = model.detect([image], verbose=0)[0]
            

            num_preds = r["rois"].shape[0]
            final_num_preds = num_preds
            if PLC == True:
                for j in range(num_preds):

                    #print(image.shape)
                    x1, y1, x2, y2 = r["rois"][j]
                    width = x2 - x1
                    height = y2 - y1
                    if(width <= 224 and height <=224 and width >= 10 and height >=10):
                        #print(r["scores"][j])
                        mid_x = int((x2+x1)/2)
                        mid_y = int((y2+y1)/2)

                        new_x1 = mid_x - 112
                        new_x2 = mid_x + 112
                        new_y1 = mid_y - 112
                        new_y2 = mid_y + 112

                        #print(x1,x2,y1,y2,mid_x,mid_y,new_x1,new_x2,new_y1,new_y2)

                        if (new_x1<0 or new_y1<0 or new_x2>2590 or new_y2>1942):
                            skipped_bad1 += 1
                            continue


                        cropped_img = image[new_x1:new_x2, new_y1:new_y2]
                        cnt_bb_box += 1
                        

                        if cropped_img.shape == (224,224,3):
                            x = keras_image.img_to_array(cropped_img)
                            x = x/255
                            x = np.expand_dims(x, axis =0)
                            preds = plc_model.predict(x)
                            print(preds[0][1])
                            pred_prob = preds[0][1]
                            pred = np.argmax(preds)
                            #import pdb; pdb.set_trace()
                            if pred == 1:
                                final_num_preds = final_num_preds - 1
                                skimage.io.imsave("/home/faurecia/FAQT-retinanet/Mask_RCNN/val_output/gloss_bb/false_negatives/"+str(j)+str(i), cropped_img)
                                print("bb_box removed")
                                cnt_remove += 1
                            #skimage.io.imsave("/home/faurecia/FAQT-retinanet/Mask_RCNN/bounding boxes/gloss/test/"+str(j)+str(i), cropped_img)
                        else :
                            print("bb_box shape mismatch")

                    else:
                        skipped_bad2 += 1
            print("processing time: ", time.time() - start)

            print("**********NUMBER OF PREDS : ",final_num_preds)
            if final_num_preds > 0 :
                true_positives += 1

            elif final_num_preds <= 0:
                print("NUM_PREDS <= 0")
                false_negatives += 1

            if (save_images == True):
                # Color splash
                splash = color_splash(image, r['masks'])
                # Save output
                file_name = i
                skimage.io.imsave("/home/faurecia/FAQT-retinanet/Mask_RCNN/val_output/NEW_VAL/fn/" + i, splash)


        print("**********FALSE NEGATIVES :", false_negatives)

    

    

    

    

    



    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    print("False negatives : " + str(false_negatives))
    print("False positives : " + str(false_positives))
    print("True positives : " + str(true_positives))
    print("RECALL : " + str(recall))
    print("PRECISION : " + str(precision))
    #print("F1 SCORE : " + str(f1))
    print("skipped 1 ", skipped_bad1)
    print("skipped 2 ", skipped_bad2)
    print("TOTAL BB : ", cnt_bb_box)
    print("TOTAL BB REMOVED : ", cnt_remove)
    print("BB BOX ACCURACY ", (cnt_remove/cnt_bb_box))
    f1 = (2 * precision * recall) / (precision + recall)

    #print("CONFIDENCE THRESHOLD : " + str(CONF_THRESHOLD))
    #print("Confidence threshold : " + str())
    print("False negatives : " + str(false_negatives))
    print("False positives : " + str(false_positives))
    print("True positives : " + str(true_positives))
    print("RECALL : " + str(recall))
    print("PRECISION : " + str(precision))
    print("F1 SCORE : " + str(f1))
    print("skipped 1 ", skipped_bad1)
    print("skipped 2 ", skipped_bad2)
    print("TOTAL BB : ", cnt_bb_box)
    print("TOTAL BB REMOVED : ", cnt_remove)
    print("BB BOX ACCURACY ", bb_box_acc)
    """
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
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
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()

    print("Saved to ", file_name)
    """
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
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
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--conf', required=False,
                        metavar="confidence threhsold",
                        help='Minimum confidence threhsold')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
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
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
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
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "eval":
        evaluation(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
