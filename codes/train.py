"""
Training part of my solution to The 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018
Goal of the competition was to create an algorithm to
automate nucleus detection from biomedical images.

author: Inom Mirzaev
github: https://github.com/mirzaevinom
"""
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from config import *
import h5py
import json
import os
import skimage
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle


class KaggleDataset(utils.Dataset):
    """wrapper for loading bowl datasets
    """

    def load_shapes(self, annotations, train_path):
        """initialize the class with dataset info.
        """
        # Add classes
        self.add_class('images', 1, "nucleus")
        self.train_path = train_path

        # annotations = json.load(open("via_region_data.json"))
        # annotations = list(annotations.values())
        # annotations = [a for a in annotations if a['regions']]

        dataset_dir = "../data/stage1_train"
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            image_path = os.path.join(dataset_dir, a['filename'].split(".")[0], "images", a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image("images", image_id=a['filename'], path=image_path, width=width, height=height,
                           polygons=polygons)

    # def load_shapes_test(self, id_list, ):

    def load_image(self, image_id, color):
        """Load image from directory
        """

        info = self.image_info[image_id]
        # path = self.train_path + info['img_name'] + \
        #        '/images/' + info['img_name'] + '.png'
        path = info["path"]
        img = load_img(path, color=color)

        return img

    def image_reference(self, image_id):
        """Return the images data of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'images':
            return info['images']
        else:
            super(self.__class__).image_reference(self, image_id)

    # def load_mask(self, image_id):
    #     """Generate instance masks for images of the given image ID.
    #     """
    #
    #     info = self.image_info[image_id]
    #
    #     path = self.train_path + info['img_name'] + \
    #         '/masks/' + info['img_name'] + '.h5'
    #     if os.path.exists(path):
    #         # For faster data loading run augment_preprocess.py file first
    #         # That should save masks in a single h5 file
    #         with h5py.File(path, "r") as hf:
    #             mask = hf["arr"][()]
    #     else:
    #         path = self.train_path + info['img_name']
    #         mask = []
    #
    #         for mask_file in next(os.walk(path + '/masks/'))[2]:
    #             if 'png' in mask_file:
    #                 mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
    #                 mask_ = np.where(mask_ > 128, 1, 0)
    #                 # Fill holes in the mask
    #                 mask_ = binary_fill_holes(mask_).astype(np.int32)
    #                 # Add mask only if its area is larger than one pixel
    #                 if np.sum(mask_) >= 1:
    #                     mask.append(np.squeeze(mask_))
    #
    #         mask = np.stack(mask, axis=-1)
    #         mask = mask.astype(np.uint8)
    #
    #     # Class ids: all ones since all are foreground objects
    #     class_ids = np.ones(mask.shape[2])
    #
    #     return mask.astype(np.uint8), class_ids.astype(np.int8)

    def load_mask(self, image_id):
        """Generate instance masks for images of the given image ID.
        """

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


def train_validation_split(train_path, seed=10, test_size=0.1):
    """
    Split the dataset into train and validation sets.
    External data and mosaics are directly appended to training set.
    """
    from sklearn.model_selection import train_test_split

    image_ids = list(
        filter(lambda x: ('mosaic' not in x) and ('TCGA' not in x), os.listdir(train_path)))
    mosaic_ids = list(filter(lambda x: 'mosaic' in x, os.listdir(train_path)))
    external_ids = list(filter(lambda x: 'TCGA' in x, os.listdir(train_path)))

    # Load and preprocess the dataset with train image modalities
    df = pd.read_csv('../data/classes.csv')
    df['labels'] = df['foreground'].astype(str) + df['background']
    df['filename'] = df['filename'].apply(lambda x: x[:-4])
    df = df.set_index('filename')
    df = df.loc[image_ids]

    # Split training set based on provided image modalities
    # This ensures that model validates on all image modalities.
    train_list, val_list = train_test_split(df.index, test_size=test_size,
                                            random_state=seed, stratify=df['labels'])

    # Add external data and mos ids to training list
    train_list = list(train_list) + mosaic_ids + external_ids
    val_list = list(val_list)

    return train_list, val_list


class BalloonDataset(utils.Dataset):
    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

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
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
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

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

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

    def load_image(self, image_id, color):

        info = self.image_info[image_id]
        path = info["path"]
        # path = self.train_path + info['img_name'] + \
        #     '/images/' + info['img_name'] + '.png'
        img = load_img(path, color=color)
        # img = cv2.resize(img, (256, 256))
        return img


if __name__ == '__main__':
    import time

    train_path = '../data/stage1_train/'

    start = time.time()

    # Split the training set into training and validation
    # train_list, val_list = train_validation_split(train_path, seed=11, test_size=0.1)

    anno = json.load(open("via_region_data.json"))
    anno = list(anno.values())
    anno = [a for a in anno if a['regions']]
    shuffle(anno)
    # anno_train = anno[0: int(len(anno) * 0.7)]
    # anno_val = anno[int(len(anno) * 0.7):]

    anno_train = anno
    anno_val = anno


    dataset_train = KaggleDataset()
    dataset_train.load_shapes(anno_train, train_path)
    dataset_train.prepare()

    dataset_val = KaggleDataset()
    dataset_val.load_shapes(anno_val, train_path)
    dataset_val.prepare()

    # dataset_train = BalloonDataset()
    # dataset_train.load_balloon("data", "train")
    # dataset_train.prepare()
    #
    # dataset_val = BalloonDataset()
    # dataset_val.load_balloon("data", "train")
    # dataset_val.prepare()

    # Create model configuration in training mode
    config = KaggleBowlConfig()
    config.STEPS_PER_EPOCH = len(anno_train)//config.BATCH_SIZE
    config.VALIDATION_STEPS = len(anno_val)//config.BATCH_SIZE
    config.IMAGES_PER_GPU = 1
    # config.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # config.AUGMENT = False
    config.CROP_SHAPE = np.array([96, 96, 3])
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # model.keras_model.summary()
    # exit()

    # Model weights to start training with
    weights_path = 'kaggle_bowl.h5'
    model.load_weights(weights_path, by_name=True)
    print('Loading weights from ', weights_path)

    # Train the model for 75 epochs
    model.train(dataset_train, dataset_val,
                learning_rate=1e-4,
                epochs=25,
                verbose=1,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=1e-5,
                epochs=50,
                verbose=1,
                layers='heads')
    #
    model.train(dataset_train, dataset_val,
                learning_rate=1e-6,
                epochs=75,
                verbose=1,
                layers='heads')
    print('Elapsed time', round((time.time() - start) / 60, 1), 'minutes')
