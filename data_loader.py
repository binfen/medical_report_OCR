# -*- coding:utf8 -*-

import os
import random
import numpy as np
import math
from PIL import Image

from image_augmentor import ImageAugmentor


class CharsLoader:
    """Class that loads and prepares the OCR chars dataset

    This Class was constructed to read the Chars dataset, separate the
    training, validation and evaluation test. It also provides function for
    geting one-shot task batches.

    Attributes:
        dataset_path: path of Chars Dataset
        train_dictionary: dictionary of the files of the train set (background set). 
            This dictionary is used to load the batch for training and validation.
        evaluation_dictionary: dictionary of the evaluation set. 
        image_width: self explanatory
        image_height: self explanatory
        batch_size: size of the batch to be used in training
        use_augmentation: boolean that allows us to select if data augmentation is 
            used or not
        image_augmentor: instance of class ImageAugmentor that augments the images
            with the affine transformations referred in the paper

    """

    def __init__(self, dataset_path, use_augmentation, batch_size):
        """Inits OmniglotLoader with the provided values for the attributes.

        It also creates an Image Augmentor object and loads the train set and 
        evaluation set into dictionaries for future batch loading.

        Arguments:
            dataset_path: path of chars dataset
            use_augmentation: boolean that allows us to select if data augmentation 
                is used or not       
            batch_size: size of the batch to be used in training，包含正负样本对
            假设batch_size=16，那么image_number=16*2*2
        """

        self.dataset_path = dataset_path
        self.train_dictionary = {}  #set value by hand?
        self.evaluation_dictionary = {}
        self.image_width = 105
        self.image_height = 105
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self.__train_buckets = []
        self.__validation_buckets = []
        self.__evaluation_buckets = []
        self.__current_train_bucket_index = 0
        self.__current_validation_bucket_index = 0
        self.__current_evaluation_bucket_index = 0

        self.load_dataset()

        if (self.use_augmentation):
            self.image_augmentor = self.createAugmentor()
        else:
            self.use_augmentation = []

    def load_dataset(self):
        """Loads the buckets into dictionaries

        Loads the Chars dataset and stores the available images
        for each of the train and evaluation set.

        """

        train_path = os.path.join(self.dataset_path, 'images_background')
        validation_path = os.path.join(self.dataset_path, 'images_evaluation')

        # First let's take care of the train chars
        # if current_bucket_dictionary is null, what will be happened?
        for bucket in os.listdir(train_path):
            bucket_path = os.path.join(train_path, bucket)

            current_bucket_dictionary = {}

            for character in os.listdir(bucket_path):
                character_path = os.path.join(bucket_path, character)

                current_bucket_dictionary[character] = os.listdir(
                    character_path)

            self.train_dictionary[bucket] = current_bucket_dictionary

        # Now it's time for the validation buckets
        for bucket in os.listdir(validation_path):
            bucket_path = os.path.join(validation_path, bucket)

            current_bucket_dictionary = {}

            for character in os.listdir(bucket_path):
                character_path = os.path.join(bucket_path, character)

                current_bucket_dictionary[character] = os.listdir(
                    character_path)

            self.evaluation_dictionary[bucket] = current_bucket_dictionary

    def createAugmentor(self):
        """ Creates ImageAugmentor object with the parameters for image augmentation

        Rotation range was set in -15 to 15 degrees
        Shear Range was set in between -0.3 and 0.3 radians
        Zoom range between 0.8 and 2 
        Shift range was set in +/- 5 pixels
        梯形仿射？

        Returns:
            ImageAugmentor object

        """
        rotation_range = [-15, 15]
        shear_range = [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi]
        zoom_range = [0.8, 2]
        shift_range = [5, 5]

        return ImageAugmentor(0.5, shear_range, rotation_range, shift_range, zoom_range)

    def split_train_datasets(self):
        """ Splits the train set in train and validation for 85% - 15% split

        """
        available_buckets = list(self.train_dictionary.keys())
        number_of_buckets = len(available_buckets)

        train_indexes = random.sample(
            range(0, number_of_buckets - 1), int(0.85 * number_of_buckets))

        # If we sort the indexes in reverse order we can pop them from the list
        # and don't care because the indexes do not change
        train_indexes.sort(reverse=True)

        for index in train_indexes:
            self.__train_buckets.append(available_buckets[index])
            available_buckets.pop(index)

        # The remaining buckets are saved for validation
        self.__validation_buckets = available_buckets

        # get the list of names of buckets
        self.__train_buckets = list(self.train_dictionary.keys())
        self.__evaluation_buckets = list(self.evaluation_dictionary.keys())

    def __convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):
        """ Loads the images and its correspondent labels from the path

        Take the list with the path from the current batch, read the images and
        return the pairs of images and the labels
        If the batch is from train or validation the labels are alternately 1's and
        0's. If it is a evaluation set only the first pair has label 1

        Arguments:
            path_list: list of images to be loaded in this batch
            is_one_shot_task: flag sinalizing if the batch is for one-shot task or if
                it is for training

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros(
            (number_of_pairs, self.image_height, self.image_height, 1)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = Image.open(path_list[pair * 2])
            image = np.asarray(image).astype(np.float64)
            image = image / image.std() - image.mean()
            # [0]代表存放正/负样本
            pairs_of_images[0][pair, :, :, 0] = image

            image = Image.open(path_list[pair * 2 + 1])
            image = np.asarray(image).astype(np.float64)
            image = image / image.std() - image.mean()
            # [1]代表存放对应样本
            pairs_of_images[1][pair, :, :, 0] = image

            if not is_one_shot_task:
                # 因为path_list的奇数代表负样本，偶数代表正样本
                if (pair + 1) % 2 == 0:
                    labels[pair] = 0
                else:
                    labels[pair] = 1

            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        if not is_one_shot_task:
            random_permutation = np.random.permutation(number_of_pairs)
            labels = labels[random_permutation]
            pairs_of_images[0][:, :, :,
                               :] = pairs_of_images[0][random_permutation, :, :, :]
            pairs_of_images[1][:, :, :,
                               :] = pairs_of_images[1][random_permutation, :, :, :]

        return pairs_of_images, labels

    def get_train_batch(self):
        """ Loads and returns a batch of train images

        Get a batch of pairs from the training set. Each batch will contain
        images from a single bucket. I decided to select one single example
        from random n/2(n is the number of batch size) characters in each bucket.
        If the current bucket has lower number of characters than n/2 (some of them have 14)
        we sample repeated classed for that batch per character in the bucket
        to pair with a different categories. In the other half of the batch
        I selected pairs of same characters. In resume we will have a batch
        size of n, with n/2 pairs of different classes and n/2 pairs of the same
        class. Each batch will only contains samples from one single bucket.

        对一个bucket下的所有charactor目录进行遍历，获得按照batch_size大小的image_pairs，
        并且是经过augmented的正负样本对。每次iteration都进行get_train_batch()，

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """

        current_bucket = self.__train_buckets[self.__current_train_bucket_index]
        available_characters = list(
            self.train_dictionary[current_bucket].keys())
        number_of_characters = len(available_characters)

        bacth_images_path = []

        # If the number of classes if less than self.batch_size/2
        # we have to repeat characters
        selected_characters_indexes = [random.randint(
            0, number_of_characters-1) for i in range(self.batch_size)]
        
        for index in selected_characters_indexes:
            current_character = available_characters[index]
            available_images = (self.train_dictionary[current_bucket])[
                current_character]
            image_path = os.path.join(
                self.dataset_path, 'images_background', current_bucket, current_character)

            # Random select a 3 indexes of images from the same character (Remember
            # that for each character we have 20 examples).
            image_indexes = random.sample(range(0, 19), 3)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(image)
            image = os.path.join(
                image_path, available_images[image_indexes[1]])
            bacth_images_path.append(image)

            # Now let's take care of the pair of images from different characters
            image = os.path.join(
                image_path, available_images[image_indexes[2]])
            bacth_images_path.append(image)
            different_characters = available_characters[:]
            different_characters.pop(index)
            different_character_index = random.sample(
                range(0, number_of_characters - 1), 1)
            current_character = different_characters[different_character_index[0]]
            available_images = (self.train_dictionary[current_bucket])[
                current_character]
            image_indexes = random.sample(range(0, 19), 1)
            image_path = os.path.join(
                self.dataset_path, 'images_background', current_bucket, current_character)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(image)

        self.__current_train_bucket_index += 1

        # 超过80%的trainset的bucket时，从头再来
        if (self.__current_train_bucket_index > 5):
            self.__current_train_bucket_index = 0

        images, labels = self.__convert_path_list_to_images_and_labels(
            bacth_images_path, is_one_shot_task=False)

        # Get random transforms if augmentation is on
        if self.use_augmentation:
            images = self.image_augmentor.get_random_transform(images)

        return images, labels

    def get_one_shot_batch(self, support_set_size, is_validation):
        """ Loads and returns a batch for one-shot task images

        Gets a one-shot batch for evaluation or validation set, it consists in a
        single image that will be compared with a support set of images. It returns
        the pair of images to be compared by the model and it's labels (the first
        pair is always 1) and the remaining ones are 0's

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels 1 for same class, 0 for different classes

        """

        # Set some variables that will be different for validation and evaluation sets
        if is_validation:
            buckets = self.__validation_buckets
            current_bucket_index = self.__current_validation_bucket_index
            image_folder_name = 'images_background'
            dictionary = self.train_dictionary
        else:
            buckets = self.__evaluation_buckets
            current_bucket_index = self.__current_evaluation_bucket_index
            image_folder_name = 'images_evaluation'
            dictionary = self.evaluation_dictionary

        current_bucket = buckets[current_bucket_index]
        available_characters = list(dictionary[current_bucket].keys())
        number_of_characters = len(available_characters)

        bacth_images_path = []

        test_character_index = random.sample(
            range(0, number_of_characters), 1)

        # Get test image
        current_character = available_characters[test_character_index[0]]

        available_images = (dictionary[current_bucket])[current_character]

        image_indexes = random.sample(range(0, 20), 2)
        image_path = os.path.join(
            self.dataset_path, image_folder_name, current_bucket, current_character)

        test_image = os.path.join(
            image_path, available_images[image_indexes[0]])
        bacth_images_path.append(test_image)
        image = os.path.join(
            image_path, available_images[image_indexes[1]])
        bacth_images_path.append(image)

        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_characters = number_of_characters
        else:
            number_of_support_characters = support_set_size

        different_characters = available_characters[:]
        different_characters.pop(test_character_index[0])

        # There may be some buckets with less than 20 characters
        if number_of_characters < number_of_support_characters:
            number_of_support_characters = number_of_characters

        support_characters_indexes = random.sample(
            range(0, number_of_characters - 1), number_of_support_characters - 1)

        for index in support_characters_indexes:
            current_character = different_characters[index]
            available_images = (dictionary[current_bucket])[
                current_character]
            image_path = os.path.join(
                self.dataset_path, image_folder_name, current_bucket, current_character)

            image_indexes = random.sample(range(0, 20), 1)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(test_image)
            bacth_images_path.append(image)

        images, labels = self.__convert_path_list_to_images_and_labels(
            bacth_images_path, is_one_shot_task=True)

        return images, labels

    def one_shot_test(self, model, support_set_size, number_of_tasks_per_bucket,
                      is_validation):
        """ Prepare one-shot task and evaluate its performance

        Make one shot task in validation and evaluation sets
        if support_set_size = -1 we perform a N-Way one-shot task with
        N being the total of characters in the bucket

        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """

        # Set some variables that depend on dataset
        if is_validation:
            buckets = self.__validation_buckets
            print('\nMaking One Shot Task on validation buckets:')
        else:
            buckets = self.__evaluation_buckets
            print('\nMaking One Shot Task on evaluation buckets:')

        mean_global_accuracy = 0

        for bucket in buckets:
            mean_bucket_accuracy = 0
            for _ in range(number_of_tasks_per_bucket):
                images, _ = self.get_one_shot_batch(
                    support_set_size, is_validation=is_validation)
                probabilities = model.predict_on_batch(images)

                # Added this condition because noticed that sometimes the outputs
                # of the classifier was almost the same in all images, meaning that
                # the argmax would be always by defenition 0.
                if np.argmax(probabilities) == 0 and probabilities.std()>0.01:
                    accuracy = 1.0
                else:
                    accuracy = 0.0

                mean_bucket_accuracy += accuracy
                mean_global_accuracy += accuracy

            mean_bucket_accuracy /= number_of_tasks_per_bucket

            print(bucket + ' bucket' + ', accuracy: ' +
                  str(mean_bucket_accuracy))
            if is_validation:
                self.__current_validation_bucket_index += 1
            else:
                self.__current_evaluation_bucket_index += 1

        mean_global_accuracy /= (len(buckets) *
                                 number_of_tasks_per_bucket)

        print('\nMean global accuracy: ' + str(mean_global_accuracy))

        # reset counter
        if is_validation:
            self.__current_validation_bucket_index = 0
        else:
            self.__current_evaluation_bucket_index = 0

        return mean_global_accuracy

def main():
    loader = CharsLoader(
        dataset_path="Omniglot Dataset", use_augmentation=True, batch_size=16)
    loader.load_dataset()
    loader.split_train_datasets()
    images, labels = loader.get_train_batch()
    images, _ = loader.get_one_shot_batch(40, is_validation=True)


if __name__ == "__main__":
    main()
