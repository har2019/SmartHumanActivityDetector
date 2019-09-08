# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
from scipy import signal
import matplotlib.pyplot as plt 
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from Activity import Activity

# global const variables
DATASET_NUM = 19
COLUMN_NUM = 25
TRAINING_DATASET_PERCENTAGE = 0.8
SAMPLE_SEGMENT_SIZE = 1000
TRAINING_DATA_OUTPUT = 'training_data.csv'
TESTING_DATA_OUTPUT = 'testing_data.csv'

class HumanActivityRecognition(object):
    """This class is to provide a friendly interface to process DaLiAc Database, including data visulaization, noise removing, featuring, train and evaluation classification modal 
    """
    def __init__(self, dataset_path: str):
        """ Initialize a new object with a give dataset path, the user doesn't care the exact path.
            Args:
                dataset_path (str): a existed folder path including all the DaLiAc dataset files
        """
        self.dataset_path = dataset_path

    def read_dataset_as_pd(self, dataset: int) -> pd.DataFrame:
        """ Read given dataset file
            Args:
                activity (Activity): the enum item of Activity class, for example, for sitting dataset, the parameter should be Activity.SITTING
        """
        return pd.read_csv('{}/dataset_{}.txt'.format(self.dataset_path, dataset), sep=',', header=None)

    def data_visulization(self, activity: Activity, dataset: int, columns=range(DATASET_NUM)):
        """ Plot give activity and sensors data
            Args:
                activity (Activity): the enum item of Activity class, for example, for sitting dataset, the parameter should be Activity.SITTING
                dataset (int): the dataset index attempt to visualize
                columns (list): it contains given int list, each element represents a column index in dataset
            Examples:
                To show the sitting activity chest gyroscope data(from column 6 to 9) in dataset_1
                >>> har = HumanActivityRecognition("dataset/")
                >>> har.data_visulization(Activity.SITTING, 1, range(6, 9))
        """
        # read dataset file
        df = self.read_dataset_as_pd(dataset)
        df_activity = df[df[COLUMN_NUM-1] == activity.value].values
        plt.plot(df_activity[:, columns])
        plt.show()

    def noise_removing(self, arr: np.ndarray, columns=range(COLUMN_NUM-1)) -> np.ndarray:
        """ Remove noise for give N-dimension array, if columns given, only process on give columns, whether plot it out is also optional.
            Args:
                arr (np.ndarray): the input N-dimension array
                columns (list): default value is all the columns except for label column
            Returns:
                the arr removed noise
            Examples:
                For current dataset, to remove noise for sitting activity we can do that:
                >>> har = HumanActivityRecognition("dataset/")
                >>> df_activity = df[df[24] == Activity.SITTING.value].values
                >>> df_activity = har.noise_removing(df_activity)
        """
        # Butterworth low-pass filter.
        b, a = signal.butter(4, 0.04, 'low', analog=False)
        for i in columns:
            arr[:, i] = signal.lfilter(b, a, arr[:, i])
        return arr

    def feature_selection(self, columns=range(COLUMN_NUM-1)):
        """ According to selected columns to produce training and testing data
            Args:
                columns (list): default value is all the columns except for label column
            Examples:
                If only consider select first 3 columns to produce feature data:
                >>> har = HumanActivityRecognition("dataset/")
                >>> har.feature_selection(columns=range(3))
        """
        training = np.empty(shape=(0, len(columns)*3 + 1))
        testing = np.empty(shape=(0, len(columns)*3 + 1))
        # deal with each dataset file
        print("{} datasets is processing...".format(DATASET_NUM))
        for i in range(DATASET_NUM):
            df = self.read_dataset_as_pd(i+1)
            for activity in list(Activity):
                # remove noise for current activity lable
                activity_data = df[df[COLUMN_NUM-1] == activity.value].values
                activity_data = self.noise_removing(activity_data)
                
                # split dataset into training data and testing data
                datat_len = len(activity_data)
                training_len = math.floor(datat_len * TRAINING_DATASET_PERCENTAGE)
                training_data = activity_data[:training_len, :]
                testing_data = activity_data[training_len:, :]

                # aggregate data by segment
                training_sample_number = training_len // SAMPLE_SEGMENT_SIZE + 1
                testing_sample_number = (datat_len - training_len) // SAMPLE_SEGMENT_SIZE + 1

                # append the current dataset's feature dataset into output dataset
                training = np.concatenate((training, self.sample_data(training_data, training_sample_number, columns)), axis=0)
                testing = np.concatenate((testing, self.sample_data(testing_data, testing_sample_number, columns)), axis=0)

        # output to files
        df_training = pd.DataFrame(training)
        df_testing = pd.DataFrame(testing)
        df_training.to_csv(TRAINING_DATA_OUTPUT, index=None, header=None)
        df_testing.to_csv(TESTING_DATA_OUTPUT, index=None, header=None)
        print("Training data is output into {}".format(TRAINING_DATA_OUTPUT))
        print("Testing data is output into {}".format(TESTING_DATA_OUTPUT))

    def sample_data(self, data: np.ndarray, sample_number:int, columns=range(COLUMN_NUM-1)) -> np.ndarray:
        """ Split dataset into given number of segments and for each segment, calculate is max, min and mean value as features
            Args:
                data (np.ndarray): a N-dimension dataset as input
                sample_number (int): segment number
                columns: default value is all the columns except for label column
            Returns:
                concated feature dataset
        """
        sample_output = np.empty(shape=(0, len(columns)*3 + 1))
        for s in range(sample_number):
            if s < sample_number - 1:
                sample_data = data[SAMPLE_SEGMENT_SIZE*s:SAMPLE_SEGMENT_SIZE*(s + 1), :]
            else:
                sample_data = data[SAMPLE_SEGMENT_SIZE*s:, :]
            
            sample_segment = []
            for i in columns:
                sample_segment.append(np.min(sample_data[:, i]))
                sample_segment.append(np.max(sample_data[:, i]))
                sample_segment.append(np.mean(sample_data[:, i]))
            sample_segment.append(sample_data[0, -1])
            sample_segment = np.array([sample_segment])
            sample_output = np.concatenate((sample_output, sample_segment), axis=0)
        return sample_output




if __name__ == '__main__':
    har = HumanActivityRecognition("dataset/")
    # har.data_visulization(1)
    # df = har.read_dataset_as_pd(Activity.SITTING)
    # df_activity = df[df[24] == Activity.SITTING.value].values
    # df_activity = har.noise_removing(df_activity)
    # plt.plot(df_activity[:, range(COLUMN_NUM-1)])
    # plt.show()
    har.feature_selection(columns=range(3))