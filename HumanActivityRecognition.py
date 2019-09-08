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

class HumanActivityRecognition(object):
    """This class is to provide a friendly interface to process DaLiAc Database, including data visulaization, noise removing, featuring, train and evaluation classification modal 
    """
    def __init__(self, dataset_path: str):
        """ Initialize a new object with a give dataset path, the user doesn't care the exact path.
            Args:
                dataset_path (str): a existed folder path including all the DaLiAc dataset files
        """
        self.dataset_path = dataset_path

    def read_dataset_as_pd(self, activity: Activity) -> pd.DataFrame:
        """ Read given activity dataset file
            Args:
                activity (Activity): the enum item of Activity class, for example, for sitting dataset, the parameter should be Activity.SITTING
        """
        return pd.read_csv('{}/dataset_{}.txt'.format(self.dataset_path, activity.value), sep=',', header=None)

    def data_visulization(self, activity: Activity, columns=range(DATASET_NUM)):
        """ Plot give activity and sensors data
            Args:
                activity (Activity): the enum item of Activity class, for example, for sitting dataset, the parameter should be Activity.SITTING
                columns (list): it contains given int list, each element represents a column index in dataset
            Examples:
                To show the sitting activity chest gyroscope data(from column 6 to 9)
                >>> har = HumanActivityRecognition("dataset/")
                >>> har.data_visulization(Activity.SITTING, range(6, 9))
        """
        # read dataset file
        df = self.read_dataset_as_pd(activity)
        df_activity = df[df[COLUMN_NUM-1] == activity.value].values
        plt.plot(df_activity[:, columns])
        plt.show()

    def noise_removing(self, arr: np.array, columns=range(COLUMN_NUM-1)) -> np.array:
        """ Remove noise for give N-dimension array, if columns given, only process on give columns, whether plot it out is also optional.
            Args:
                arr (np.array): the input N-dimension array
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


if __name__ == '__main__':
    har = HumanActivityRecognition("dataset/")
    # har.data_visulization(Activity.SITTING)
    df = har.read_dataset_as_pd(Activity.SITTING)
    df_activity = df[df[24] == Activity.SITTING.value].values
    df_activity = har.noise_removing(df_activity)
    plt.plot(df_activity[:, range(COLUMN_NUM-1)])
    plt.show()