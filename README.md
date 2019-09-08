# SmartHumanActivityDetector
## Environment Requirements
`python >= 3`

## Prepare Dataset
1. Download dataset from [DaLiAc – Daily Life Activities](https://www.mad.tf.fau.de/research/activitynet/daliac-daily-life-activities/)
2. Unzip the dataset file

## Run
Sample code to run it:
```
from Activity import Activity

if __name__ == '__main__':
    har = HumanActivityRecognition("your_dataset_path")
    # Explore data in dataset_1.txt
    har.data_visulization(Activity.SITTING, 1)
    # Remove noise for all columns
    df = har.read_dataset_as_pd(1)
    df_activity = df[df[24] == Activity.SITTING.value].values
    df_activity = har.noise_removing(df_activity)
    plt.plot(df_activity[:, range(COLUMN_NUM-1)])
    plt.show()
    # Select features and output to training_data.csv and testing_data.csv
    har.feature_selection()
    # Train by KNN
    print(har.classify_by_KNN(10))
    # Train by SVN
    tuned_parameters = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

    print(har.classify_by_SVN(tuned_parameters))

```

Alternatively,
You can run `Run.ipynb` in Jupyter environment, it also gives examples.