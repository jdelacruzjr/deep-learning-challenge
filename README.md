# Machine Learning and Neural Networks - Successfully Funding Ventures with the Non-profit organization Alphabet Soup

![neural_network](Images/neural_network.png)
## Background

This repository is designed to help the non-profit Alphabet Soup select applicants to fund by selecting ventures with the best chances of success.

### Step One: Preprocess the Data
Started by uploading the starter .ipynb file to Google Colab and used Pandas and `StandardScaler()` to preprocess the data

* Read in the `charity_data.csv` to a Pandas DataFrame, and identified the following within the dataset:
* Variable(s) are the target(s) for your model?
* Variable(s) are the feature(s) for your model?
* Dropped the `EIN` and `NAME` columns.
* Determined the number of unique values for each column.
* For columns that have more than 10 unique values, determined the number of data points for each unique value.
* Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
* Used `pd.get_dummies()` to encode categorical variables.
* Split the preprocessed data into a features array, `X`, and a target array, `y`. Used these arrays and the `train_test_split` function to split the data into training and testing datasets.
* Scaled the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.

### Step Two: Compile, Train, and Evaluate the Data
Used `TensorFlow` to design a neural network, or deep learning model to create a binary classifciation model that can predict if an Alphabet Soup funded organization will be successful based on the features in the dataset.

* Continued using the file in Google Colab in which I performed the preprocessing steps from Step 1.
* Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
* Created the first hidden layer and choose an appropriate activation function. Added a second hidden layer with an appropriate activation function, when deemed necessary.
* Created an output layer with an appropriate activation function.
* Checked the structure of the model.
* Compiled and train the model.
* Evaluated the model using the test data to determine the loss and accuracy.
* Saved and exported results to an HDF5 file. Named the file `AlphabetSoupCharity.h5`.
### Step Three: Optimize the Model
Used `TensorFlow` and attempted to optimize the model to achieve a target predictive accuracy score higher than 75%.

* Dropping more or fewer columns
* Increasing the number of values for each bin
* Added more neurons to a hidden layer
* Added and reduced the number of epochs to the training regimen

Was unsuccessful at achieving an accuracy score higher than 75%, through the attempts above.
### Notes & References:

  I created this shareable link to my repository <https://github.com/jdelacruzjr/deep-learning-challenge.git> and submitted it to <https://bootcampspot-v2.com>
### Copyright

USGS Image © 2019 All Rights Reserved.

Trilogy Education Services © 2023. All Rights Reserved.