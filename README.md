# Machine Learning Applications on Driving Behavior
# HTGR - Hit the Ground Running Team
## Summary

This Machine Learning Project is designed to hone Machine Learning concepts and algorithms. Over the course of a semester we took a [driving behavior dataset](https://www.kaggle.com/datasets/shashwatwork/driving-behavior-dataset), cleaned it up and split it in order to train and test our models. The data set was a car accelerometer and gyroscope time-series data that would be used to determine if the car movements, such as sudden acceleration, sudden right, sudden left or sudden break. With the preprocessed data, developed 3 different models: LSTM (Long short term memory), SVM (Support Vector Machine), and Random Forest Model to predict driving behavior through the preprocessed time-series data.

This project was completed in 3 months using agile methedology and self-study. 

Contributors:
- Team Lead: 
    - Prerna Joshi
- Team Members:
    - Elena Hernandez
    - Tony Gonzalez
    - Yunseon Choi
    - Arnav Sahu
    - Jeffrey Hoang
    - Akshat Bist
    - Michelle Reyes
    - Henry Do
    - Chi Hao Nguyen

# Splitting the dataset
The dataset was split in a 7:1:2 ratio for training, validating and testing purposes. We split the data so that it was proportionally distributed within the labels making it more efficient for our models to understand it quickly instead of just randomly splitting it. More details for this split can be seen in this file https://github.com/cppsea/ML_HTGR/blob/main/DataCleaning/splitAndScale.py 

# LSTM Documentation 
![image](https://github.com/cppsea/ML_HTGR/assets/119718093/e4b1f2ca-f064-41f5-ba22-30e197dac141)

Blue: accuracy of model run with training data

Red: accuracy of model run with validation data

Green: accuracy of model run with test data

For all the runs, did 10 epochs with batch size of 16.
- Run 1
    - Accuracy of model with Training Data: **97.38%**
- Run 2
    - Accuracy of model with Validation Data: **88.87%**
- Run 3
    - Accuracy of model with Training Data: **96.94%**
    - Nothing was changed in the code, but the slight decrease in accuracy could be a result of the new
validation data that was run with the model.
- Run 4
    - Training Data: **97.22%**
    - Nothing was changed in the code again, but the slight increase in accuracy could be a result of
the model having another run with the training data.
- Run 5
    - Validation Data: **95.42%**
    - This increase in accuracy from the first run (using the validation data) can be attributed to the
change in the number of epochs. We increased the number of epochs in the LSTM model from
10 to 30. In general, increasing the number of epochs helps improve accuracy of the model.
However, an excessive number of epochs can lead to overfitting the model, whereas an
insufficient number of epochs can cause underfitting the model.
- Run 6
    - Accuracy of model with Training Data: **97.13%**
    - Nothing was changed in the code; just retraining the model with the training data. The slight
decrease in accuracy could be a result of the new validation data that was run with the model.
- Run 7
    - Accuracy of model with Training Data: **97.10%**
    - Nothing was changed in the code; just retraining the model with the training data. Again, the
slight decrease in accuracy could be a result of the new validation data that was run with the
model.
- Run 8
    - Accuracy of model with Validation Data: **95.46%**
    - Nothing was changed in the code; just ran the model again with the validation data. The slight
increase in accuracy for the run with validation data could be a result of the model having
previous runs with the training data and validation data.
- Run 9
    - Accuracy of model with Test Data: **96.95%**
    - Nothing in the code was changed; ran the model using the test data.


# SVM Documentation 
![image](https://github.com/cppsea/ML_HTGR/assets/119718093/f4c370e0-df74-445f-826b-4066cf6fee84)
Blue: accuracy of model run with training data

Red/Orange: accuracy of model run with validation data

Green: accuracy of model run with test data

- Run 1
    - Accuracy of model using the Training Data: **86.45%**
    - In the first training run, the SVM model achieved an accuracy of 86.45% on the training data. This 
indicates that the model correctly classified 86.45% of the instances in the training set.
- Run 2
    - Accuracy of model using the Validation Data: **72.73%**
    - The model was evaluated on a separate validation dataset, and it achieved an accuracy of 72.73%.
The decrease in accuracy from the training to the validation data suggests a potential issue with 
overfitting. Overfitting occurs when the model performs well on the training data but fails to 
generalize to new, unseen data.
- Run 3
    - Accuracy of model using the Training Data: **81.29%**
    - In the second training run, the model achieved a lower accuracy of 81.29% on the training data. 
This may indicate changes in the data distribution or potential model sensitivity to parameter 
variations.
- Run 4
    - Accuracy of model using the Training Data: **85.81%**
    - The model's accuracy on the training data increased to 85.81% in the third run. Variations in 
accuracy between runs could be attributed to factors like data shuffling, model initialization, or 
random splitting.
- Run 5
    - Accuracy of model using the Validation Data: **84.52%**
    - The accuracy on the validation data in the second run was 84.52%. This indicates how well the 
model generalizes to new, unseen data in the context of the changes made in the second run.
- Run 6
    - Accuracy of model using the Training Data: **85.16%**
    - The model achieved an accuracy of 85.16% on the training data. The fluctuation in accuracy 
suggests sensitivity to training conditions or inherent variability in the data.
- Run 7
    - Accuracy of model using the Training Data: **85.81%**
    - Similar to earlier runs, variations in accuracy may be influenced by factors such as data 
characteristics or model parameters.
- Run 8
    - Accuracy of model using the Validation Data: **86.36%**
    - This provides insight into how well the model generalizes when considering changes made in the 
third run.
- Run 9
    - Accuracy of model using the Test Data: **84.09%**
    - The model's performance on a separate test dataset, not used during training or validation, was 
84.09% in the first run. This reflects the model's ability to generalize to completely new data


# Random Forest Documentation 
![image](https://github.com/cppsea/ML_HTGR/assets/119718093/29b47e61-beb7-4fd2-951c-706111f0fd65)
Blue: accuracy of model run with training data

Red/Orange: accuracy of model run with validation data

Green: accuracy of model run with test data

- Run 1 
    - Accuracy of model using the Training Data: 86.9%
    - Accuracy alternated between 100% and 86.9% between different models created.
Random forest is more complex and prone to overfitting with smaller/less complicated
data sets.
- Run 2
    - Accuracy of model using the Validation Data: 99.81%
    - Overfitting or data memorization seems to have taken effect.
- Run 3
    - Accuracy of model using the Training Data: 100%
    - Model will now repeatedly output accuracy of 100% for any training/validation runs.
- Run 4
    - Accuracy of model using the Training Data: 100%
- Run 5
    - Accuracy of model using the Validation Data: 100%
    - We skip to the final test data set, which also happens to lack complexity for this model.
- Run 6
    - Accuracy of model using the Test Data: 99.92%
    - The accuracy of this model over the other models is a result of the overfitting that occured in the training and validating stages. Because of the data being significanty smaller to the overfitting done, the accuracy for this models test data ended up being much higher than the other two models.
