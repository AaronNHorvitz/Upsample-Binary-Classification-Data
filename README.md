# Upsample-Binary-Classification-Data
Code to upsample binary classification data when you have a rare event problem. 

Rare events occure when you have one class that is less than 10% of the total.  This is also called imbalanced data. 

This code takes the training data, after it's been split by scikit learn, and then determines which classified value 
occures the least and which one occures the most. It then upsamples the lowest occuring value to equal the the value 
that occures the most.  This treatment often helps improve the performance of classification models, especially when one class 
is a rare event.   

Exmample code:
#Returns upsampled X_train and y_train data, after it's been split by scikit-learn.

X_train, y_train = upsample_data(X_train, y_train)
