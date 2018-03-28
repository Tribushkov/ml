import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib


# Load the data set into pandas data frame
df = pd.read_csv("ml_house_data_set.csv")

# Lets do feature engeneering work
# Remove the fields from the data set that we don't want to include in our model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categorical data with one-hot encoded data
# 'get dummies' function for one-hot encoding
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the sale price from the feature data
del features_df['sale_price']

# Create the X and y arrays
# X - input features
# Y - expected output to predict
# as_matrix - to make sure its numpy matrix and not pandas data frame
X = features_df.as_matrix()
y = df['sale_price'].as_matrix()

# !!! Now we have our features ready for ML use !!!

# Split the data set in a training set (70%) and a test set (30%)
# And shuffles data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Let's create and train our Machine Learning model
# Fit regression model
# set hyperparameters
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000, # number of decision trees to build. more accurate and more time
    learning_rate=0.1, # how each decision tree influences overall prediction; lover rate - higher accuracy
    max_depth=6, # max layers of each decision tree
    min_samples_leaf=9, # how many times a value must appear in our training set; at least 9 houses should have same properties before we consider it meaningful
    max_features=0.1, # % of features which we randomly choose to consider each time we create a branch in our decision tree
    loss='huber', # controls how sci-kit learn calculates error rate
    random_state=0
)

# tell our model to train!
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_house_classifier_model.pkl')

# !!! now we can run the program !!!!

# next step is to find out how well our model is peforming

# Find the error rate on the training set
# mean absolute error looks at every prediction and gives an avarage of how wrong it was across all predictions
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)

# run and see how it works
