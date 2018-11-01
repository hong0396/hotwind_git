# -*- coding: utf-8 -*-
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures,load_robot_execution_failures
import matplotlib.pylab as plt
# from pandas.core import datetools
import seaborn as sns
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

# download_robot_execution_failures()
df, y = load_robot_execution_failures()
if __name__ == '__main__':
    X = extract_features(df, column_id='id', column_sort='time')#, default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
    X_filtered = extract_relevant_features(df, y,column_id='id', column_sort='time', default_fc_parameters=ComprehensiveFCParameters())
    X_filtered.info()
    X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y, test_size=4)

    cl =DecisionTreeClassifier()
    cl.fit(X_train, y_train)

    print(classification_report(y_test, cl.predict(X_test)))

    cl.n_features_

    cl2 = DecisionTreeClassifier()
    cl2.fit(X_filtered_train, y_train)
    print(classification_report(y_test, cl2.predict(X_filtered_test)))



    