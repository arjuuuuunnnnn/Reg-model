import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import statsmodel.api as sm 

from data_processing import split_data

def correlation_among_numeric_features(df, cols):
    numeric_col = df[cols]
    corr = numeric_col.corr()

    corr_features = set()
    for i in range(i):
        for j in range(j):
            if abs(corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                corr_features.add(colname)

    return corr_features

def lr_model(x_train, y_train):
    # create a fitted model
    x_train_with_intercept = sm.add_constant(x_train)
    lr = sm.OLS(y_train, x_train_with_intercept).fit()
    return lr

def identify_significant_vars(lr, p_value_treshold=0.05):
    # print the p values
    print(lr.pvalues)
    # print the r-squared value for the model
    print(lr.rsquared)
    # print the adjusted value for the model
    print(lr.rsquared_adj)
    # identify the significant variables
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var]]
    return significant_vars

if __name__ == "__main__":
    capped_data = pd.read_csv("capped_data.csv")
    print(capped_data.shape)

    corr_features = correlation_among_numeric_features(capped_data, capped_data.columns)
    print(corr_features)
    # after printing put the same columns below
    highly_corr_cols = [
        "povertypercent",
        "median",
        "pctprivatecoveragealone",
        "medianagefemale",
        "pctempprivcoverage",
        "pctblack",
        "popest2015",
        "pctmarriedhouseholds",
        "upper_bound",
        "lower_bound",
        "pctprivatecoverage",
        "medianagemale",
        "state_ District of Columbia",
        "pctpubliccoveragealone",
    ]
    cols = [col for col in capped_data.columns if col not in highly_corr_cols]
    x_train, x_test, y_train, y_test = split_data(capped_data[cols], "target_deathrate")
    lr = lr_model(x_train, y_train)
    summary = lr.summary()
    print(summary) # for this we get a lot of columns

    # so identify significant vars and use them so that it would be short and effective
    significant_vars = identify_significant_vars(lr)
    print(len(significant_vars))

    significant_vars.remove("const")
    lr = lr_model(x_train[significant_vars], y_train)
    summary = lr.summary()
    print(summary)
