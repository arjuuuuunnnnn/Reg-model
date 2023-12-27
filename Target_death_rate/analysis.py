import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import normaltest

data = pd.read_csv("./Target_death_rate/data/cancer_reg.csv")

print(data.head())


# understand the shape and other things by plotting the graphs
# columns which you want to 


mean_ = data["avganncount"].mean()
std_ = data["avganncount"].std()
print(f"Mean : {mean_} and standard deviation is : {std_}")

plt.hist(data["column"], bins=20)
plt.xlabel("avgAnnCount")
plt.ylabel("Frequency")
plt.title("Histogram of avgAnnCount")
plt.show()

fig = go.Figure(data=[go.Box(y=data['avganncount'], boxpoints="outliers", jitter=0)])
fig.update_layout(
    title='Boxplot of Avgnncount',
    yaxis_title="Death rate",
    width=700,
    height=500
)
fig.show()

corr = data["avganncount"].corr(data["target_deathrate"])
print("Correlation between avgAnnCount and Target_deathrate is : ",corr)

# then plot a scatter plot to visualize the relationship between avgAnnCount and Target_deathrate

plt.scatter(data["avganncount"], data["target_deathrate"])
plt.xlabel("avgAnnCount")
plt.ylabel("target_deathrate")
plt.title("Scatter plot of avgAnnCount vs target_deathrate")
plt.show()

# do this for all the columns which you want to get to know the outliers

# Normal test for detection of outliers

numerical_columns = processed_data.select_dtypes(include=np.number)
gaussian_cols = []
non_gaussian_cols = []
for col in numerical_columns:
    stat, p = normaltest(processed_data[col])
    print('Statistics=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        gaussian_cols.append(col)
    else:
        non_gaussian_cols.append(col)

print(gaussian_cols)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
for i, col in enumerate(gaussian_cols):
    processed_data[col].plot(kind='density', ax=axes, subplots=True, sharex=False)
plt.tight_layout()
plt.show()



# in the z-score test any value goes beyond +3 or -3 will be outliers
def deal_with_outliers(df, col, basic_info):
    highest_allowed = basic_info[col]["mean"] + 3*basic_info[col]["std"]
    lowest_allowed = basic_info[col]["mean"] - 3*basic_info[col]["std"]
    df = df[(df[col] > highest_allowed) | (df[col] < lowest_allowed)]
    return df
cols_have_outliers = []
for col in gaussian_cols:
    df = deal_with_outliers(gaussian_data, col, basic_info_gaussian)
    shape = df.shape
    if shape[0] > 0:
        cols_have_outliers.append(col)
print(cols_have_outliers)
# this above snippet of code is to identify and find the outliers in columns 

# now the way to handle outliers is 
# one trimming and other capping

# trimming
for cols in cols_have_outliers:
    highest_allowed = basic_info[col]["mean"] + 3*basic_info[col]["std"]
    lowest_allowed = basic_info[col]["mean"] - 3*basic_info[col]["std"]
    trimmed_data = processed_data[(processed_data[col] < highest_allowed) & (processed_data[col] > lowest_allowed)]

# capping
# it is replacing the outliers with maximum or minimum values
# to put it another way .., this is replacing the upper outliers with the value of z-score = +3 and replacing the lower outliers with the value z-score with -3
# this is done when I don't want to loose the number of rows
for col in cols_have_outliers:
    highest_allowed = basic_info[col]["mean"] + 3*basic_info[col]["std"]
    lowest_allowed = basic_info[col]["mean"] - 3*basic_info[col]["std"]
    capped_data = processed_data.copy()
    capped_data.loc[capped_data[col] > highest_allowed, col] = highest_allowed
    capped_data.loc[capped_data[col] < lowest_allowed, col] = lowest_allowed


# To identify the skewness
# if the unique
cols_to_remove = []
for col in processed_data.columns:
    if processed_data[col].nunique() < 10:
        # the above line is that if the number of unique values in that current column
        cols_to_remove.append(col)
print(len(cols_to_remove))
# if there are no more unique values then data would not be skewed 
# instead we get a graph with slope almost 0
data_for_skewness = processed_data.drop(cols_to_remove, axis=1)
