import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

def bin_to_num(data):
    binnedinc = []
    for i in data["binnedinc"]:
        # first remove the parantheses and brackets
        i = i.strip("()[]")
        
        # split the string into a list after splitting wrt ,
        i = i.split(",")

        # convert the list into a tuple
        i = tuple(i)

        # convert individual elements to float
        i = tuple(map(float, i))

        # convert the tuple to a list
        i = list(i)

        # append the list to the "binnedinc" list
        binnedinc.append(i)

    data["binnedinc"] = binnedinc

    # make a new column lower and upper bound
    data["lower_bound"] = [i[0] for i in data["binnedinc"]]
    data["upper_bound"] = [i[1] for i in data["binnedinc"]]
    
    data["median"] = (data["lower_bound"] + data["upper_bound"]) / 2
    data.drop("binnedinc", axis=1, inplace=True)
    return data

def cat_to_col(data):
    # make a new column by splitting the geography column
    data["country"] = [i.split(",")[0] for i in data["geography"]]
    data["state"] = [i.split(",")[1] for i in data["geography"]]
    # drop the geography column
    data.drop("geography", axis=1, inplace=True)
    return data

def one_hot_encoding(data):
    # select categorical columns
    categorical_columns = data.select_dtypes(include=["object"]).columns
    # one hot encoding these selected categorical columns
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    one_hot_encoded = one_hot_encoder.fit_transform(data[categorical_columns])
    # now convert the encoded array to data frame
    one_hot_encoded = pd.DataFrame(
        one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns)
    )
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, one_hot_encoded], axis=1)
    return data
