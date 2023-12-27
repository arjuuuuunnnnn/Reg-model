from sklearn.model_selection import train_test_split

def find_constant_columns(dataframe):
    '''
    this function is used to identify the columns with same value for all the rows and returns a list containing columns of single values
    '''
    constant_columns = []
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()

        if len(unique_values) == 1:
            constant_columns.append(column)
    return constant_columns

def delete_constant_columns(dataframe, columns_to_delete):
    '''
    this line is used to drop the columns which are constants
    anyways in this case we have no such columns
    '''
    dataframe = dataframe.drop(columns_to_delete, axis=1)
    return dataframe

def find_columns_with_few_values(dataframe, treshold):
    few_values_columns = []
    for column in dataframe.columns:

        unique_values_count = len(dataframe[column].unique())

        if unique_values_count < treshold:
            few_values_columns.append(column)
    return few_values_columns

def find_duplicate_rows(dataframe):
    duplicate_rows = dataframe[dataframe.duplicated()]
    return duplicate_rows

def delete_duplicate_rows(dataframe):
    dataframe = dataframe.drop_duplicates(keep="first")
    return dataframe

def drop_and_fill(dataframe):
    cols_to_drop = dataframe.columns[dataframe.isnull().mean() > 0.5]
    # this above particular line is for if the data has missing values more than 
    # 50% then we are going to drop that column
    dataframe = dataframe.drop(cols_to_drop, axis=1)
    # fill the remaining missing values with the mean of the column
    # if the missing values is less than 50%
    dataframe = dataframe.fillna(dataframe.mean())
    return dataframe

def split_data(dataframe, target_column):
    x = dataframe.drop(target_column, axis=1)
    y = dataframe[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test
