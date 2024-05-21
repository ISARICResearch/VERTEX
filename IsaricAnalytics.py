import pandas as pd

def get_variables_type(data):
    final_binary_variables = []
    final_numeric_variables = []
    final_categorical_variables = []

    for column in data:
        column_data = data[column].dropna()

        if column_data.empty:
            continue

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(column_data):
            unique_values = column_data.unique()
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                final_binary_variables.append(column)
            else:
                final_numeric_variables.append(column)
        else:
            unique_values = column_data.unique()
            # Consider column as categorical if it has a few unique values
            if len(unique_values) <= 10:
                final_categorical_variables.append(column)

    return final_binary_variables, final_numeric_variables, final_categorical_variables
