import pandas as pd

# create Table Data Frame
data = {
    'table_data': {
        'ages': [14, 18, 24, 42],
        'heights': [1.65, 1.80, 1.76, 1.88],
        'weight': [45, 60, 82, 65]
    },
    'index_data': ['Katya', 'Egor', 'Eldar', 'Danya']
}
df = pd.DataFrame(data['table_data'], index=data['index_data'])

# add a new column (Series)
df['BMI'] = df['weight'] / (df['heights']**2)

# get all table
print(f'{df}\n')

# get only one raw by index
print(f'{df.loc["Danya"]}\n')

# get column by Series
print(f'{df[["ages"]]}\n')

# Slices
print(f'{df.iloc[1]}\n')  # all information about Egor
print(f'{df.iloc[2:]}\n')  # all information about Eldar and Danya
print(f'{df.iloc[1:3]}\n')  # all information about Egor and Eldar

# Conditionals
# And
print(f'{df[(df["ages"] > 17) & (df["heights"] > 179)]}\n')  # all information about Egor and Danya
# Or
print(f'{df[(df["ages"] > 17) | (df["heights"] > 179)]}\n')  # all information about Egor, Eldar and Danya

# Get data by Series and conditional
print(f'{df["heights"][df["ages"] == 42]}\n')

# get count, mean, std, min, max
print(f'{df.describe()}\n')
# get count, mean, std, min, max from one column
print(f'{df["ages"].describe()}\n')

# get value frequency
print(df['ages'].value_counts())

# Grouping
df.groupby('BMI')['ages'].sum()

# make tables from files (csv, json, xls, sql...)
dsh = pd.read_excel("dashboard.xls")

dsh.set_index("Product", inplace=True)  # set another column

# delete column (axis=1)
dsh.drop("Sponsored brands (HSA)", axis=1, inplace=True)
# delete row (axis=0)


print(dsh['Sales'])
print(dsh.head(10))  # get first 10 rows
print(dsh.tail(10))  # get last 10 rows
print(dsh.info())  # get info about table
