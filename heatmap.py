import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mat
from azureml.opendatasets import Diabetes

diabetes = Diabetes.get_tabular_dataset()
diabetes_df = diabetes.to_pandas_dataframe() # data
attributes = ['Women Entrepreneurship Index', 'Entrepreneurship Index', 'Inflation rete', 'Female Labor Force Participation Rate']
df = pd.DataFrame([
    [1.0, 0.91, -0.46, 0.44],
    [0.91, 1, -0.4, 0.33],
    [-0.46, -0.4, 1, -0.14],
    [0.44, 0.33, -0.14, 1],
], columns=attributes, index=attributes)
sns.heatmap(diabetes_df.head())
mat.show()