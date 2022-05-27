import numpy as np
import pytorch_widedeep as wd
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import Accuracy
import pandas as pd
import torch
# from sklearn.metrics import log_loss, roc_auc_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
# from deepctr_torch.models import *
# This is a package in preview.
from azureml.opendatasets import Diabetes

diabetes = Diabetes.get_tabular_dataset()
diabetes_df = diabetes.to_pandas_dataframe() # data

# print(diabetes_df['AGE'][0])
# train, test = train_test_split(diabetes_df, test_size=0.2)

wide_cols = [
    "AGE",
    "SEX",
    "BMI",
    "BP",
    "S1",
    "Y",
]
crossed_cols = [("AGE", "BMI"), ("AGE", "SEX")]
wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
X_wide = wide_preprocessor.fit_transform(diabetes_df)
print(X_wide)

cat_embed_cols = [
    ("AGE"),
    ("SEX"),
    ("BMI"),
    ("BP"),
]
continuous_cols = ["S1", "S2"]
target = "Y"
target = diabetes_df[target].values

tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols
)
X_tab = tab_preprocessor.fit_transform(diabetes_df)

print(X_tab)

wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=continuous_cols,
)
model = WideDeep(wide=wide, deeptabular=tab_mlp)

trainer = Trainer(model, objective="binary", metrics=[Accuracy])
trainer.fit(
    X_wide=X_wide,
    X_tab=X_tab,
    target=target,
    n_epochs=5,
    # batch_size=256,
)


'''
import numpy as np
import pandas as pd
import pytorch_widedeep as wd

from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import WidePreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import Accuracy

df = load_adult(as_frame=True)
# print(df.head())

wide_cols = [
    "education",
    "relationship",
    "workclass",
    "occupation",
    "native-country",
    "gender",
]
crossed_cols = [("education", "occupation"), ("native-country", "occupation")]

wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
X_wide = wide_preprocessor.fit_transform(df)
# From here on, any new observation can be prepared by simply running `.transform`
# new_X_wide = wide_preprocessor.transform(new_df)
# print(X_wide)

# print(wide_preprocessor.inverse_transform(X_wide[:1]))

from pytorch_widedeep.preprocessing import TabPreprocessor
# cat_embed_cols = [(column_name, embed_dim), ...]
cat_embed_cols = [
    ("education"),
    ("relationship"),
    ("workclass"),
    ("occupation"),
    ("native-country"),
]
continuous_cols = ["age", "hours-per-week"]
target = "income"
target = df[target].values

tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols
)
X_tab = tab_preprocessor.fit_transform(df)

# print(X_tab)

wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=continuous_cols,
)
model = WideDeep(wide=wide, deeptabular=tab_mlp)


trainer = Trainer(model, objective="binary", metrics=[Accuracy])
trainer.fit(
    X_wide=X_wide,
    X_tab=X_tab,
    target=target,
    n_epochs=5,
    # batch_size=256,
)
'''