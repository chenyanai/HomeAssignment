import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

full_df = pd.read_csv('listings_dataset.csv', index_col=0)

temp = full_df.isnull().mean()
temp[temp > 0].plot(kind='barh')
plt.title('Normalized missing data count per column')
plt.xlim(0, 1)

"""

The filed 'context' has only 'search' in it - we might be able to drop it as it will not help with out predictions
The following columns are not categorical: 
1. 'gig_title' - it has free text in it
2. 'listing_id' - ID of the listings, IDs are not categorical
3. 'gig_id' - ID of the gigs, IDs are not categorical
4. 'search_query' - free text as well

We can add:
categorical_columns = list(set(categorical_columns) - set(['search_query', 'gig_title', 'listing_id', 'gig_id']))

user_timezone - can be preprocessed better, we can convert it to zone (America, Europe...) and to specific time in it.
Typo - 'is_seller_onlie' should be 'is_seller_online', need to be fixed in the orig df

"""

numeric_columns = ['position', 'gig_price', 'gig_avg_rating', 'gig_rated_orders', 'cnt_helpful_reviews_last_year', 'avg_review_length_last_3_months',
       'avg_position_shown_last_60d', 'previous_order_amount']
dates_columns = ['created_at', 'user_reg_date', 'previous_order_date']
binary_columns = ['is_click', 'is_filtered', 'is_user_buyer','is_seller_onlie']
categorical_columns = list(set(full_df.columns) - set(numeric_columns) - set(dates_columns) - set(binary_columns))

# Empty values are floats, we need to use fillna('') to fill them with strings.
# We also might consider changing the empty strings in the continent column to something else
full_df['continent'] = full_df['user_timezone'].apply(lambda x: x.split('/')[0].lower())
vc = full_df.continent.value_counts()
vc.plot(kind='bar')

# We need to add 'continent' to categorical columns to make sure it will be processed accordingly

temp = full_df.groupby('gig_sc_id').agg(
    num_clicks=('is_click', 'sum'),
    group_size=('gig_sc_id', 'size')
)
temp['sc_id_ctr'] = temp.num_clicks / temp.group_size
full_df = full_df.join(temp, on='gig_sc_id').drop(['num_clicks','group_size'], axis=1)

for col in numeric_columns:
    bins=1000
    full_df[col].plot(kind='hist', bins=bins, figsize=(13, 5), color='b', alpha=0.5, density=True)
    plt.title(f'{col} histogram')
    plt.show()

layout = np.arange(np.array(binary_columns).shape[0]).reshape(-1, 2)
fig, axs = plt.subplot_mosaic(layout, figsize=(10, 10))
for i, col in enumerate(binary_columns):
    ser = full_df[col].value_counts()
    ser.plot(kind='pie', autopct='%1.1f%%', title=col, ylabel='', ax=axs[i])
plt.show();

def add_time_features(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['user_reg_date'] = pd.to_datetime(df['user_reg_date'])
    df['previous_order_date'] = pd.to_datetime(df['previous_order_date'])
       # As we already converted the values to datetime we should use the attributes of this object instead of calcualting it ourself.
       # For example, just use the 'month' attribute to calcualte the months difference
       # Week in month might not be correct as well, as in a case were the month starts in the middle of the week the whole calculation might be incorrect
    df['hour'] = (df.created_at.dt.hour + (df.created_at.dt.minute / 60)).round(1)
    df['day'] = df.created_at.dt.day_name().apply(lambda x: x.lower()[:3])
    df['week_in_month'] = df.created_at.dt.day // 7
    df['months_since_reg'] = ((df.created_at - df.user_reg_date).dt.days / 30.2).round(1)
       # This code will not account for different days, we might also have scenarios were we will have negative values which should not be possible in this column
    df['hour_diff_from_last_purchase'] = (df.created_at.dt.hour - df.previous_order_date.dt.hour).round(1)
    df['days_since_last_purchase'] = (df.created_at - df.previous_order_date).dt.days
       # is_weekend is not defined
    df['is_weekend'] = df.apply(is_weekend, axis=1)
    df = df.drop(['created_at','previous_order_date','user_reg_date'], axis=1)
    return df

full_df = add_time_features(full_df)

"""
The threshold is set as constant for all the columns without accounting for each column's distribution.
For example, for the columns 'previous_order_sc_id' and 'gig_sc_id' the threshold is too high and this preprocessing step will make us lose information.
In the column 'operating_system' we just replace the value 'linux' to other, which can hurt explainablitiy.
"""
def rare_cats_to_other(ser, thresh=0.05):
    value_counts = ser.value_counts(normalize=True)
    values_to_replace = value_counts[value_counts < thresh].index
    return ser.replace(values_to_replace, 'other')

for col in categorical_columns:
    full_df[col] = rare_cats_to_other(full_df[col])
    full_df[col].fillna('other', inplace=True)

for col in numeric_columns:
    full_df[col].fillna(0, inplace=True)

def compare_sc_id(row):
    if pd.notna(row['gig_sc_id']) and pd.notna(row['previous_order_sc_id']):
        return int(row['gig_sc_id'] == row['previous_order_sc_id'])

def compare_order_amounts(row):
    if pd.notna(row['gig_price']) and pd.notna(row['previous_order_amount']) and row['previous_order_amount'] > 0:
        return row['gig_price'] / row['previous_order_amount']

def add_gig_features(df):
    df['price_change_from_last_purchase'] = df.apply(compare_order_amounts, axis=1)
    df['is_same_sc_as_last_purchase'] = df.apply(compare_sc_id, axis=1)
       # position_diff_from_avg - we might want to normalize it to different categories and gig types
    df['position_diff_from_avg'] = df.position - df.avg_position_shown_last_60d
    return df

full_df = add_gig_features(full_df)

x_train, x_test, y_train, y_test = train_test_split(full_df.drop('is_click', axis=1), full_df['is_click'], test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

for row in full_df.dtypes.reset_index().iterrows():
    if (row[1][0] == 'object'):
        full_df[row[1]['index']] = full_df[row[1]['index']].astype('category')

# learning_rate - is a bit high, we might want to lower it
model_params = dict(
    learning_rate=0.2,
    n_estimators=200, 
    max_depth=7,
    colsample_bytree=0.9,
    colsample_bylevel=0.9,
    colsample_bynode=0.9,
    importance_type='cover',
    scale_pos_weight=2,
)

"""
This code does not hold a test set which is not part of the KFold training.
For each fold we take all of the data and split it.
Therefore, a gig which was part of the test set in the a previous fold can be part of the training set in the current one.
We might face data leakage and the evaluation of our model will be incorrect.
To fix it, we need to split the data to test and train, and then split the train to validation and train.
For our final estimation of our model we would evaluate it on the test set.
"""

feature_importance = []
cv_metrics = []
n_folds = 6
full_df = full_df.reset_index(drop=True)

splitter = KFold(n_splits=n_folds, shuffle=True, random_state=13)
for i, (train_indices, test_indices) in enumerate(splitter.split(full_df)):
    print(f'Fold {i}')
    x_train = full_df.loc[train_indices, :].drop('is_click', axis=1)
    y_train = full_df['is_click'][train_indices].astype(int)
    x_test = full_df.loc[test_indices, :].drop('is_click', axis=1)
    y_test = full_df['is_click'][test_indices].astype(int)
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=13)
    
    model = xgb.XGBClassifier(eval_metric='logloss', enable_categorical=True, **model_params)
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_val, y_val)], verbose=50)
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    feature_importance.append(dict(zip(model.feature_names_in_, model.feature_importances_)))
    cv_metrics.append({
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'balanced_accuracy': metrics.balanced_accuracy_score(y_test, y_pred),
        'f1': metrics.f1_score(y_test, y_pred),
        'auc_roc': metrics.roc_auc_score(y_test, y_prob),
    })
cv_df = pd.DataFrame(cv_metrics)
display(cv_df)
cv_df.plot()
plt.xticks(range(n_folds), range(1, n_folds+1))
plt.grid()
plt.show();

pd.DataFrame.from_dict(feature_importance).mean().sort_values().plot(kind='barh', figsize=(8, 10))
plt.title('Mean feature importance (cover)')
plt.show();

means = cv_df.mean()
stds = cv_df.std()
plt.barh(means.index, means.values, xerr=stds.values, capsize=5, alpha=0.5)
plt.title('Mean CV classification metrics')
plt.ylabel('Score')
plt.grid(axis='x');

"""
After training the model I got the following results:

	accuracy	balanced_accuracy	f1	auc_roc
0	0.894955	0.558373	0.203616	0.661120
1	0.892947	0.552500	0.189934	0.673858
2	0.901857	0.559144	0.205285	0.665322
3	0.902234	0.553254	0.189386	0.665385
4	0.896824	0.554392	0.195695	0.671589
5	0.902096	0.552446	0.187500	0.676548

Inspecting the results I could understand the following key points:
1. Our data is imbalanced as can be seen by the difference in accuracy score to the scores of balanced_accuracy and f1 metrics.
2. Building upon the previous point, our model is probably overfitting for the majority class

full_df stats confirm it:

is_click
0    43428
1     4378
Name: count, dtype: int64
Selection deleted

"""
