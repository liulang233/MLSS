#shapley for the whole stage
import shap
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colormaps
import joblib
import numpy as np

# Import Data
dataset = pd.read_csv('MLSSNH3.csv')

# Separate features (X), target (y), and groups (exp_id)
X = dataset.iloc[:, 0:8].values  # Assuming first 8 columns are features
y = dataset.iloc[:, 8].values    # Assuming 9th column is target
groups = dataset['exp_id'].values  # Assuming 'exp_id' is the column name for experimental IDs

# Initialize GroupShuffleSplit for train-test split
group_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
train_idx, test_idx = next(group_split.split(X, y, groups=groups))

# Split data
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Scale features
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)
feature = ['Day','T', 'pH', 'Ammonia', 'Nitrate', 'TN',  'OM',   'AR']
rescaledX_train =pd.DataFrame(rescaledX_train,columns=feature)
rescaledX_test =pd.DataFrame(rescaledX_test,columns=feature)
full_x = pd.concat([rescaledX_train, rescaledX_test])

# Training the SVR model
model = joblib.load('MLP_model.pkl')
# Define the svr_predict function
def svr_predict(X):
    return model.predict(X)
# Use the shap library to interpret the model
explainer = shap.KernelExplainer(svr_predict, full_x)
shapva = explainer(full_x)
shap_values =shapva.values


# Set the global font for matplotlib
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Create a figure and axes object
fig, ax = plt.subplots()
# Visualize Feature Importance
shap.summary_plot(shap_values, full_x,cmap = colormaps.get_cmap('viridis'), plot_size=(10.5, 8))
ax.spines['top'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.show()
# Save graphics
plt.savefig('total shap_mlp.pdf', format='pdf', bbox_inches='tight')

# Calculate the average absolute SHAP value for each feature
feature_importance = np.average(abs(shap_values),0)
# Output feature importance
for feature_index, importance in enumerate(feature_importance):
    print(f"Feature {feature_index}: Importance {importance}")


#shapley for the mesophilic and thermophilic stage
import shap
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colormaps
import joblib
import numpy as np
# Import Data
dataset = pd.read_csv('MLSSNH3-meso.csv')

# Separate features (X), target (y), and groups (exp_id)
X = dataset.iloc[:, 0:8].values  # Assuming first 8 columns are features
y = dataset.iloc[:, 8].values    # Assuming 9th column is target
groups = dataset['exp_id'].values  # Assuming 'exp_id' is the column name for experimental IDs

# Initialize GroupShuffleSplit for train-test split
group_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
train_idx, test_idx = next(group_split.split(X, y, groups=groups))

# Split data
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Scale features
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)
feature = ['Day','T', 'pH', 'Ammonia', 'Nitrate', 'TN',  'OM',   'AR']
rescaledX_train =pd.DataFrame(rescaledX_train,columns=feature)
rescaledX_test =pd.DataFrame(rescaledX_test,columns=feature)
full_x = pd.concat([rescaledX_train, rescaledX_test])

# Training the SVR model
model = joblib.load('SVR_model_12.pkl')
# Define the svr_predict function
def svr_predict(X):
    return model.predict(X)
# Use the shap library to interpret the model
explainer = shap.KernelExplainer(svr_predict, full_x)
shapva = explainer(full_x)
shap_values =shapva.values


# Set the global font for matplotlib
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Create a figure and axes object
fig, ax = plt.subplots()
# Visualize Feature Importance
shap.summary_plot(shap_values, full_x,cmap = colormaps.get_cmap('viridis'), plot_size=(10.5, 8))
ax.spines['top'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.show()
# Save graphics
plt.savefig('12 shap_et.pdf', format='pdf', bbox_inches='tight')



# Calculate the average absolute SHAP value for each feature
feature_importance = np.average(abs(shap_values),0)
# Output feature importance
for feature_index, importance in enumerate(feature_importance):
    print(f"Feature {feature_index}: Importance {importance}")




#shapley for the cooling and mature stage
import shap
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colormaps
import joblib
import numpy as np
# Import Data
dataset = pd.read_csv('MLSSNH3-cool.csv')


# Separate features (X), target (y), and groups (exp_id)
X = dataset.iloc[:, 0:8].values  # Assuming first 8 columns are features
y = dataset.iloc[:, 8].values    # Assuming 9th column is target
groups = dataset['exp_id'].values  # Assuming 'exp_id' is the column name for experimental IDs

# Initialize GroupShuffleSplit for train-test split
group_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
train_idx, test_idx = next(group_split.split(X, y, groups=groups))

# Split data
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Scale features
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)
feature = ['Day','T', 'pH', 'Ammonia', 'Nitrate', 'TN',  'OM',   'AR']
rescaledX_train =pd.DataFrame(rescaledX_train,columns=feature)
rescaledX_test =pd.DataFrame(rescaledX_test,columns=feature)
full_x = pd.concat([rescaledX_train, rescaledX_test])

# Training the SVR model
model = joblib.load('MLP_model_3.pkl')
# Define the svr_predict function
def svr_predict(X):
    return model.predict(X)
# Use the shap library to interpret the model
explainer = shap.KernelExplainer(svr_predict, full_x)
shapva = explainer(full_x)
shap_values =shapva.values


# Set the global font for matplotlib
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Create a figure and axes object
fig, ax = plt.subplots()
# Visualize Feature Importance
shap.summary_plot(shap_values, full_x,cmap = colormaps.get_cmap('viridis'), plot_size=(10.5, 8))
ax.spines['top'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.show()
# Save graphics
plt.savefig('3 shap_mlp.pdf', format='pdf', bbox_inches='tight')



# Calculate the average absolute SHAP value for each feature
feature_importance = np.average(abs(shap_values),0)
# Output feature importance
for feature_index, importance in enumerate(feature_importance):
    print(f"Feature {feature_index}: Importance {importance}")