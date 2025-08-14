#PDP for the mesophilic and thermophilic stage

import joblib
from sklearn.inspection import partial_dependence
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import Data
dataset = pd.read_csv('MLSSNH3-meso.csv')

# Separate features (X), target (y), and groups (exp_id)
X = dataset.iloc[:, 0:8].values  # Assuming first 8 columns are features
y = dataset.iloc[:, 8].values  # Assuming 9th column is target
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

# Training the SVR model
model = joblib.load('SVR_model_12.pkl')
feature = ['Day', 'T', 'pH', 'Ammonia', 'Nitrate', 'TN', 'OM', 'AR']
dataset = pd.DataFrame(dataset, columns=feature)

# Define feature pairs to analyze
feature_pairs = [
    ('Day', 'Ammonia'),
    ('AR', 'pH'),
    ('OM', 'Nitrate')
]

# Set global style
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10  # Slightly larger font
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Create figure with adjusted width for 3 subplots
fig = plt.figure(figsize=(18, 4.5))  # Slightly taller for better colorbar
gs_main = fig.add_gridspec(1, 3, wspace=0.4)  # More space between subplots

# Color settings
main_cmap = 'viridis'  # Main colormap
univariate_color = '#E74C3C'  # Attractive red for univariate plots
contour_color = 'white'  # Contour line color
contour_linewidth = 0.5  # Slightly thicker contour lines

for i, (feat1, feat2) in enumerate(feature_pairs):
    feat1_idx = dataset.columns.get_loc(feat1)
    feat2_idx = dataset.columns.get_loc(feat2)

    # Calculate bivariate partial dependence
    pd_results_2d = partial_dependence(model, rescaledX_train,
                                       features=[feat1_idx, feat2_idx],
                                       grid_resolution=35)
    grid_2d = pd_results_2d['grid_values']
    pd_values_2d = pd_results_2d['average']

    # Calculate univariate partial dependence
    pd_results_x = partial_dependence(model, rescaledX_train,
                                      features=[feat1_idx],
                                      grid_resolution=35)
    pd_results_y = partial_dependence(model, rescaledX_train,
                                      features=[feat2_idx],
                                      grid_resolution=35)

    # Unscale feature values
    center1 = scaler.center_[feat1_idx]
    scale1 = scaler.scale_[feat1_idx]
    grid_x = grid_2d[0] * scale1 + center1

    center2 = scaler.center_[feat2_idx]
    scale2 = scaler.scale_[feat2_idx]
    grid_y = grid_2d[1] * scale2 + center2

    # Create subplot grid
    gs_sub = gs_main[i].subgridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4],
                                    wspace=0.1, hspace=0.1)  # Tighter spacing

    # Main plot (bivariate PDP)
    ax_main = fig.add_subplot(gs_sub[1, 0])
    CS = ax_main.contourf(grid_x, grid_y, pd_values_2d[0].T, cmap=main_cmap, levels=20)
    contour = ax_main.contour(grid_x, grid_y, pd_values_2d[0].T,
                              colors=contour_color, levels=CS.levels,
                              linewidths=contour_linewidth)
    ax_main.clabel(contour, inline=True, fontsize=6)
    ax_main.set_xlabel(feat1, fontweight='bold')
    ax_main.set_ylabel(feat2, fontweight='bold')
    ax_main.grid(True, linestyle=':', alpha=0.5)

    # Top plot (univariate PDP for feat1)
    ax_top = fig.add_subplot(gs_sub[0, 0], sharex=ax_main)
    x_values = pd_results_x['grid_values'][0] * scale1 + center1
    ax_top.plot(x_values, pd_results_x['average'][0],
                color=univariate_color, linewidth=1.5)
    ax_top.set_ylabel('PD', fontweight='bold')
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.grid(True, linestyle=':', alpha=0.5)

    # Right plot (univariate PDP for feat2)
    ax_right = fig.add_subplot(gs_sub[1, 1], sharey=ax_main)
    y_values = pd_results_y['grid_values'][0] * scale2 + center2
    ax_right.plot(pd_results_y['average'][0], y_values,
                  color=univariate_color, linewidth=1.5)
    ax_right.set_xlabel('PD', fontweight='bold')
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.grid(True, linestyle=':', alpha=0.5)

    # Add colorbar (wider and better positioned)
    divider = make_axes_locatable(ax_right)
    cax = divider.append_axes("right", size="20%", pad=0.1)  # Wider colorbar
    cbar = plt.colorbar(CS, cax=cax, label='Partial Dependence')
    cbar.set_ticks(CS.levels[::4])
    cbar.outline.set_visible(False)  # Cleaner look

# Adjust layout
plt.show()
plt.savefig("NH3 DOUBLE PDP-12.pdf", format="pdf")




#PDP for the cooling and mature stage
import joblib
from sklearn.inspection import partial_dependence
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import Data
dataset = pd.read_csv('MLSSNH3-cool.csv')

# Separate features (X), target (y), and groups (exp_id)
X = dataset.iloc[:, 0:8].values  # Assuming first 8 columns are features
y = dataset.iloc[:, 8].values  # Assuming 9th column is target
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

# Training the SVR model
model = joblib.load('MLP_model_3.pkl')
feature = ['Day', 'T', 'pH', 'Ammonia', 'Nitrate', 'TN', 'OM', 'AR']
dataset = pd.DataFrame(dataset, columns=feature)

# Define feature pairs to analyze
feature_pairs = [
    ('AR', 'T'),
    ('pH', 'Nitrate'),
    ('OM', 'TN')
]

# Set global style
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10  # Slightly larger font
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Create figure with adjusted width for 3 subplots
fig = plt.figure(figsize=(18, 4.5))  # Slightly taller for better colorbar
gs_main = fig.add_gridspec(1, 3, wspace=0.4)  # More space between subplots

# Color settings
main_cmap = 'viridis'  # Main colormap
univariate_color = '#E74C3C'  # Attractive red for univariate plots
contour_color = 'white'  # Contour line color
contour_linewidth = 0.5  # Slightly thicker contour lines

for i, (feat1, feat2) in enumerate(feature_pairs):
    feat1_idx = dataset.columns.get_loc(feat1)
    feat2_idx = dataset.columns.get_loc(feat2)

    # Calculate bivariate partial dependence
    pd_results_2d = partial_dependence(model, rescaledX_train,
                                       features=[feat1_idx, feat2_idx],
                                       grid_resolution=35)
    grid_2d = pd_results_2d['grid_values']
    pd_values_2d = pd_results_2d['average']

    # Calculate univariate partial dependence
    pd_results_x = partial_dependence(model, rescaledX_train,
                                      features=[feat1_idx],
                                      grid_resolution=35)
    pd_results_y = partial_dependence(model, rescaledX_train,
                                      features=[feat2_idx],
                                      grid_resolution=35)

    # Unscale feature values
    center1 = scaler.center_[feat1_idx]
    scale1 = scaler.scale_[feat1_idx]
    grid_x = grid_2d[0] * scale1 + center1

    center2 = scaler.center_[feat2_idx]
    scale2 = scaler.scale_[feat2_idx]
    grid_y = grid_2d[1] * scale2 + center2

    # Create subplot grid
    gs_sub = gs_main[i].subgridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4],
                                    wspace=0.1, hspace=0.1)  # Tighter spacing

    # Main plot (bivariate PDP)
    ax_main = fig.add_subplot(gs_sub[1, 0])
    CS = ax_main.contourf(grid_x, grid_y, pd_values_2d[0].T, cmap=main_cmap, levels=20)
    contour = ax_main.contour(grid_x, grid_y, pd_values_2d[0].T,
                              colors=contour_color, levels=CS.levels,
                              linewidths=contour_linewidth)
    ax_main.clabel(contour, inline=True, fontsize=6)
    ax_main.set_xlabel(feat1, fontweight='bold')
    ax_main.set_ylabel(feat2, fontweight='bold')
    ax_main.grid(True, linestyle=':', alpha=0.5)

    # Top plot (univariate PDP for feat1)
    ax_top = fig.add_subplot(gs_sub[0, 0], sharex=ax_main)
    x_values = pd_results_x['grid_values'][0] * scale1 + center1
    ax_top.plot(x_values, pd_results_x['average'][0],
                color=univariate_color, linewidth=1.5)
    ax_top.set_ylabel('PD', fontweight='bold')
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.grid(True, linestyle=':', alpha=0.5)

    # Right plot (univariate PDP for feat2)
    ax_right = fig.add_subplot(gs_sub[1, 1], sharey=ax_main)
    y_values = pd_results_y['grid_values'][0] * scale2 + center2
    ax_right.plot(pd_results_y['average'][0], y_values,
                  color=univariate_color, linewidth=1.5)
    ax_right.set_xlabel('PD', fontweight='bold')
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.grid(True, linestyle=':', alpha=0.5)

    # Add colorbar (wider and better positioned)
    divider = make_axes_locatable(ax_right)
    cax = divider.append_axes("right", size="20%", pad=0.1)  # Wider colorbar
    cbar = plt.colorbar(CS, cax=cax, label='Partial Dependence')
    cbar.set_ticks(CS.levels[::4])
    cbar.outline.set_visible(False)  # Cleaner look

# Adjust layout
plt.show()
plt.savefig("NH3 DOUBLE PDP-3.pdf", format="pdf")


#Interaction analysis for the mesophilic and thermophilic stage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
import joblib
from mpl_toolkits.axes_grid1 import make_axes_locatable


dataset = pd.read_csv('MLSSNH3-meso.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values
groups = dataset['exp_id'].values


group_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
train_idx, test_idx = next(group_split.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)


model = joblib.load('SVR_model_12.pkl')
feature_names = ['Day', 'T', 'pH', 'Ammonia', 'Nitrate', 'TN', 'OM', 'AR']


feature_pairs = [
    ('Day', 'Ammonia'),
    ('AR', 'pH'),
    ('OM', 'Nitrate')
]


mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '0.3'
})



def calculate_interaction(feat1, feat2):

    feat1_idx = feature_names.index(feat1)
    feat2_idx = feature_names.index(feat2)


    pd_joint = partial_dependence(model, rescaledX_train, [feat1_idx, feat2_idx],
                                  grid_resolution=35, kind="average")
    pd_feat1 = partial_dependence(model, rescaledX_train, [feat1_idx],
                                  grid_resolution=35, kind="average")
    pd_feat2 = partial_dependence(model, rescaledX_train, [feat2_idx],
                                  grid_resolution=35, kind="average")


    grid_x = pd_joint['grid_values'][0] * scaler.scale_[feat1_idx] + scaler.center_[feat1_idx]
    grid_y = pd_joint['grid_values'][1] * scaler.scale_[feat2_idx] + scaler.center_[feat2_idx]


    interaction = pd_joint['average'][0] - (
            np.tile(pd_feat1['average'][0], (len(grid_y), 1)).T +
            np.tile(pd_feat2['average'][0], (len(grid_x), 1))
    )

    return grid_x, grid_y, interaction



fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
plt.subplots_adjust(wspace=0.4, bottom=0.15, top=0.9)


cmap = plt.get_cmap('RdBu_r').copy()
cmap.set_bad(color='0.95')

for i, (feat1, feat2) in enumerate(feature_pairs):
    grid_x, grid_y, interaction = calculate_interaction(feat1, feat2)
    ax = axes[i]


    abs_max = np.max(np.abs(interaction))
    vmin, vmax = -abs_max, abs_max


    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    interaction_clean = np.ma.masked_invalid(interaction.T)


    im = ax.contourf(
        X_grid, Y_grid, interaction_clean,
        cmap=cmap, levels=np.linspace(vmin, vmax, 21),
        vmin=vmin, vmax=vmax,
        alpha=0.9
    )


    contours = ax.contour(
        X_grid, Y_grid, interaction_clean,
        colors=['0.3'], linewidths=0.6, levels=11,
        linestyles='-', alpha=0.7
    )


    zero_contour = ax.contour(
        X_grid, Y_grid, interaction_clean,
        colors=['k'], linewidths=1.2, levels=[0],
        linestyles='--'
    )


    ax.set_xlabel(feat1, fontweight='bold', labelpad=8, fontsize=11)
    ax.set_ylabel(feat2, fontweight='bold', labelpad=8, fontsize=11)
    ax.set_title(f'{feat1} × {feat2} Interaction', pad=12,
                 fontweight='bold', fontsize=12)


    ax.grid(True, linestyle=':', color='0.8', alpha=0.5)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Interaction Effect', fontweight='bold', fontsize=10)
    cbar.ax.tick_params(labelsize=8)


    cbar.outline.set_linewidth(0.5)
    cbar.ax.yaxis.set_ticks_position('right')


    ax.tick_params(axis='both', which='major', labelsize=9)


fig.suptitle('Feature Interaction Analysis', y=0.98,
             fontsize=14, fontweight='bold')

plt.show()
plt.savefig('Interaction_Effects_12.pdf', bbox_inches='tight', dpi=300)



#Interaction analysis for the cooling and mature stage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.inspection import partial_dependence
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
import joblib
from mpl_toolkits.axes_grid1 import make_axes_locatable


dataset = pd.read_csv('MLSSNH3-cool.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values
groups = dataset['exp_id'].values


group_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
train_idx, test_idx = next(group_split.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)


model = joblib.load('MLP_model_3.pkl')
feature_names = ['Day', 'T', 'pH', 'Ammonia', 'Nitrate', 'TN', 'OM', 'AR']


feature_pairs = [
    ('AR', 'T'),
    ('pH', 'Nitrate'),
    ('OM', 'TN')
]


mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '0.3'
})



def calculate_interaction(feat1, feat2):

    feat1_idx = feature_names.index(feat1)
    feat2_idx = feature_names.index(feat2)


    pd_joint = partial_dependence(model, rescaledX_train, [feat1_idx, feat2_idx],
                                  grid_resolution=35, kind="average")
    pd_feat1 = partial_dependence(model, rescaledX_train, [feat1_idx],
                                  grid_resolution=35, kind="average")
    pd_feat2 = partial_dependence(model, rescaledX_train, [feat2_idx],
                                  grid_resolution=35, kind="average")


    grid_x = pd_joint['grid_values'][0] * scaler.scale_[feat1_idx] + scaler.center_[feat1_idx]
    grid_y = pd_joint['grid_values'][1] * scaler.scale_[feat2_idx] + scaler.center_[feat2_idx]


    interaction = pd_joint['average'][0] - (
            np.tile(pd_feat1['average'][0], (len(grid_y), 1)).T +
            np.tile(pd_feat2['average'][0], (len(grid_x), 1))
    )

    return grid_x, grid_y, interaction



fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
plt.subplots_adjust(wspace=0.4, bottom=0.15, top=0.9)


cmap = plt.get_cmap('RdBu_r').copy()
cmap.set_bad(color='0.95')

for i, (feat1, feat2) in enumerate(feature_pairs):
    grid_x, grid_y, interaction = calculate_interaction(feat1, feat2)
    ax = axes[i]


    abs_max = np.max(np.abs(interaction))
    vmin, vmax = -abs_max, abs_max


    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    interaction_clean = np.ma.masked_invalid(interaction.T)


    im = ax.contourf(
        X_grid, Y_grid, interaction_clean,
        cmap=cmap, levels=np.linspace(vmin, vmax, 21),
        vmin=vmin, vmax=vmax,
        alpha=0.9
    )


    contours = ax.contour(
        X_grid, Y_grid, interaction_clean,
        colors=['0.3'], linewidths=0.6, levels=11,
        linestyles='-', alpha=0.7
    )


    zero_contour = ax.contour(
        X_grid, Y_grid, interaction_clean,
        colors=['k'], linewidths=1.2, levels=[0],
        linestyles='--'
    )


    ax.set_xlabel(feat1, fontweight='bold', labelpad=8, fontsize=11)
    ax.set_ylabel(feat2, fontweight='bold', labelpad=8, fontsize=11)
    ax.set_title(f'{feat1} × {feat2} Interaction', pad=12,
                 fontweight='bold', fontsize=12)


    ax.grid(True, linestyle=':', color='0.8', alpha=0.5)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Interaction Effect', fontweight='bold', fontsize=10)
    cbar.ax.tick_params(labelsize=8)


    cbar.outline.set_linewidth(0.5)
    cbar.ax.yaxis.set_ticks_position('right')


    ax.tick_params(axis='both', which='major', labelsize=9)


fig.suptitle('Feature Interaction Analysis', y=0.98,
             fontsize=14, fontweight='bold')


plt.show()
plt.savefig('Interaction_Effects_3.pdf', bbox_inches='tight', dpi=300)

