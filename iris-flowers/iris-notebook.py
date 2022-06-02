# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
random_state = 42
# np.random.seed = random_state
rng = np.random.default_rng(random_state)

# %%
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
iris

# %%
iris.info()

# %% [markdown]
# missing values

# %%
iris.isna().sum().sum()

# %%
# split the dataset into train and test
from sklearn.model_selection import train_test_split

train, test = train_test_split(iris, test_size=0.2, stratify=iris['species'], random_state=random_state)

# %%
y_col = 'species'

# %% [markdown]
# we don't look at testset

# %%
train.describe()

# %% [markdown]
# but, it's okay to look at the population distribution of the entire dataset(including testset)

# %%
# target distribution in percentage
_, axs = plt.subplots(1, 3, figsize=(18, 8))
train[y_col].value_counts(normalize=True).plot(kind="bar", title="distribution of Popularity in train", ax=axs[0])
test[y_col].value_counts(normalize=True).plot(kind="bar", title="distribution of Popularity in test", ax=axs[1])
iris[y_col].value_counts(normalize=True).plot(kind="bar", title="distribution of Popularity in train-test", ax=axs[2])
plt.show()

# %% [markdown]
# population is normally distributed: no target imbalance

# %% [markdown]
# # EDA

# %%
# Use Pairplot to understand relationships among paramaters
_ = sns.pairplot(train, kind='kde', markers='*')
plt.show()

# %% [markdown]
# can't see any significant outliers that would skew the distributions

# %%
# Use Pairplot to understand relationships among paramaters
_ = sns.pairplot(train, hue='species', markers='*')
plt.show()

# %% [markdown]
# * Because Patel length and Petal width have saperable distributions for different iris species, these 2 variables would have higher influence over classifying the species. 
# * We can also observe from the scatterplots of `petal_length` x `petal_width`, `petal_width` x `sepal_width` and `petal_width` x `sepal_length`, that the species clusters are clearly distinguishable and can easily be saperated a line. 
# * Hence, a sinple linear model(s) would be best suited for this type of classification here, without requiring of any feature transformations as the features are already distinguishable.
# * The setosa species is the most easily distinguishable because of its small feature size.

# %%
def plot_heatmap(matrix, size=10, title=None):
    labels = matrix.applymap(lambda v: str(np.round(v, decimals=2)) if 1 > np.abs(v) >= .5 else '')
    mask = np.zeros_like(matrix)
    mask[np.tril_indices_from(mask)] = True
    sns.clustermap(matrix, center=0, cmap='vlag', linewidth=1, annot=labels, fmt='')
    plt.title(title)
    plt.show()

# %%
# plot heatmap of correlation between parameters and show only the upper triangle
plot_heatmap(train.corr(), title='Correlation between parameters')

# %% [markdown]
# * if we look further we can see that `petal_width` is highly +vely correlated to `petal_length` and `sepal_length`. Similarly, `petal_length` and `sepal_length` are also sognificantly correlated.
# * Also, `sepal_width` is slightly -vely correlated with other variables.

# %%
# compute covariance matrix
cov_matrix = np.cov(train.drop('species', axis=1).T)
cov_matrix = pd.DataFrame(cov_matrix, columns=train.columns[:-1], index=train.columns[:-1])
plot_heatmap(cov_matrix, title='Covariance matrix')

# %% [markdown]
# A strong correlation is present between petal width and petal length.

# %% [markdown]
# # Modeling
# 
# we earlier found out that the dataset is not too complex and simple models(most probably linear models) would be better suited for this problem.

# %%
X_train, X_test, y_train, y_test = train_test_split(iris.drop(y_col, axis=1), iris[y_col], test_size=0.2, random_state=random_state, stratify=iris[y_col])

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {
    'C': [0.1, 1, 10, 100], 
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    # 'l1_ratio': [0, 0.1, 0.5, 0.9, 1]
    }

logreg = LogisticRegression(tol=1e-6, fit_intercept=True, n_jobs=-1, verbose=0)
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# %%
grid.best_params_

# %%
grid.best_score_

# %%
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_recall_curve

print(classification_report(y_test, grid.predict(X_test)))

# %%
def plot_confusion_matrix(y_true, y_pred, classes, title, normalize='true'):
    """
    Plot the confusion matrix of the model.
    Parameters
    ----------
    y_true: np.array
        The true target of the dataset.
    y_pred: np.array
        The predicted target of the dataset.
    classes: np.array
        The classes of the dataset.
    title: str
        The title of the plot.
    regr: bool, optional
        If True, the confusion matrix is computed as the mean of the errors (MSE, i.e. Mean Squared Errors).
        Otherwise, the confusion matrix is computed as the mean of the accuracies (i.e. Accuracy).
        By default, it is True.
    cmap: matplotlib.colors.Colormap, optional
        The colormap used to plot the confusion matrix.
    Returns
    -------
    None
    """

    # Compute confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=classes,
        # cmap=cmap,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# %%
plot_confusion_matrix(y_test, grid.predict(X_test), classes=iris[y_col].unique(), title='Confusion matrix')

# %% [markdown]
# I don't think any explanation is needed at this point.


