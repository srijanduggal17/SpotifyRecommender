########## Imports ##########
# Data Manipulation and Stats
import pandas as pd
import numpy as np
import time
from scipy.stats import t

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Model Selection and Metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# Plotting and Writing to File
from matplotlib import pyplot as plt
import seaborn as sns
from decimal import Decimal
import os
import shutil


########## Load and Inspect Data ##########
raw_data = pd.read_csv('dataset.csv')
# print('Raw Data Shape: {}'.format(raw_data.shape))
# print(raw_data.head())
print('Dataset Breakdown:')
print('Class - saved: {:.2f}%'.format(100*sum(raw_data.label=='saved')/raw_data.shape[0]))
print('Class - unsaved: {:.2f}%'.format(100*sum(raw_data.label=='unsaved')/raw_data.shape[0]))
raw_data = raw_data.astype({'key': 'float64', 'time_signature':'float64'}, copy=True)

x = raw_data.drop(['label'], axis=1)
y = raw_data.label
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=252)
# print('Training data example:')
# print(X_train.head())
# print(y_train.head())


dir = '_output'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

########## Scale Features ##########
scaler = StandardScaler()

cols_to_scale = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
'acousticness','instrumentalness','liveness','valence','tempo','time_signature']

X_train_scaled = X_train.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled = X_test.copy()
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

X_train_scaled = X_train_scaled.to_numpy()
y_train = y_train.to_numpy()
X_test_scaled = X_test_scaled.to_numpy()
y_test = y_test.to_numpy()
# print('Testing and Training Dataset shapes:')
# print('X Train: {}'.format(X_train_scaled.shape))
# print('Y Train: {}'.format(y_train.shape))
# print('X Test: {}'.format(X_test_scaled.shape))
# print('Y Test: {}'.format(y_test.shape))


########## Hyperparameter Tuning ##########
scorer = make_scorer(precision_score, pos_label='saved')

def calculateCI(search, results):
    results = results.copy()
    results['validation_lower_ci'] = results.apply(lambda row: t.interval(0.95, search.n_splits_ - 1, loc=row.mean_test_score, scale=row.std_test_score)[0], axis=1)
    results['validation_upper_ci'] = results.apply(lambda row: t.interval(0.95, search.n_splits_ - 1, loc=row.mean_test_score, scale=row.std_test_score)[1], axis=1)
    results['train_lower_ci'] = results.apply(lambda row: t.interval(0.95, search.n_splits_ - 1, loc=row.mean_train_score, scale=row.std_train_score)[0], axis=1)
    results['train_upper_ci'] = results.apply(lambda row: t.interval(0.95, search.n_splits_ - 1, loc=row.mean_train_score, scale=row.std_train_score)[1], axis=1)
    results.drop(['std_fit_time', 'mean_score_time', 'std_score_time', 'params',
       'split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'rank_test_score', 'split0_train_score',
       'split1_train_score', 'split2_train_score', 'split3_train_score',
       'split4_train_score'], axis=1, inplace=True)
    return results

##### Random Forests with Boosting #####
def plot_adaboost_search_results(search, results, experiment):
    results = calculateCI(search, results)
    
    learning_rates = np.unique(results.param_learning_rate.values)

    os.mkdir(dir+experiment)
    for learning_rate in learning_rates:
        df_current = None
        df_current = results.copy()
        df_current = df_current[df_current.param_learning_rate == learning_rate]
        plt.figure()
        plt.plot(df_current.param_n_estimators, df_current.mean_train_score)
        plt.plot(df_current.param_n_estimators, df_current.mean_test_score)
        df_current = df_current.astype({'param_n_estimators': 'float64'}, copy=True)
        title = 'Learning Rate of {}'.format(learning_rate)
        plt.title(title)
        plt.xlabel('n_estimators')
        plt.ylabel('Precision')
        plt.ylim(top=1.0)
        plt.ylim(bottom=0.0)
        plt.fill_between(df_current.param_n_estimators, df_current.train_lower_ci, df_current.train_upper_ci, alpha=0.2)
        plt.fill_between(df_current.param_n_estimators, df_current.validation_lower_ci, df_current.validation_upper_ci, alpha=0.2)
        plt.legend(['Train', 'Validation'])
        filetitle = dir+experiment+'/Learning Rate of {:.2E}.png'.format(Decimal(learning_rate))
        plt.savefig(filetitle)
    return results
    
### Experiment 1 ###
exp_1_param_grid = [{'n_estimators': np.arange(10, 360, 10),
               'learning_rate': [1, .1, .01, .001, .0001]}
             ]

exp_1_start_time = time.time()
exp_1_search = GridSearchCV(AdaBoostClassifier(random_state=62), param_grid=exp_1_param_grid, scoring=scorer, cv=5, return_train_score=True, n_jobs=-1)
exp_1_search.fit(X_train_scaled, y_train)
exp_1_search_results = pd.DataFrame(exp_1_search.cv_results_)
exp_1_results = plot_adaboost_search_results(exp_1_search, exp_1_search_results, '/Experiment 1')
print('Experiment 1 Time: {:6.4f} seconds'.format(time.time() - exp_1_start_time))
# print('Experiment 1 Results')
# print(exp_1_results)

### Experiment 2 ###
exp_2_param_grid = [{'n_estimators': np.arange(10, 360, 10),
               'learning_rate': [.03, .05, .07, .09, 0.3, 0.5, 0.7, 0.9]}
             ]

exp_2_start_time = time.time()
exp_2_search = GridSearchCV(AdaBoostClassifier(random_state=62), param_grid=exp_2_param_grid, scoring=scorer, cv=5, return_train_score=True, n_jobs=-1)
exp_2_search.fit(X_train_scaled, y_train)
exp_2_search_results = pd.DataFrame(exp_2_search.cv_results_)
exp_2_results = plot_adaboost_search_results(exp_2_search, exp_2_search_results, '/Experiment 2')
print('Experiment 2 Time: {:6.4f} seconds'.format(time.time() - exp_2_start_time))
# print('Experiment 2 Results')
# print(exp_2_results)

### Hyperparameter Choice ###
rfboost_criteria = (exp_2_results.param_learning_rate == 0.7) & (exp_2_results.param_n_estimators == 10)
chosen_rfboost_precision = exp_2_results[rfboost_criteria].mean_test_score.values[0]
chosen_rfboost_tolerance = exp_2_results[rfboost_criteria].validation_upper_ci.values[0] - chosen_rfboost_precision
print('The chosen Random Forest with Boosting model has 10 estimators and a learning rate of 0.7')
print('This model has a precision of {:.2f} +/- {:.2f}'.format(chosen_rfboost_precision, chosen_rfboost_tolerance))


##### Support Vector Machines #####
def plot_svm_search_results(search, results):
    results = calculateCI(search, results)
    
    os.mkdir(dir+'/Experiment 3')

    linear_results = results[results.param_kernel == 'linear']

    fig = plt.figure()
    plt.plot(linear_results.param_C, linear_results.mean_train_score)
    plt.plot(linear_results.param_C, linear_results.mean_test_score)
    linear_results = linear_results.astype({'param_C': 'float64'}, copy=True)
    plt.title('Linear Kernel')
    plt.xlabel('C')
    plt.ylabel('Precision')
    plt.xscale('log')
    plt.ylim(top=1.0)
    plt.ylim(bottom=0.0)
    plt.fill_between(linear_results.param_C, linear_results.train_lower_ci, linear_results.train_upper_ci, alpha=0.2)
    plt.fill_between(linear_results.param_C, linear_results.validation_lower_ci, linear_results.validation_upper_ci, alpha=0.2)
    plt.legend(['Train', 'Validation'])

    filetitle = dir+'/Experiment 3/Linear Kernel.png'
    plt.savefig(filetitle)

    rbf_results = results[results.param_kernel == 'rbf']
    rbf_results = rbf_results.astype({'param_C': 'float64',}, copy=True)
    
    sigmoid_results = results[results.param_kernel == 'sigmoid']
    sigmoid_results = sigmoid_results.astype({'param_C': 'float64',}, copy=True)

    poly_3_results = results[(results.param_kernel == 'poly') & (results.param_degree == 3)]
    poly_3_results = poly_3_results.astype({'param_C': 'float64', 'param_degree': 'float64'}, copy=True)
    
    poly_5_results = results[(results.param_kernel == 'poly') & (results.param_degree == 5)]
    poly_5_results = poly_5_results.astype({'param_C': 'float64', 'param_degree': 'float64'}, copy=True)
    
    poly_7_results = results[(results.param_kernel == 'poly') & (results.param_degree == 7)]
    poly_7_results = poly_7_results.astype({'param_C': 'float64', 'param_degree': 'float64'}, copy=True)
    
    def plot_heatmap(res, kernel_name):
        validation_results = res.pivot(index='param_C', columns='param_gamma', values='mean_test_score')
        plt.figure()
        ax = plt.axes()
        sns.heatmap(validation_results, cbar_kws={'label': 'Precision'}, ax=ax, vmin=0.0, vmax=1.0, annot=True)
        ax.set_title(kernel_name+' Validation Results')
        filetitle = dir+'/Experiment 3/'+kernel_name+' Validation.png'
        plt.savefig(filetitle)
        train_results = res.pivot(index='param_C', columns='param_gamma', values='mean_train_score')
        plt.figure()
        ax = plt.axes()
        sns.heatmap(train_results, cbar_kws={'label': 'Precision'}, ax=ax, vmin=0.0, vmax=1.0, annot=True)
        ax.set_title(kernel_name+' Training Results')
        filetitle = dir+'/Experiment 3/'+kernel_name+' Training.png'
        plt.savefig(filetitle)
    
    plot_heatmap(rbf_results, 'rbf Kernel')
    plot_heatmap(sigmoid_results, 'Sigmoid Kernel')
    plot_heatmap(poly_3_results, 'Degree 3 Polynomial Kernel')
    plot_heatmap(poly_5_results, 'Degree 5 Polynomial Kernel')
    plot_heatmap(poly_7_results, 'Degree 7 Polynomial Kernel')
    
    return results

### Experiment 3 ###
exp_3_param_grid = [{   'kernel': ['poly'],
                        'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                        'degree': [3, 5, 7],
                        'gamma': [1, 0.1, 0.01, .001]},
                    {   'kernel': ['linear'],
                        'C': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
                    {
                        'kernel': ['sigmoid', 'rbf'],
                        'C': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                        'gamma': [1, 0.1, 0.01, .001]}
                    ]

exp_3_start_time = time.time()
exp_3_search = GridSearchCV(SVC(gamma='scale', class_weight='balanced', random_state=75), param_grid=exp_3_param_grid, scoring=scorer, cv=5, return_train_score=True, n_jobs=-1)
exp_3_search.fit(X_train_scaled, y_train)
exp_3_search_results = pd.DataFrame(exp_3_search.cv_results_)
exp_3_results = plot_svm_search_results(exp_3_search, exp_3_search_results)
print('Experiment Time: {:6.4f} seconds'.format(time.time() - exp_3_start_time))
# print('Experiment 3 Results')
# print(exp_3_results)


### Hyperparameter Choice ###
svm_criteria = (exp_3_results.param_kernel == 'linear') & (exp_3_results.param_C == 0.001)
chosen_svm_precision = exp_3_results[svm_criteria].mean_test_score.values[0]
chosen_svm_tolerance = exp_3_results[svm_criteria].validation_upper_ci.values[0] - chosen_svm_precision
print('The chosen SVM model had a linear kernel and C = 0.001')
print('This model has a precision of {:.2f} +/- {:.2f}'.format(chosen_svm_precision, chosen_svm_tolerance))


##### Neural Networks #####
def plot_nn_search_results(search, results):
    results = calculateCI(search, results)
    
    results['num_hidden_layers'] = results.apply(lambda row: len(row.param_hidden_layer_sizes), axis=1)
    results['nodes_per_layer'] = results.apply(lambda row: row.param_hidden_layer_sizes[0], axis=1)
                 
    unique_depths = np.unique(results.num_hidden_layers.values)
    unique_activations = np.unique(results.param_activation.values)
    
    unique_heatmaps = [(depth,activation) for depth in unique_depths for activation in unique_activations]
        
    best_nn_combos = pd.DataFrame(columns=results.columns)

    os.mkdir(dir+'/Experiment 4')
    
    for vals in unique_heatmaps:
        df_current = None
        df_current = results.copy()
        df_current = df_current[(df_current.param_activation == vals[1]) & (df_current.num_hidden_layers ==  vals[0])]
        validation_results = df_current.pivot(index='param_alpha', columns='nodes_per_layer', values='mean_test_score')
        train_results = df_current.pivot(index='param_alpha', columns='nodes_per_layer', values='mean_train_score')
        plt.figure()
        ax = plt.axes()
        sns.heatmap(validation_results, cbar_kws={'label': 'Precision'}, ax=ax, vmin=0.0, vmax=1.0)
        ax.set_title('Validation Results: {} Activation with {} hidden layers'.format(vals[1], vals[0]))
        filetitle = dir+'/Experiment 4/{} with {} hidden layers Validation.png'.format(vals[1], vals[0])
        plt.savefig(filetitle)
        plt.figure()
        ax = plt.axes()
        sns.heatmap(train_results, cbar_kws={'label': 'Precision'}, ax=ax, vmin=0.0, vmax=1.0)
        ax.set_title('Training Results: {} Activation with {} hidden layers'.format(vals[1], vals[0]))
        filetitle = dir+'/Experiment 4/{} with {} hidden layers Training.png'.format(vals[1], vals[0])
        plt.savefig(filetitle)
        ndx_of_max = df_current.mean_test_score.values.argmax()
        max_row = df_current.iloc[ndx_of_max,:]
        best_nn_combos = best_nn_combos.append(max_row)

    best_results = best_nn_combos.copy()
    fig = plt.figure(figsize=(11,8))
    best_nn_combos['label'] = best_nn_combos.apply(lambda row: '{}, {}, {}'.format(row.param_activation, row.param_alpha, row.param_hidden_layer_sizes), axis=1)
    best_nn_combos['val_lower'] = best_nn_combos.mean_test_score - best_nn_combos.validation_lower_ci
    best_nn_combos['val_upper'] = best_nn_combos.validation_upper_ci - best_nn_combos.mean_test_score
    best_nn_combos['trn_lower'] = best_nn_combos.mean_train_score - best_nn_combos.train_lower_ci
    best_nn_combos['trn_upper'] = best_nn_combos.train_upper_ci - best_nn_combos.mean_train_score
    print(best_nn_combos)
    print([best_nn_combos.validation_lower_ci.values, best_nn_combos.validation_upper_ci.values])
    width = 0.4
    plt.bar(np.arange(len(best_nn_combos))+width, best_nn_combos.mean_train_score, yerr=[best_nn_combos.trn_lower, best_nn_combos.trn_upper], width=width, tick_label=best_nn_combos.label)
    plt.bar(np.arange(len(best_nn_combos)), best_nn_combos.mean_test_score, yerr=[best_nn_combos.val_lower, best_nn_combos.val_upper], width=width, tick_label=best_nn_combos.label)
    plt.xticks(rotation=70)
    plt.title('Best Model for each Activation & Depth combination')
    plt.xlabel('Activation, Alpha, Hidden Layer Sizes')
    plt.ylabel('Precision')
    plt.legend(['Train', 'Validation'])
    plt.savefig(dir+'/Experiment 4/Best Neural Networks.png', bbox_inches='tight')

    return best_results
    
### Experiment 4 ###
hidden_layers_1 = [(x,x) for x in np.arange(1,26,1)]
hidden_layers_2 = [(x,x,x) for x in np.arange(1,26,1)]
param_grid = [{
                'hidden_layer_sizes': hidden_layers_1 + hidden_layers_2,
                'activation': ['logistic','relu','tanh'],
                'alpha': [1, .1, .01, .001, 0.0001, .00001, .000001, .0000001]
              }]

exp_4_start_time = time.time()
exp_4_search = GridSearchCV(MLPClassifier(solver='lbfgs', random_state=193), param_grid=param_grid, scoring=scorer, cv=5, return_train_score=True, n_jobs=-1)
exp_4_search.fit(X_train_scaled, y_train)
exp_4_search_results = pd.DataFrame(exp_4_search.cv_results_)
exp_4_best_results = plot_nn_search_results(exp_4_search, exp_4_search_results)
print('Experiment Time: {:6.4f} seconds'.format(time.time() - exp_4_start_time))
# print('Experiment 4 Best Results')
# print(exp_4_best_results)

### Hyperparameter Choice ###
nn_criteria = (exp_4_best_results.param_activation == 'tanh') & (exp_4_best_results.num_hidden_layers == 2)
chosen_nn_precision = exp_4_best_results[nn_criteria].mean_test_score.values[0]
chosen_nn_tolerance = exp_4_best_results[nn_criteria].validation_upper_ci.values[0] - chosen_nn_precision
print('The chosen Neural Network model had a tanh activation, 2 hidden layers, 7 nodes per layer, and alpha = 1.0')
print('This model has a precision of {:.2f} +/- {:.2f}'.format(chosen_nn_precision, chosen_nn_tolerance))


########## Algorithm Performance ##########
### Experiment 5 ###
kf = KFold(n_splits=3, shuffle=True, random_state=84)
splits = kf.split(X_test_scaled)

results_arr = []
np.random.seed(47)

boosted_time = []
support_time = []
nn_time = []

for train_ind, test_ind in splits:
    X_additional_train_cur = X_test_scaled[train_ind]
    y_additional_train_cur = y_test[train_ind]
    X_test_cur = X_test_scaled[test_ind]
    y_test_cur = y_test[test_ind]
    
    X_train_cur = np.concatenate((X_train_scaled,X_additional_train_cur))
    y_train_cur = np.concatenate((y_train, y_additional_train_cur))
    
    indices = np.arange(0, len(X_train_cur), 1)
    np.random.shuffle(indices)
    
    X_train_cur = X_train_cur[indices]
    y_train_cur = y_train_cur[indices]
    
    pre_boost = time.time()
    boosted = AdaBoostClassifier(random_state=62, learning_rate=0.7, n_estimators=10)
    boosted.fit(X_train_cur, y_train_cur)
    boosted_time.append(time.time()-pre_boost)
    
    pre_support = time.time()
    support = SVC(gamma='scale', class_weight='balanced', random_state=75, kernel='linear', C=0.001)
    support.fit(X_train_cur, y_train_cur)
    support_time.append(time.time()-pre_support)
    
    pre_nn = time.time()
    nn = MLPClassifier(solver='lbfgs', random_state=193, hidden_layer_sizes=(5,5), activation='tanh', alpha=1)
    nn.fit(X_train_cur, y_train_cur)
    nn_time.append(time.time()-pre_nn)

    boosted_train_predict = boosted.predict(X_train_cur)
    boosted_test_predict = boosted.predict(X_test_cur)
    support_train_predict = support.predict(X_train_cur)
    support_test_predict = support.predict(X_test_cur)
    nn_train_predict = nn.predict(X_train_cur)
    nn_test_predict = nn.predict(X_test_cur)

    boosted_train_precision = precision_score(y_train_cur, boosted_train_predict, pos_label='saved')
    boosted_test_precision = precision_score(y_test_cur, boosted_test_predict, pos_label='saved')
    support_train_precision = precision_score(y_train_cur, support_train_predict, pos_label='saved')
    support_test_precision = precision_score(y_test_cur, support_test_predict, pos_label='saved')
    nn_train_precision = precision_score(y_train_cur, nn_train_predict, pos_label='saved')
    nn_test_precision = precision_score(y_test_cur, nn_test_predict, pos_label='saved')
    
    boosted_train_confusion = confusion_matrix(y_train_cur, boosted_train_predict, labels=['saved', 'unsaved'])
    boosted_test_confusion = confusion_matrix(y_test_cur, boosted_test_predict, labels=['saved', 'unsaved'])
    support_train_confusion = confusion_matrix(y_train_cur, support_train_predict, labels=['saved', 'unsaved'])
    support_test_confusion = confusion_matrix(y_test_cur, support_test_predict, labels=['saved', 'unsaved'])
    nn_train_confusion = confusion_matrix(y_train_cur, nn_train_predict, labels=['saved', 'unsaved'])
    nn_test_confusion = confusion_matrix(y_test_cur, nn_test_predict, labels=['saved', 'unsaved'])
    
    results_cur = {
        'boosted_train_precision' : boosted_train_precision,
        'boosted_test_precision' : boosted_test_precision,
        'support_train_precision' : support_train_precision,
        'support_test_precision' : support_test_precision,
        'nn_train_precision': nn_train_precision,
        'nn_test_precision' : nn_test_precision,
        'boosted_train_confusion' : boosted_train_confusion,
        'boosted_test_confusion' : boosted_test_confusion,
        'support_train_confusion' : support_train_confusion,
        'support_test_confusion' : support_test_confusion,
        'nn_train_confusion' : nn_train_confusion,
        'nn_test_confusion' : nn_test_confusion
    }
    

    results_arr.append(results_cur)

print('Random Forests with Boosting Mean Training Time: {:.2f} seconds'.format(np.mean(boosted_time)))
print('Support Vector Machine Mean Training Time: {:.2f} seconds'.format(np.mean(support_time)))
print('Neural Network Mean Training Time: {:.2f} seconds'.format(np.mean(nn_time)))


df_algo_compare = pd.DataFrame(results_arr)
mean_algo_results = np.mean(df_algo_compare)
std_algo_results = np.std(df_algo_compare)

data = {
    'boosted': {
        'mean_test': mean_algo_results.boosted_test_precision,
        'mean_train': mean_algo_results.boosted_train_precision,
        'lower_test_ci': t.interval(0.95, 2, loc=mean_algo_results.boosted_test_precision,
                                    scale=std_algo_results.boosted_test_precision)[0],
        'upper_test_ci': t.interval(0.95, 2, loc=mean_algo_results.boosted_test_precision,
                                    scale=std_algo_results.boosted_test_precision)[1],
        'lower_train_ci': t.interval(0.95, 2, loc=mean_algo_results.boosted_train_precision,
                                    scale=std_algo_results.boosted_train_precision)[0],
        'upper_train_ci': t.interval(0.95, 2, loc=mean_algo_results.boosted_train_precision,
                                    scale=std_algo_results.boosted_train_precision)[1],
        'test_confusion': mean_algo_results.boosted_test_confusion,
        'train_confusion': mean_algo_results.boosted_train_confusion
    },
    'support': {
        'mean_test': mean_algo_results.support_test_precision,
        'mean_train': mean_algo_results.support_train_precision,
        'lower_test_ci': t.interval(0.95, 2, loc=mean_algo_results.support_test_precision,
                                    scale=std_algo_results.support_test_precision)[0],
        'upper_test_ci': t.interval(0.95, 2, loc=mean_algo_results.support_test_precision,
                                    scale=std_algo_results.support_test_precision)[1],
        'lower_train_ci': t.interval(0.95, 2, loc=mean_algo_results.support_train_precision,
                                    scale=std_algo_results.support_train_precision)[0],
        'upper_train_ci': t.interval(0.95, 2, loc=mean_algo_results.support_train_precision,
                                    scale=std_algo_results.support_train_precision)[1],
        'test_confusion': mean_algo_results.support_test_confusion,
        'train_confusion': mean_algo_results.support_train_confusion
    },
    'nn': {
        'mean_test': mean_algo_results.nn_test_precision,
        'mean_train': mean_algo_results.nn_train_precision,
        'lower_test_ci': t.interval(0.95, 2, loc=mean_algo_results.nn_test_precision,
                                    scale=std_algo_results.nn_test_precision)[0],
        'upper_test_ci': t.interval(0.95, 2, loc=mean_algo_results.nn_test_precision,
                                    scale=std_algo_results.nn_test_precision)[1],
        'lower_train_ci': t.interval(0.95, 2, loc=mean_algo_results.nn_train_precision,
                                    scale=std_algo_results.nn_train_precision)[0],
        'upper_train_ci': t.interval(0.95, 2, loc=mean_algo_results.nn_train_precision,
                                    scale=std_algo_results.nn_train_precision)[1],
        'test_confusion': mean_algo_results.nn_test_confusion,
        'train_confusion': mean_algo_results.nn_train_confusion
    }
}

df_algo_results = pd.DataFrame.from_dict(data, orient='index')
print('Experiment 5 Results')
print(df_algo_results)

boosted_precision = df_algo_results.mean_test.boosted
boosted_tolerance = df_algo_results.upper_test_ci.boosted - boosted_precision
support_precision = df_algo_results.mean_test.support
support_tolerance = df_algo_results.upper_test_ci.support - support_precision
nn_precision = df_algo_results.mean_test.nn
nn_tolerance = df_algo_results.upper_test_ci.nn - nn_precision

print('The Random Forest with Boosting  has a precision of {:.2f} +/- {:.2f}'.format(boosted_precision, boosted_tolerance))
print('The Support Vector Machine has a precision of {:.2f} +/- {:.2f}'.format(support_precision, support_tolerance))
print('The Neural Network has a precision of {:.2f} +/- {:.2f}'.format(nn_precision, nn_tolerance))

df_algo_results['test_upper'] = df_algo_results.mean_test - df_algo_results.lower_test_ci
df_algo_results['test_lower'] = df_algo_results.upper_test_ci - df_algo_results.mean_test
df_algo_results['trn_lower'] = df_algo_results.mean_train - df_algo_results.lower_train_ci
df_algo_results['trn_upper'] = df_algo_results.upper_train_ci - df_algo_results.mean_train

fig = plt.figure(figsize=(11,8))
width=0.4
plt.bar(np.arange(len(df_algo_results))+width, df_algo_results.mean_train, yerr=[df_algo_results.trn_lower, df_algo_results.trn_upper], width=width, tick_label=df_algo_results.index)
plt.bar(np.arange(len(df_algo_results)), df_algo_results.mean_test, yerr=[df_algo_results.test_lower, df_algo_results.test_upper], width=width, tick_label=df_algo_results.index)
plt.xticks(rotation=70, ticks=np.arange(3), labels=['Random Forest with Boosting', 'Support Vector Machine', 'Neural Network'])
plt.title('Algorithm Comparison')
plt.ylabel('Precision')
plt.legend(['Train', 'Validation'])
plt.savefig(dir+'/Experiment 5 Plot.png', bbox_inches='tight')

########## Algorithm Characteristics ##########
### Confusion Matrices ###
boosted_conf = df_algo_results.test_confusion.boosted
svm_conf = df_algo_results.test_confusion.support
nn_conf = df_algo_results.test_confusion.nn

max_predict = np.max(np.concatenate((boosted_conf,svm_conf,nn_conf)))

plt.figure()
ax = plt.axes()
sns.heatmap(df_algo_results.test_confusion.boosted, cbar_kws={'label':'Average # Predicted'}, ax=ax, 
            annot=True, annot_kws={'size': 20}, fmt='.2f', vmin=0, vmax=max_predict)
ax.set_xticklabels(['predicted saved', 'predicted unsaved'], {'fontsize': 14})
ax.set_yticklabels(['saved', 'unsaved'], {'fontsize': 14})
ax.set_title('Confusion Matrix for Random Forest with Boosting', {'fontsize': 16})
plt.savefig(dir+'/Confusion Matrix Random Forest.png')

plt.figure()
ax = plt.axes()
sns.heatmap(df_algo_results.test_confusion.support, cbar_kws={'label':'Average # Predicted'}, ax=ax,
            annot=True, annot_kws={'size': 20}, fmt='.2f', vmin=0, vmax=max_predict)
ax.set_xticklabels(['predicted saved', 'predicted unsaved'], {'fontsize': 14})
ax.set_yticklabels(['saved', 'unsaved'], {'fontsize': 14})
ax.set_title('Confusion Matrix for Support Vector Machine', {'fontsize': 16})
plt.savefig(dir+'/Confusion Matrix Support Vector Machine.png')

plt.figure()
ax = plt.axes()
sns.heatmap(df_algo_results.test_confusion.nn, cbar_kws={'label':'Average # Predicted'}, ax=ax,
            annot=True, annot_kws={'size': 20}, fmt='.2f', vmin=0, vmax=max_predict)
ax.set_xticklabels(['predicted saved', 'predicted unsaved'], {'fontsize': 14})
ax.set_yticklabels(['saved', 'unsaved'], {'fontsize': 14})
ax.set_title('Confusion Matrix for Neural Network', {'fontsize': 16})
plt.savefig(dir+'/Confusion Matrix Neural Network.png')