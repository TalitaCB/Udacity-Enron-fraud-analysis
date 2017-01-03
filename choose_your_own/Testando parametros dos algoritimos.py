import numpy as np

from rf_tester import *
from parameter_plots import *

from prep_terrain_data import makeTerrainData

from bokeh.io import output_notebook
output_notebook()


#Random Forest
data_generator = makeTerrainData
sample_size = 1000
random_state = 3599

default_n_estimators = 10
default_criterion = "gini"
default_max_depth = None
default_min_samples_split = 2
default_min_samples_leaf = 1
default_min_weight_fraction_leaf = 0.0
default_max_features = 'auto'
default_max_leaf_nodes = None
default_bootstrap = True
default_oob_score = False
default_warm_start = False
default_class_weight = None


seeds = range(1,10000)
seed_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = seeds,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )


parameter_plots(seeds, results_dict=seed_results,
                x_label="Seed #",
                title_accuracy="Seed # vs Accuracy",
                title_time="Seed # vs Training Time",
                legend_pos="left_center")

sample_sizes = range(1000, 30000+1, 1000) + \
               range(35000, 60000+1, 5000) + \
               range(70000, 100000+1, 10000) + \
               [150000, 200000]
sample_sizes_results = loop_rf(data_generator,
            sample_size=sample_sizes,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = 303,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )

scaled_sample_sizes = np.array(sample_sizes) / 1000.0
parameter_plots(scaled_sample_sizes, results_dict=sample_sizes_results,
                x_label="Sample Size (in thousands)",
                title_accuracy="Sample Size vs Accuracy",
                title_time="Sample Size vs Training Time",
                legend_pos="bottom_right")


criteria = ["gini", "entropy"]
criteria_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = criteria,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = random_state,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )


max_depths = range(1,30) + range(35,60,5) + range(70, 200, 10)
max_depths_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = max_depths,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = random_state,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )


parameter_plots(max_depths, results_dict=max_depths_results,
                x_label="Max Depth",
                title_accuracy="Max Depth vs Accuracy",
                title_time="Max Depth vs Training Time",
                legend_pos="bottom_right")

min_samples_splits = range(2, 100)
min_samples_split_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = min_samples_splits,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = random_state,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )

parameter_plots(min_samples_splits, results_dict=min_samples_split_results,
                x_label="min samples splits",
                title_accuracy="min samples splits vs Accuracy",
                title_time="min samples splits vs Training Time",
                legend_pos="top_right")


min_samples_leafs = range(1, 100)
min_samples_leaf_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = min_samples_leafs,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = random_state,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )

parameter_plots(min_samples_leafs, results_dict=min_samples_leaf_results,
                x_label="min_samples_leafs",
                title_accuracy="min_samples_leafs vs Accuracy",
                title_time="min_samples_leafs vs Training Time",
                legend_pos="top_right")

max_leaf_nodes = range(2,300)
max_leaf_nodes_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = random_state,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )


parameter_plots(max_leaf_nodes, results_dict=max_leaf_nodes_results,
                x_label="max_leaf_nodes",
                title_accuracy="max_leaf_nodes vs Accuracy",
                title_time="max_leaf_nodes vs Training Time",
                legend_pos="top_right")


bootstraps = [False, True]
bootstrap_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = bootstraps,
            oob_score = default_oob_score,
            random_state = random_state,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )


parameter_plots(bootstraps, results_dict=bootstrap_results,
                x_label="bootstrap",
                title_accuracy="bootstrap vs Accuracy",
                title_time="bootstrap vs Training Time",
                legend_pos="left_center")

oobs = [False, True]
oob_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = oobs,
            random_state = random_state,
            warm_start = default_warm_start,
            class_weight = default_class_weight
            )

parameter_plots(oobs, results_dict=oob_results,
                x_label="Using Out of bag Error Estimate",
                title_accuracy="Using Out of bag Error Estimate vs Accuracy",
                title_time="Using Out of bag Error Estimate vs Training Time",
                legend_pos="top_right")


warm_starts = [False, True]
warm_start_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = random_state,
            warm_start = warm_starts,
            class_weight = default_class_weight
            )

parameter_plots(warm_starts, results_dict=warm_start_results,
                x_label="Warm Start",
                title_accuracy="Warm Start vs Accuracy",
                title_time="Warm Start vs Training Time",
                legend_pos="right_center")

class_weights = ["balanced", "balanced_subsample", None]
class_weight_results = loop_rf(data_generator,
            sample_size=sample_size,
            n_estimators = default_n_estimators,
            criterion = default_criterion,
            max_depth = default_max_depth,
            min_samples_split = default_min_samples_split,
            min_samples_leaf = default_min_samples_leaf,
            min_weight_fraction_leaf = default_min_weight_fraction_leaf,
            max_features = default_max_features,
            max_leaf_nodes = default_max_leaf_nodes,
            bootstrap = default_bootstrap,
            oob_score = default_oob_score,
            random_state = random_state,
            warm_start = default_warm_start,
            class_weight = class_weights
            )


#Adaboost
from adaboost_tester import *


#==========================================================================
#                                                       SCATTERPLOT OF DATA
#==========================================================================
X_train,Y_train,X_test,Y_test = makeTerrainData(n_points=1000)
X_train = np.array(X_train)

p = figure(plot_width=500, plot_height=500,
          title="Terrain Data", x_axis_label='x1', y_axis_label='x2')
p.circle(X_train[:,0], X_train[:,1], size=10,
         color=map(lambda x: COLORS[int(x)], Y_train), alpha=0.3)
p.logo = None
p.toolbar_location = None
#output_file("myPlot.html")
show(p)

data_generator = makeTerrainData
default_n_points=1000      # Size of dataset generated.
default_max_depth=1         # Max depth of weak decision tree classifier.
default_n_estimators=50     # Number of classifiers to use for boosting.
default_learning_rate=1.0   # Learning rate for the boosted classifiers
seed = 387                  # Setting random seed for reproducible results


sample_sizes = range(1000, 30000+1, 1000) + \
               range(35000, 60000+1, 5000) + \
               range(70000, 100000+1, 10000) + \
               [150000, 200000]
sample_size_results = loop_adaboost_with_simple_tree(data_generator,
                               sample_size=sample_sizes,
                               max_depth=default_max_depth,
                               n_estimators=default_n_estimators,
                               learning_rate=default_learning_rate,
                               random_state=seed
                               )

scaled_sample_sizes = np.array(sample_sizes) / 1000.0
parameter_plots(scaled_sample_sizes, results_dict=sample_size_results,
                x_label="Sample Size (in thousands)",
                title_accuracy="Sample Size vs Accuracy",
                title_time="Sample Size vs Training Time")


max_depths = range(1,20+1)
max_depth_results = loop_adaboost_with_simple_tree(data_generator,
                               sample_size=default_n_points,
                               max_depth=max_depths,
                               n_estimators=default_n_estimators,
                               learning_rate=default_learning_rate,
                               random_state=seed
                               )

parameter_plots(max_depths, results_dict=max_depth_results,
                x_label="Max Depth",
                title_accuracy="Max Depth vs Accuracy",
                title_time="Max Depth vs Training Time",
                legend_pos="right_center")


n_estimators= range(1,20+1) + range(25, 100+1, 5) + range(125, 300, 25)
n_estimators_results = loop_adaboost_with_simple_tree(data_generator,
                               sample_size=default_n_points,
                               max_depth=default_max_depth,
                               n_estimators=n_estimators,
                               learning_rate=default_learning_rate,
                               random_state=seed
                               )

parameter_plots(n_estimators, results_dict=n_estimators_results,
                x_label="Number of Estimators",
                title_accuracy="Num Estimators vs Accuracy",
                title_time="Num Estimators vs Training Time",
                legend_pos="bottom_right")

learning_rates = list(np.arange(0.001, 0.1 + 0.001, 0.001)) + \
                 list(np.arange(0.1, 1.0 + 0.1, 0.01)) + \
                 list(np.arange(1.5, 10.0 + 0.5, 0.5))

learning_rates_results = loop_adaboost_with_simple_tree(data_generator,
                                                        sample_size=default_n_points,
                                                        max_depth=default_max_depth,
                                                        n_estimators=default_n_estimators,
                                                        learning_rate=learning_rates,
                                                        random_state=seed
                                                        )


parameter_plots(learning_rates, results_dict=learning_rates_results,
                x_label="Learning Rate",
                title_accuracy="Learning Rate vs Accuracy",
                title_time="Learning Rate vs Training Time",
                legend_pos="bottom_right")


#k-nearest

data_generator = makeTerrainData
sample_size = 1000
default_k = 8
default_weights = "uniform"
default_algorithm = "auto"
n_jobs = 1

sample_sizes = range(1000, 30000+1, 1000) + \
               range(35000, 60000+1, 5000) + \
               range(70000, 100000+1, 10000) + \
               [150000, 200000]
sample_size_results = loop_knn(data_generator,
                               sample_size=sample_sizes,
                               k=default_k,
                               weights=default_weights,
                               algorithm=default_algorithm,
                               n_jobs=n_jobs,
                               )

scaled_sample_sizes = np.array(sample_sizes) / 1000.0
parameter_plots(scaled_sample_sizes, results_dict=sample_size_results,
                x_label="Sample Size (in thousands)",
                title_accuracy="Sample Size vs Accuracy",
                title_time="Sample Size vs Training Time",
                legend_pos="bottom_right")

ks = range(1, 50) + range(55, 100+1, 5) + range(120, 300+1, 20)
k_results = loop_knn(data_generator,
                               sample_size=sample_size,
                               k=ks,
                               weights=default_weights,
                               algorithm=default_algorithm,
                               n_jobs=n_jobs,
                               )

parameter_plots(ks, results_dict=k_results,
                x_label="# of Neighbors",
                title_accuracy="# of Neighbors vs Accuracy",
                title_time="# of Neighbors vs Training Time",
                legend_pos="top_right")

weights = ["uniform", "distance"]
weights_results = loop_knn(data_generator,
                               sample_size=sample_size,
                               k=default_k,
                               weights=weights,
                               algorithm=default_algorithm,
                               n_jobs=n_jobs,
                               )

weights_labels = [0,1]
parameter_plots(weights_labels, results_dict=weights_results,
                x_label="Weighting Method (0=uniform, 1=distance)",
                title_accuracy="Weighting Method vs Accuracy",
                title_time="Weighting Method vs Training Time",
                legend_pos="right_center")

algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
algorithms_results = loop_knn(data_generator,
                               sample_size=sample_size,
                               k=default_k,
                               weights=default_weights,
                               algorithm=algorithms,
                               n_jobs=n_jobs
                               )

algorithms_labels = range(len(algorithms))
parameter_plots(algorithms_labels, results_dict=algorithms_results,
                x_label="0=auto 1=ball_tree "\
                "2=kd_tree 3=brute)",
                title_accuracy="Algorithm vs Accuracy",
                title_time="Algorithm vs Training Time",
                legend_pos="right_center")

