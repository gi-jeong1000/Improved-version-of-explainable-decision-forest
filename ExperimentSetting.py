import datetime
from ConjunctionSet import ConjunctionSet
from DataPreperation import *
from Tuningrandomforest import *
from QuickPruning import *
from pruningFunctions import *
from NewModelBuilder import *
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
import pickle
from CMM import *
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from Node import *
from sklearn.model_selection import GridSearchCV
import warnings
import visualize
warnings.filterwarnings('ignore')


class ExperimentSetting():
    def __init__(self, number_of_branches_threshold, df_names, number_of_estimators, fixed_params, num_of_iterations=30):
        self.num_of_iterations = num_of_iterations
        self.number_of_branches_threshold = number_of_branches_threshold
        self.df_names = df_names
        self.fixed_params = fixed_params
        self.number_of_estimators = number_of_estimators
        self.label_encoder = LabelEncoder()

    def run(self):
        self.experiments = []
        for threshold in self.number_of_branches_threshold:
            for df_name in self.df_names:
                df, x_columns, y_column, feature_types = get_dataset_by_string(df_name)
                self.label_encoder.fit(df[y_column].unique())
                df[y_column] = self.label_encoder.transform(df[y_column])  # 레이블 인코딩
                d = {'max_number_of_branches': threshold, 'df_name': df_name, 'number_of_estimators': self.number_of_estimators}
                print(d)
                self.run_experiment(threshold, df, x_columns, y_column, feature_types, d)
                self.save_results_to_text("experiment_penguin.txt", self.experiments)

    def visualize_results(self):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='dataset', y='accuracy', data=self.results_df)
        sns.swarmplot(x='dataset', y='accuracy', data=self.results_df, color=".25")
        plt.title('Accuracy by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Accuracy')
        plt.show()

    def visualize_tree(self, new_model, branches_df):
        # Print If-Then rules
        print("Decision Tree as If-Then Rules:")
        visualize.print_if_then_rules(new_model)

        # Generate DOT representation
        dot = visualize.generate_dot(new_model)
        dot.render('decision_tree', format='png', view=True)

    def run(self):
        self.experiments = []
        for threshold in self.number_of_branches_threshold:
            for df_name in self.df_names:
                df, x_columns, y_column, feature_types = get_dataset_by_string(df_name)
                self.label_encoder.fit(df[y_column].unique())
                df[y_column] = self.label_encoder.transform(df[y_column])  # 레이블 인코딩
                d = {'max_number_of_branches': threshold, 'df_name': df_name,
                     'number_of_estimators': self.number_of_estimators}
                print(d)
                self.run_experiment(threshold, df, x_columns, y_column, feature_types, d)
                self.save_results_to_text("experiment_penguin.txt", self.experiments)

    def visualize_results(self):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='dataset', y='accuracy', data=self.results_df)
        sns.swarmplot(x='dataset', y='accuracy', data=self.results_df, color=".25")
        plt.title('Accuracy by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Accuracy')
        plt.show()

    def visualize_tree(self, new_model, branches_df):
        # Print If-Then rules
        print("Decision Tree as If-Then Rules:")
        visualize.print_if_then_rules(new_model)

        # Generate DOT representation
        dot = visualize.generate_dot(new_model)
        dot.render('decision_tree', format='png', view=True)

    def run_experiment(self, branch_probability_threshold, df, x_columns, y_column, feature_types,
                       hyper_parameters_dict):
        for i in range(self.num_of_iterations):
            print(i)
            np.random.seed(i)
            num_of_estimators = hyper_parameters_dict['number_of_estimators']
            result_dict = dict(hyper_parameters_dict)
            result_dict['iteration'] = i
            output_path = f'pickles_100trees/{hyper_parameters_dict["df_name"]}_{result_dict["iteration"]}'

            trainAndValidation_x, trainAndValidation_y, test_x, test_y = divide_to_train_test(df, x_columns, y_column)
            train_x = trainAndValidation_x[:int(len(trainAndValidation_x) * 0.8)]
            train_y = trainAndValidation_y[:int(len(trainAndValidation_x) * 0.8)]
            validation_x = trainAndValidation_x[int(len(trainAndValidation_x) * 0.8):]
            validation_y = trainAndValidation_y[int(len(trainAndValidation_x) * 0.8):]

            start_temp = datetime.datetime.now()
            rf = RandomForestClassifier(n_estimators=num_of_estimators, max_depth=5,
                                        min_samples_leaf=max(1, int(0.02 * len(train_x))), **self.fixed_params)
            rf.fit(trainAndValidation_x, trainAndValidation_y)
            result_dict['random forest training time'] = (datetime.datetime.now() - start_temp).total_seconds()
            self.classes_ = rf.classes_

            start_temp1 = datetime.datetime.now()
            new_pruning(rf, trainAndValidation_x, trainAndValidation_y, 10)
            result_dict['Prposed pruning time'] = (datetime.datetime.now() - start_temp1).total_seconds()

            start_temp2 = datetime.datetime.now()
            reduce_error_pruning(rf, trainAndValidation_x, trainAndValidation_y, 10)
            result_dict['old pruning time'] = (datetime.datetime.now() - start_temp2).total_seconds()

            start_temp = datetime.datetime.now()
            cs = ConjunctionSet(x_columns, trainAndValidation_x, trainAndValidation_x, trainAndValidation_y, rf,
                                feature_types, hyper_parameters_dict['max_number_of_branches'])
            result_dict['conjunction set training time'] = (datetime.datetime.now() - start_temp).total_seconds()
            result_dict['number of branches per iteration'] = cs.number_of_branches_per_iteration
            result_dict['number_of_branches'] = len(cs.conjunctionSet)

            start_temp = datetime.datetime.now()
            branches_df = cs.get_conjunction_set_df().round(decimals=5)
            result_dict['number_of_features_for_new_model'] = len(branches_df.columns)
            for i in range(2):
                branches_df[rf.classes_[i]] = [probas[i] for probas in branches_df['probas']]
            df_dict = {col: branches_df[col].values for col in branches_df.columns}
            new_model = Node([True] * len(branches_df), feature_names=x_columns, label_encoder=self.label_encoder)

            new_model.split(df_dict)
            result_dict['new model training time'] = (datetime.datetime.now() - start_temp).total_seconds()

            predictions_new_model = [np.argmax(new_model.predict_probas_and_depth(inst, branches_df)[0]) for inst in
                                     test_x]

            accuracy_new_model = np.sum(predictions_new_model == test_y) / len(test_y)
            precision_new_model = precision_score(test_y, predictions_new_model, average='macro')
            recall_new_model = recall_score(test_y, predictions_new_model, average='macro')
            f1_new_model = f1_score(test_y, predictions_new_model, average='macro')

            result_dict.update({
                'new_model_accuracy': accuracy_new_model,
                'new_model_precision': precision_new_model,
                'new_model_recall': recall_new_model,
                'new_model_f1': f1_new_model
            })

            with open(output_path, 'wb') as fp:
                pickle.dump(result_dict, fp)
            self.experiments.append(result_dict)

            # visualize the new model after training and testing
            self.visualize_tree(new_model, branches_df)

    def save_results_to_text(self, file_path, results):
        with open(file_path, 'w') as file:
            for result in results:
                for key, value in result.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")


    def decision_tree_measures(self, X, Y, dt_model):
        result_dict = {}
        probas = []
        depths = []
        for inst in X:
            pred, dept = self.tree_depth_and_prediction(inst, dt_model.tree_)
            probas.append(pred)
            depths.append(dept)
        predictions = dt_model.predict(X)
        result_dict['decision_tree_average_depth'] = np.mean(depths)
        result_dict['decision_tree_min_depth'] = np.min(depths)
        result_dict['decision_tree_max_depth'] = np.max(depths)
        result_dict['decision_tree_accuracy'] = np.sum(predictions == Y) / len(Y)
        result_dict['decision_tree_auc'] = self.get_auc(Y, np.array(probas), dt_model.classes_)
        result_dict['decision_tree_kappa'] = cohen_kappa_score(Y, predictions)
        return result_dict

    def cmm_tree_measures(self, X, Y, dt_model):
        return {k.replace('decision_tree', 'cmm_tree'): v for k, v in
                self.decision_tree_measures(X, Y, dt_model).items()}

    def new_model_measures(self, X, Y, new_model, branches_df):
        result_dict = {}
        probas, depths = [], []
        for inst in X:
            prob, depth = new_model.predict_probas_and_depth(inst, branches_df)
            probas.append(prob)
            depths.append(depth)
        predictions = [self.classes_[i] for i in np.array([np.argmax(prob) for prob in probas])]
        result_dict['new_model_average_depth'] = np.mean(depths)
        result_dict['new_model_min_depth'] = np.min(depths)
        result_dict['new_model_max_depth'] = np.max(depths)
        result_dict['new_model_accuracy'] = np.sum(predictions == Y) / len(Y)
        result_dict['new_model_auc'] = self.get_auc(Y, np.array(probas), self.classes_)
        result_dict['new_model_kappa'] = cohen_kappa_score(Y, predictions)
        result_dict['new_model_number_of_nodes'] = new_model.number_of_children()
        result_dict['new_model_probas'] = probas

        return result_dict

    def ensemble_measures(self, X, Y, rf):
        result_dict = {}
        predictions, depths = self.ensemble_prediction(X, rf)
        result_dict['ensemble_average_depth'] = np.mean(depths)
        result_dict['ensemble_min_depth'] = np.min(depths)
        result_dict['ensemble max_depth'] = np.max(depths)
        ensemble_probas = rf.predict_proba(X)
        result_dict['ensemble_accuracy'] = np.sum(rf.predict(X) == Y) / len(Y)
        result_dict['ensemble_auc'] = self.get_auc(Y, ensemble_probas, rf.classes_)
        result_dict['ensemble_kappa'] = cohen_kappa_score(Y, rf.predict(X))
        result_dict['ensemble_probas'] = ensemble_probas
        return result_dict

    def ensemble_prediction(self, X, rf):
        predictions = []
        depths = []
        for inst in X:
            pred = []
            depth = 0
            for base_model in rf.estimators_:
                res = self.tree_depth_and_prediction(inst, base_model.tree_)
                pred.append(res[0])
                depth += res[1]
            predictions.append(np.array(pred).mean(axis=0))
            depths.append(depth)
        return predictions, depths

    def tree_depth_and_prediction(self, inst, t):
        indx = 0
        depth = 0
        epsilon = 0.0000001
        while t.feature[indx] >= 0:
            if inst[t.feature[indx]] <= t.threshold[indx] + epsilon:
                indx = t.children_left[indx]
            else:
                indx = t.children_right[indx]
            depth += 1
        return np.array([i / np.sum(t.value[indx][0]) for i in t.value[indx][0]]), depth

    def get_auc(self, Y, y_score, classes):
        y_test_binarize = np.array([[1 if i == c else 0 for c in classes] for i in Y])
        fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
        return auc(fpr, tpr)

    def fit_decision_tree_model(self, train_x, train_y):
        parameters = {'max_depth': [3, 10, 20],
                      'criterion': ['gini', 'entropy'],
                      'min_samples_leaf': [1, 2, 10]}
        model = DecisionTreeClassifier()
        clfGS = GridSearchCV(model, parameters, cv=10)
        clfGS.fit(train_x, train_y)
        model = clfGS.best_estimator_
        return model

    def fit_cmm_tree(self, df, x_columns, y_column, rf):
        synthetic_data = get_synthetic_data(df)
        cmm_dt = train_dt_for_synthetic_data(synthetic_data, x_columns, y_column, rf)
        return cmm_dt

    def save_results_to_text(self,file_path, results):
        with open(file_path, 'w') as file:
            for result in results:
                for key, value in result.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")  # 각 실험 결과 사이에 공백 줄 추가