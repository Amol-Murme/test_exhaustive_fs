# %%
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from django.conf import settings
import os
import json
from typing import List
# import scorecardpy as sc

#import statsmodels.formula.api as smf
# from sklearn.metrics import auc, roc_curve, accuracy_score, roc_auc_score
from matplotlib import pyplot as plt
#from .logistic_results import OverallClassificationResults, auto_str, ClassificationMetrics, plot_precision_recall, plot_roc
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, average_precision_score, auc, roc_curve
from sklearn.metrics import PrecisionRecallDisplay


class LogisticRegressionModel_sklearn():
    def __init__(self, train_filepath, label_col, feature_cols, train_percent, test_percent, cross_validation, test_filepath=None, normalise=False, inferSchema=True, exclude_features=None, stratify=True, logistic_threshold=0.5, seed: int = 2022):
        # self.spark  = self.get_spark_session()
        # self.train_result = ClassificationMetrics(type='train')  # initialise
        # self.test_result = ClassificationMetrics(type='test')  # initialise
        # self.overall_result = OverallClassificationResults()  # initialise this
        self.label_col = label_col
        self.cv = cross_validation
        self.logistic_threshold = logistic_threshold
        self.train_input_data = self.read_csv_data(train_filepath, inferSchema=inferSchema)
        self.feature_cols = feature_cols
        if test_filepath:
            self.test_input_data = self.read_csv_data(test_filepath)
        else:
            self.test_input_data = None

        # self.features = self.train_input_data.columns
        self.exclude_features = exclude_features

        self.remove_cols()
        if normalise:
            self.add_normaliser(label_col)
        self.remove_non_numerical_features()
        if not self.test_input_data:
            self.train_size = train_percent
            self.test_size = test_percent
            self.stratify = stratify
            self.train_split()
        else:
            self.create_train_test_features()
        #self.delete_inputs()
        #self.get_train_test_row_count()
        # print("self.train.count " + str(self.train.count()))
        # print("self.test.count " + str(self.test.count()))
        # self.
        self.build_pipeline()
        self.transform_data()
        # self.lrModel = self.build_logistic_regression_model()
        self.lrModel = self.build_logistic_regression_model_using_sklearn()
        # self.aggregate_test_accuracy_results(type='train')
        # self.aggregate_test_accuracy_results(type='test')
        # self.get_coefficients_names()
        # self.get_intercept()
        
    def read_csv_data(self, filepath: str, inferSchema):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            abs_file_path = os.path.join(
                Path(settings.BASE_DIR).parent, filepath)
            print(abs_file_path)
            if os.path.exists(abs_file_path):
                df = pd.read_csv(abs_file_path)
            else:
                RuntimeError(f"Path {filepath} not found")
        return df

    def remove_cols(self,):
        remove_cols = []
        if self.feature_cols:
            remove_cols = [col for col in list(self.train_input_data.columns) if col not in self.feature_cols]
        if self.exclude_features:
            remove_cols = list(set(remove_cols + self.exclude_features))

        if remove_cols:
            if self.label_col in remove_cols:
                remove_cols.remove(self.label_col)
            
            self.train_input_data.drop(remove_cols, axis=1, inplace=True)
            if self.test_input_data:
                self.test_input_data.drop(remove_cols, axis=1, inplace=True)

    def get_feature_colums(self, label_col):
        cols = self.train_input_data.columns
        cols.remove(label_col)
        return cols

    def remove_non_numerical_features(self):
        numerics = ['int16', 'int32', 'int64', 'float64', 'float32']
        self.train_input_data = self.train_input_data.select_dtypes(include=numerics)

    def train_split(self):
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(self.train_input_data.drop(
            columns=self.label_col), self.train_input_data[self.label_col], train_size=self.train_size, test_size=self.test_size, stratify=self.train_input_data[self.label_col], random_state=41)

    def create_train_test_features(self):
        self.train_features = self.train_input_data.drop(columns=self.label_col)
        self.test_features = self.test_input_data.drop(columns=self.label_col)
        self.train_labels = self.train_input_data[self.label_col]
        self.test_labels = self.test_input_data[self.label_col]

    def delete_inputs(self):
        #  to release memory
        self.train_input_data = None
        self.test_input_data = None
    
    def get_train_test_row_count(self):
        self.overall_result.train_nrows = self.train_features.shape[0]
        self.overall_result.test_nrows = self.test_features.shape[0]

    def build_pipeline(self):
        pass

    def transform_data(self):
        pass

    def build_logistic_regression_model_using_sklearn(self):
        from sklearn.linear_model import LogisticRegressionCV
        log_reg = LogisticRegressionCV(cv=self.cv)
        X = self.train_input_data.drop(self.label_col,axis=1)
        y = self.train_input_data[self.label_col]
        log_reg.fit(X,y)
        self.Y_pred= log_reg.predict(X)
        self.Y_pred_proba = log_reg.predict_proba(X)
        self.logit = log_reg
        # extract the positive columns for each output
        print("Accuracy : ",accuracy_score(y,self.Y_pred))
        print("roc_auc score : ",roc_auc_score(y, self.Y_pred_proba[:,1]))
        # fpr, tpr, thresholds = roc_curve(self.test_labels,self.Y_preds_test, pos_label=2)
        # print(auc(fpr, tpr))
        # self.Y_preds = Y_preds
        return

    def build_logistic_regression_using_statsmodels(self):
        self.train_features[label_col] = self.train_labels

        # logit = LogisticRegression(solver='lbfgs')
        # logit.fit(x_train,y_train)0:
        # logit = smf.logit("TARGET ~ ind_var12_0 + saldo_medio_var5_hace2", data=self.train_features).fit()
        all_features = [
            column for column in self.train_features.columns if column != self.label_col]
        formula_all_columns = "+".join(all_features)
        logit = smf.logit(f"TARGET ~  {formula_all_columns}", data=self.train_features).fit(
            method='bfgs', maxiter=1000)
        self.logit = logit
        self.Y_preds_prob_test = logit.predict(self.test_features)
        self.Y_preds_prob_train = logit.predict(self.train_features)

        self.Y_preds_train = self.Y_preds_prob_train.map(
            lambda x: 1 if x > self.logistic_threshold else 0)
        self.Y_preds_test = self.Y_preds_prob_test.map(
            lambda x: 1 if x > self.logistic_threshold else 0)

    @staticmethod
    def binary_classification_performance(y_ground_truth, y_pred, y_pred_prob, type):
        result = ClassificationMetrics(type=type)
        result.tp, result.fp, result.fn, result.tn = confusion_matrix(
            y_ground_truth, y_pred).ravel()
        result.accuracy = round(accuracy_score(
            y_pred=y_pred, y_true=y_ground_truth), 2)

        result.precision = round(precision_score(
            y_pred=y_pred, y_true=y_ground_truth), 2)
        result.recall = round(recall_score(
            y_pred=y_pred, y_true=y_ground_truth), 2)
        result.f1_score = round(
            2 * result.precision * result.recall / (result.precision + result.recall), 2)
        result.specificity = round(result.tn / (result.tn + result.fp), 2)
        result.npv = round(result.tn / (result.tn + result.fn), 2)
        # average_precision_score(y_test, y_score)
        # except:
        #     print("Unable to generate Precision metrics,perhaps because no predictions as 1")
        # here we need to pass the probability of the positive class
        result.areaUnderROC = round(roc_auc_score(
            y_score=y_pred_prob, y_true=y_ground_truth), 2)

        result.FPR, result.TPR, _ = roc_curve(
            y_ground_truth, y_pred_prob)

        result.precision_plot_data, result.recall_plot_data, _ = precision_recall_curve(y_ground_truth, y_pred_prob, pos_label=1)

        result.areaUnderPR = round(
            auc(result.recall_plot_data, result.precision_plot_data), 2)
        fig = plot_precision_recall(result.recall_plot_data, result.precision_plot_data, result.areaUnderPR)
        return result

    def aggregate_test_accuracy_results(self, type):
        # https://towardsdatascience.com/calculating-and-setting-thresholds-to-optimise-logistic-regression-performance-c77e6d112d7e#:~:text=The%20logistic%20regression%20assigns%20each,0.5%20is%20the%20default%20threshold.
        pred_labels = getattr(self, 'Y_preds_' + type)
        pred_scores = getattr(self, 'Y_preds_prob_' + type)[:, 1]
        ground_truth_labels = getattr(self, type + "_labels")

        result = LogisticRegressionModel_sklearn.binary_classification_performance(
            ground_truth_labels, pred_labels, pred_scores, type=type)

        setattr(self, type + '_result', result)

    def get_coefficients_names(self):
        # Source : https://turingintern2018.github.io/sparkairplane.html
        self.overall_result.coefficients = json.dumps(
            list(self.logit.coef_[0, :]))
        self.overall_result.intercept = self.logit.intercept_[0]
        self.overall_result.features = json.dumps(
            list(self.train_features.columns))

    def get_intercept(self):
        if self.logit.intercept_[0]:
            self.overall_result.intercept = self.logit.intercept_[0]

    def plot_beta_coeficients(self):
        beta = np.sort(self.lrModel.coefficients)
        plt.plot(beta)
        plt.ylabel('Beta Coefficients')
        # plt.show()

    def predict(self, data):
        self.result.predictions = self.lrModel.transform(data)
        # self.predictions.select(prediction_cols).show(10)/
# %%

# user = django_user_model.objects.create_user(username=username, password=password)
#     santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')
#     label_col = "TARGET"
#     feature_cols = ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1']
#     print(run_in_the_background)

#     c = Classificationmodel.objects.create(traindata=santander_data, name='new', feature_cols=json.dumps(feature_cols), experiment_type='classificationmodel', label_col=label_col, run_now=True, run_in_the_background = run_in_the_background, created_by=user, train_split=0.7, test_split=0.3, enable_spark=False, save_train_test_data=save_train_test_data,test_flag=True)





if __name__ == '__main__':
    santandar_data = pd.read_csv("sample_py_df.csv")
    fp = 'sample_py_df.csv'
    feature_cols = ['M221', 'MOB']
    lgm = LogisticRegressionModel_sklearn(fp, 'def_trig', feature_cols=feature_cols,train_percent=0.8,test_percent=0.2, cross_validation=3)