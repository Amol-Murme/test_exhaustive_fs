# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

import os
from pathlib import Path
from django.conf import settings
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,recall_score,precision_score
from mlxtend.feature_selection import SequentialFeatureSelector
# %%
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from typing import List
from typing import Union


class TopModels_internal(BaseModel):
    selected_features: Union[None, List[str]] = None
    cv_scores: Union[None, List[float]] = None
    avg_score: Union[None, float] = None
    rank: Union[None, int] = None


class ResultsFeatureselection_internal(BaseModel):
    constant_features: Union[None, List[str]] = None
    quasi_constant_features: Union[None, List[str]] = None
    non_numeric_columns: Union[None, List[str]] = None
    correlated_features: Union[None, List[str]] = None
    duplicated_features: Union[None, List[str]] = None
    shortlisted_features: Union[None, List[str]] = None

# %%


class FeatureSelection():
    MAX_SHORT_LIST_FEATURES = 15

    def __init__(self, traindata_path,  label_col, feature_cols, short_list_max_features, max_features, min_features, cross_validation, exclude_features, fixed_features, variance_threshold, scoring, *args, **kwargs) -> None:
        # traindata=pd.read_csv(traindata_path)

        if os.path.exists(traindata_path):
            file_path = traindata_path
        else:
            file_path = os.path.join(
                Path(settings.BASE_DIR).parent, traindata_path)
            if not os.path.exists(file_path):
                ValueError("Input file doesnt exist")
        traindata = pd.read_csv(file_path)
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(traindata.drop(
            labels=[label_col], axis=1), traindata[label_col], test_size=0.2, stratify=traindata[label_col], random_state=41)
        del traindata
        self.max_features = max_features
        self.min_features = min_features
        self.feature_cols = feature_cols
        self.exclude_features = exclude_features
        self.fixed_features = fixed_features
        self.short_list_max_features = short_list_max_features
        self.cv = cross_validation
        self.models = None
        self.scoring = scoring
        self.ResultsFeatureselection_internal = ResultsFeatureselection_internal()
        self.remove_constant_columns()
        if variance_threshold:
            self.variance_threshold = variance_threshold
            self.remove_quasi_constant_columns(
                variance_threshold=variance_threshold)
        # self.remove_duplicate_columns()
        self.keep_numeric_columns_only()
        self.remove_correlated_columns()
        self.treat_nas()
        self.random_forest_feature_selector()
        self.exhaustive_model_builder()
        self.logistic_regression()

    def remove_constant_columns(self):
        constant_filter = VarianceThreshold(threshold=0)
        constant_filter.fit(self.train_features)
        constant_columns = [
            column for column in self.train_features.columns if column not in self.train_features.columns[constant_filter.get_support()]]
        self.train_features.drop(labels=constant_columns, axis=1, inplace=True)
        self.ResultsFeatureselection_internal.constant_features = constant_columns

    def remove_quasi_constant_columns(self, variance_threshold=0.01):
        qconstant_filter = VarianceThreshold(threshold=variance_threshold)

        qconstant_filter.fit(self.train_features)

        qconstant_columns = [
            column for column in self.train_features.columns if column not in self.train_features.columns[qconstant_filter.get_support()]]
        self.train_features.drop(
            labels=qconstant_columns, axis=1, inplace=True)
        self.ResultsFeatureselection_internal.quasi_constant_features = qconstant_columns

    def keep_numeric_columns_only(self):
        num_colums = ['int16', 'int32', 'int64',
                      'float16', 'float32', 'float64']
        numerical_columns = list(
            self.train_features.select_dtypes(include=num_colums).columns)
        self.train_features = self.train_features[numerical_columns]
        non_numeric_columns = [c for c in self.train_features.columns if c not in numerical_columns]
        self.ResultsFeatureselection_internal.non_numeric_columns = non_numeric_columns

    def treat_nas(self):
        self.train_features.fillna(0, inplace=True)

    def remove_duplicate_columns(self):
        train_features_T = self.train_features.T
        train_features_T.shape
        print(train_features_T.duplicated().sum())
        unique_features = train_features_T.drop_duplicates(keep='first').T
        duplicated_features = [
            dup_col for dup_col in self.train_features.columns if dup_col not in unique_features.columns]
        duplicated_features
        self.ResultsFeatureselection_internal.duplicated_features = duplicated_features

    def remove_correlated_columns(self):
        correlated_features = set()
        correlation_matrix = self.train_features.corr()

        # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on?
        for i in range(len(correlation_matrix .columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.8 and (correlation_matrix.columns[j] not in correlated_features):
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)

        self.train_features.drop(
            labels=correlated_features, inplace=True, axis=1)
        self.ResultsFeatureselection_internal.correlated_features = list(correlated_features)

    def random_forest_feature_selector(self):

        import math
        if self.short_list_max_features < self.max_features:
            self.short_list_features = self.max_features
        if self.short_list_max_features:
            n_estimators = min(self.short_list_max_features,
                               FeatureSelection.MAX_SHORT_LIST_FEATURES)
        else:
            n_estimators = int(math.sqrt(len(self.train_features.columns)))
        feature_selector = SequentialFeatureSelector(LogisticRegression(random_state=123), k_features=self.short_list_max_features, forward=True, verbose=2, scoring=self.scoring, cv=self.cv, n_jobs=1)
        self.short_list_features = feature_selector.fit(
            self.train_features, self.train_labels)
        # takes the best one where k=number of features, this may not be the best one as
        self.filtered_features_list = list(
            self.train_features.columns[list(self.short_list_features.k_feature_idx_)])
        # there may be models where the no of variables are lesser. See https://github.com/rasbt/mlxtend/issues/41 for discussion
        # For now, doesnt matter as this is merely being used to create a short list of features
        self.ResultsFeatureselection_internal.shortlisted_features = self.filtered_features_list

    def exhaustive_model_builder(self):
        #  this will generate all combinations of variables in an exhaustive search, will take time for large datasets
        model_type = LogisticRegression(n_jobs=-1)
        feature_selector = ExhaustiveFeatureSelector(
            model_type, min_features=self.min_features, max_features=self.max_features, scoring=self.scoring, print_progress=True, cv=self.cv)
        self.models = feature_selector.fit(
            self.train_features[self.filtered_features_list], self.train_labels)
        # This contains results of all combinations of models
        models_ = self.models.subsets_.values()
        self.models_all = sorted(models_, key=lambda d: d['avg_score'], reverse=True)[0:4]

        self.top_models_list = []
        for i, m in enumerate(self.models_all):
            tm = TopModels_internal(cv_scores=list(m['cv_scores']), avg_score=m['avg_score'], rank=i, selected_features=list(m['feature_names']))
            self.top_models_list.append(tm)

    def logistic_regression(self):
        
        log_reg = LogisticRegressionCV(cv=5,n_jobs=-1)

        log_reg.fit(self.train_features[['M221', 'MOB']], self.train_labels)
        self.Y_preds_test = log_reg.predict(self.test_features[['M221', 'MOB']])
        self.Y_preds_train = log_reg.predict(self.train_features[['M221', 'MOB']])
        self.Y_preds_prob_test = log_reg.predict_proba(self.test_features[['M221', 'MOB']])
        self.Y_preds_prob_train = log_reg.predict_proba(self.train_features[['M221', 'MOB']])
        self.logit = log_reg
        # extract the positive columns for each output
        print('accuracy : ',accuracy_score(self.test_labels,self.Y_preds_test))
        print('roc_auc :',roc_auc_score(self.test_labels, self.Y_preds_prob_test[:,1]))
        print('roc_auc :',roc_auc_score(self.train_labels, self.Y_preds_prob_train[:,1]))
        print('f1 : ',f1_score(self.test_labels,self.Y_preds_test))

# %%
if __name__ == '__main__':
    santandar_data = pd.read_csv("sample_py_df.csv")
    fp = 'sample_py_df.csv'

    mfs = FeatureSelection(fp, 'def_trig', max_features=2, feature_cols=None, cross_validation=0, exclude_features=None, fixed_features=None, variance_threshold=0.01, scoring='roc_auc',
                           min_features=2, cv=4, short_list_max_features=4)
    print(mfs.models.subsets_)
    print(mfs.models)
    print(mfs.models.best_feature_names_)
    print(mfs.models.best_score_)

# %%
