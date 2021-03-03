#!/usr/bin/env python3
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Brute_Force_Plots import Brute_Force_Plots
from Predictor_Plots_Ranking import Predictor_Plots_Ranking
from scipy import stats
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras


class Ipl_Prediction:
    def __init__(self):
        # sql_engine = sqlalchemy.create_engine(connect_string)
        # connection = sql_engine.connect()
        # metadata = sqlalchemy.MetaData()
        # game = sqlalchemy.Table('Final_diff_stats', metadata,
        # autoload=True, autoload_with=engine)
        # query = """
        #         SELECT * FROM Final_diff_stats;
        #     """
        # query = """
        #        SELECT * FROM Final_diff_stats;
        #       """
        # self.df = pd.read_sql_query(query, sql_engine)
        # self.df = pd.read_csv("../Output-File/final_ipl_stats.csv",sep='\t')
        self.df = pd.read_csv("./Output-File/final_ipl_stats.csv", sep="\t")
        if not os.path.exists("../Output-File/Final_Plots"):
            print("Creating plots")
            os.makedirs("../Output-File/Final_Plots")
        self.file_path = "../Output-File/Final_Plots/"

    def TransformCatCols(self):

        self.df["Home_team"] = self.df["Home_team"].astype(str).astype(int)
        self.df["Away_team"] = self.df["Away_team"].astype(str).astype(int)
        # Dropping NA columns as start matches for each team have no stats to judge
        self.df.dropna(inplace=True)
        # Remove any Outliers
        z_scores = stats.zscore(self.df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        self.df = self.df[filtered_entries]

    def Split_Pred_Resp(self):
        self.y = self.df["Home_team_Win"]
        self.X = self.df.loc[:, self.df.columns != "Home_team_Win"]

    def print_heading(self, title):
        print("*" * 80)
        print(title)
        print("*" * 80)
        return

    def model_fitting(self):
        # Save the Correlation matrix for all columns
        corrMatrix = self.X.corr()
        sns_plot = sns.heatmap(corrMatrix, annot=True)
        fig1 = sns_plot.get_figure()
        fig1.savefig("../Output-File/Final_Plots/full_ftr_corr.png")
        plt.clf()
        self.X.drop(["id"], axis=1, inplace=True)
        # Normalizing the data
        normalized_X = preprocessing.normalize(self.X)
        # Split Train Test last 120 rows are the test set for Neural Network
        y_train = self.y[:550]
        X_train = normalized_X[:550]
        y_test = self.y[550:]
        X_test = normalized_X[550:]
        # X_train, X_test, y_train, y_test = train_test_split(
        #    normalized_X, self.y, test_size=0.2, random_state=100
        # )
        print("\n X-train Shape:", X_train.shape)
        print("\n X-test Shape:", X_test.shape)

        # Model Fitting using 10 Fold Cross Validation
        models = []
        models.append(("LDA", LinearDiscriminantAnalysis()))
        models.append(("KNN", KNeighborsClassifier()))
        models.append(("CART", DecisionTreeClassifier()))
        models.append(("RF", RandomForestClassifier()))
        # models.append(('NB', GaussianNB()))
        models.append(("SVM", SVC()))
        # models.append(('XGB', XGBClassifier()))

        # baseline Model with all features
        results = []
        names = []

        model_perf = []
        scoring = [
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "roc_auc",
        ]
        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = model_selection.cross_validate(
                model, normalized_X, self.y, cv=kfold, scoring=scoring
            )
            # mfit = model.fit(normalized_X, self.y)
            # y_pred = mfit.predict(X_test)
            results.append(cv_results["test_accuracy"])
            names.append(name)
        model_eval = pd.DataFrame(cv_results)
        model_eval["model"] = names
        model_perf.append(model_eval)
        df_final = pd.concat(model_perf, ignore_index=True)
        self.print_heading("Model Performance Baseline Model")
        print(df_final)

        fig = plt.figure()
        fig.suptitle("Algorithm Comparison Baseline Model")
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.savefig("../Output-File/Baseline_Model_Accuracy_Boxplot.png")

        # Using PCA
        scaler = StandardScaler()
        # Fit on training set only.
        scaler.fit(self.X)
        # Apply transform to both the training set and the test set.
        train_scaled = scaler.transform(self.X)
        # train_scaled = scaler.transform(X_train)
        # test_scaled = scaler.transform(X_test)

        # Make an instance of the Model
        pca = PCA(0.95)

        pca.fit(train_scaled)
        train_scaled = pca.transform(train_scaled)

        # test_scaled = pca.transform(test_scaled)
        evr = pca.explained_variance_ratio_
        cvr = np.cumsum(pca.explained_variance_ratio_)
        pca_df = pd.DataFrame()
        pca_df["Cumulative Variance Ratio"] = cvr
        pca_df["Explained Variance Ratio"] = evr

        pca_dims = []
        for x in range(0, len(pca_df)):
            pca_dims.append("PCA Component {}".format(x))
        pca_test_df = pd.DataFrame(
            pca.components_, columns=self.X.columns, index=pca_dims
        )
        with open("../Output-File/PCA-Variance.html", "w") as _file:
            _file.write(pca_test_df.to_html(render_links=True, escape=False))

        # Using PCA

        results = []
        names = []
        pca_model_perf = []
        model_perf = []
        scoring = [
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "roc_auc",
        ]
        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = model_selection.cross_validate(
                model, train_scaled, self.y, cv=kfold, scoring=scoring
            )
            # mfit = model.fit(train_scaled, self.y)
            # y_pred = mfit.predict(test_scaled)
            results.append(cv_results["test_accuracy"])
            names.append(name)

        pca_model_eval = pd.DataFrame(cv_results)
        pca_model_eval["model"] = names
        pca_model_perf.append(pca_model_eval)
        pca_df_final = pd.concat(pca_model_perf, ignore_index=True)
        self.print_heading("Model Performance using PCA")
        print(pca_df_final)

        fig = plt.figure()
        fig.suptitle("Algorithm Comparison")
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.savefig("../Output-File/PCA_Model_Accuracy_Boxplot.png")

        # Implementing Neural Network
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=X_train.shape[1:]))
        model.add(keras.layers.Dense(300, activation="relu"))
        model.add(keras.layers.Dense(300, activation="relu"))
        # model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
        history = model.fit(
            X_train, y_train, epochs=100, validation_data=(X_test, y_test)
        )

        # y_pred = model.predict(X_test)
        _, accuracy = model.evaluate(X_test, y_test)
        print("Accuracy of NN:", accuracy)

        fig = plt.figure()
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig("../Output-File/Neural_Network_Accuracy_Plot.png")
        with open("../Output-File/Neural_Network_Model_Performance.html", "w") as _file:
            _file.write(
                "<h1>Accuracy of Neural network: </h1>\n"
                + str(accuracy)
                + "<h1> Neural Network Accuracy Plot </h1> "
                + "<img src='../Output-File/Neural_Network_Accuracy_Plot.png'"
                + "alt='Neural Network Accuracy Plot'>"
            )

        with open("../Output-File/Baseline_Model_Performance.html", "w") as _file:
            _file.write(
                "<h1>Baseline Model Performance  </h1>\n"
                + df_final.to_html()
                + "<h1> Baseline Model Performance Plot </h1> "
                + "<img src='../Output-File/Baseline_Model_Accuracy_Boxplot.png'"
                + "alt='Baseline Model Performance Plot'>"
            )

        with open("../Output-File/PCA_Model_Performance.html", "w") as _file:
            _file.write(
                "<h1>Model Performance using PCA </h1>\n"
                + pca_df_final.to_html()
                + "<h1> Model Performance Plot using PCA</h1> "
                + "<img src='../Output-File/PCA_Model_Accuracy_Boxplot.png'"
                + "alt='Model Performance Plot using PCA'>"
            )

        with open("../Output-File/Model_Performance.html", "w") as _file:
            _file.write(
                "<p><b> Model Performance <table><tr>"
                + "<tr><td><a href= 'Baseline_Model_Performance.html'>"
                + "1. Baseline Model Performance"
                + "<tr><td><a href= 'PCA_Model_Performance.html'>"
                + "2. Model Performance Using PCA"
                + "<tr><td><a href= 'Neural_Network_Model_Performance.html'>"
                + "3. Neural network Performance "
            )

    def main(self):

        print(self.print_heading("First 5 rows of Dataset"))
        print(self.df.head())
        np.random.seed(seed=100)
        # Transform Categorical columns if any
        self.TransformCatCols()
        self.Split_Pred_Resp()

        # # Calculating random Forest for Feature Importance
        inp_data = self.df.drop(["id", "season", "Home_team", "Away_team"], axis=1)
        ftr_plot = Predictor_Plots_Ranking(inp_data)
        # output_df["RandomForestVarImp"] = ftr_plot.get_feature_importance()

        output_df = ftr_plot.get_feature_plots()
        print(output_df)
        output_df = output_df.sort_values(by="MeanSqDiffWeighted", ascending=False)
        with open("../Output-File/Feature_Importance.html", "w") as _file:
            _file.write(output_df.to_html(render_links=True, escape=False))

        bf_class = Brute_Force_Plots(inp_data)
        bf_class.get_feature_correlation()

        # Building Models
        self.model_fitting()
        with open("../Output-File/Final_Project_Output.html", "w") as _file:
            _file.write(
                "<p><b> Project Output <table><tr>"
                + "<tr><td><a href= 'Feature_Importance.html'>"
                + "1. Feature Importance"
                + "<tr><td> <a href= 'BruteForce.html'>"
                + "2. Brute-Force Plot"
                + "<tr><td> <a href= 'Model_Performance.html'>"
                + "3. Model Performance"
                + "<tr><td> <a href= './Random_Forest_Feature_Importance.html'>"
                + "4. Random Forest Feature Importance Plot"
            )


if __name__ == "__main__":
    # db_user = "root"
    # db_pass = "root"  # pragma: allowlist secret
    # db_host = "127.0.0.1"
    # db_database = "IPL"
    sys.exit(sys.exit(Ipl_Prediction().main()))
