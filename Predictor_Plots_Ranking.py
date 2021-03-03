#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


class Predictor_Plots_Ranking:
    def __init__(self, inp_data):
        self.df = inp_data
        if not os.path.exists("../Output-File/Final_Plots"):
            print("Creating plots")
            os.makedirs("../Output-File/Final_Plots")
        self.file_path = "../Output-File/Final_Plots/"
        self.y = self.df["Home_team_Win"]
        self.X = self.df.loc[:, self.df.columns != "Home_team_Win"]

    def generate_dist_plot(self, col, filename):
        group_labels = ["0", "1"]
        # Create distribution plot with custom bin_size
        N = self.df[self.df["Home_team_Win"] == 0][col]
        Y = self.df[self.df["Home_team_Win"] == 1][col]
        fig_1 = ff.create_distplot([N, Y], group_labels, bin_size=0.2)
        fig_1.update_layout(
            title="Continuous Predictor "
            + col
            + " by Categorical Response Home_team_Win",
            xaxis_title="Predictor",
            yaxis_title="Distribution",
        )
        fig_1.write_html(
            file=self.file_path + filename,
            include_plotlyjs="cdn",
        )

        # Generate Heatmap Categorical Predictor by Categorical Response

    def generate_heatmap(self, col, filename):
        conf_matrix = confusion_matrix(self.df[col], self.df["Home_team_Win"])
        fig_no_relationship = go.Figure(
            data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
        )
        fig_no_relationship.update_layout(
            title="Categorical Predictor" + col + " by Categorical Response ",
            xaxis_title="Response",
            yaxis_title="Predictor",
        )

        fig_no_relationship.write_html(
            file=self.file_path + filename,
            include_plotlyjs="cdn",
        )

    def get_pval_tscore(self, col):
        predictor = sm.add_constant(self.df[col])
        logit = sm.Logit(self.df["Home_team_Win"], predictor)
        logit_fitted = logit.fit()
        # Get the stats
        t_value = round(logit_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(logit_fitted.pvalues[1])
        fn = "ranking_" + col + ".html"
        filename = self.file_path + fn
        m_plot = "<a href=" + filename + ">" + fn + "</a>"
        # Plot the figure
        fig = px.scatter(x=self.df[col], y=self.df["Home_team_Win"], trendline="ols")
        fig.update_layout(
            title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {col}",
            yaxis_title="y",
        )
        # fig.show()
        fig.write_html(file=filename, include_plotlyjs="cdn")
        return t_value, p_value, m_plot

    def calculate_mean_uw_w(self, col, pop_prop_1):
        # d3 = pd.DataFrame({}, index=[])
        d2 = self.calculate_mean_of_response(col, pop_prop_1)
        d2.columns = ["Bucket", "BinCount", "BinMean", "Mean"]
        # d3["Mean"] = d2.mean().X
        # d3["LowerBin"] = d2.min().X
        # d3["UpperBin"] = d2.max().X
        # d3["COUNT"] = d2.count().Y
        d2["Pop_mean"] = pop_prop_1
        pop_prop = d2.BinCount / len(self.df)
        # d3["BinMean"] = d2.mean().Y
        # d2["Mean"] = d2["X"].mean()
        d2["Mean_sq_diff"] = (d2.BinMean - pop_prop_1) ** 2
        d2["Mean_sq_diffW"] = d2.Mean_sq_diff * pop_prop
        filename = self.get_dif_mean_response_plots(d2, col, pop_prop_1)
        return d2["Mean_sq_diff"].sum(), d2["Mean_sq_diffW"].sum(), filename

    def calculate_mean_of_response(self, col, pop_prop_1):
        n = 8
        d1 = pd.DataFrame(
            {
                "X": self.df[col],
                "Y": self.df["Home_team_Win"],
                "Bucket": pd.cut(self.df[col], n, labels=False),
            }
        )

        # d2 = d1.groupby(["X", "Bucket"]).agg({"Y": ["count", "mean"]}).reset_index()
        d2 = (
            d1.groupby(["Bucket"])
            .agg({"Y": ["count", "mean"], "X": "mean"})
            .reset_index()
        )
        return d2

    def calculate_mean_of_response_cat(self, col, pop_prop_1):
        # bins = len(np.unique(df[col].values))
        d1 = pd.DataFrame({"X": self.df[col], "Y": self.df["Home_team_Win"]}).groupby(
            self.df[col]
        )
        return self.calculate_mean_uw_w(d1, col, pop_prop_1)

    def get_dif_mean_response_plots(self, df, col, pop_prop_1):
        plt.clf()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(x=df["Mean"], y=df["BinCount"], name="Population"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df["Mean"], y=df["BinMean"], line=dict(color="red"), name="BinMean"
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Mean"],
                y=df["Pop_mean"],
                line=dict(color="green"),
                name="PopulationMean",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            height=600, width=800, title_text="Diff in Mean with Response" + col
        )
        filename = self.file_path + "Diff_in_mean_with_response_" + col + ".html"
        fig.write_html(
            file=filename,
            include_plotlyjs="cdn",
        )
        return filename

    def plot_feature_importance(self, importance, names, model_type):

        # Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        # Create a DataFrame using a Dictionary
        data = {
            "feature_names": feature_names,
            "feature_importance": feature_importance,
        }
        fi_df = pd.DataFrame(data)

        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

        # Define size of bar plot
        plt.figure(figsize=(10, 8))
        # Plot Searborn bar chart
        sns_plot = sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
        # Add chart labels
        plt.title(model_type + " Feature Importance")
        plt.xlabel("FEATURE IMPORTANCE")
        plt.ylabel("FEATURE NAMES")
        fig1 = sns_plot.get_figure()
        fig1.savefig(
            "../Output-File/Final_Plots/" + model_type + "_Feature_Importance.png"
        )
        plt.clf()
        with open("../Output-File/Random_Forest_Feature_Importance.html", "w") as _file:
            _file.write(
                "<h1>Random Forest Feature Importance </h1>\n"
                + "<img src='./Final_Plots/Random_Forest_Feature_Importance.png'"
                + "alt='Random Forest Feature Importance'>"
            )

    def get_feature_importance(self):
        rf = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=1)
        rf.fit(self.X, self.y)
        importance = rf.feature_importances_
        self.plot_feature_importance(importance, self.X.columns, "Random_Forest")

        return importance

    def get_feature_plots(self):
        f_path = []
        p_val = []
        t_val = []
        m_plot = []
        msd_uw = []
        msd_w = []
        plot_lk = []
        col_names = [
            "Predictor",
            "Categorical/Continuous",
            "Predictor_Plot_Link",
            "t-value",
            "p-value",
            "t/p-value-plot",
            "RandomForestVarImp",
            "MeanSqDiff",
            "MeanSqDiffWeighted",
            "MeanSqDiffWPlots",
        ]
        output_df = pd.DataFrame(columns=col_names)
        output_df["Predictor"] = self.X.columns
        output_df["RandomForestVarImp"] = self.get_feature_importance()

        for col in self.X.columns:
            output_df["Categorical/Continuous"] = "Continuous"
            tval, pval, plot = self.get_pval_tscore(col)
            t_val.append(tval)
            p_val.append(pval)
            m_plot.append(plot)
            pop_prop_1 = self.df.Home_team_Win.sum() / len(self.df)

            if col != "city":
                filename = "cat_response_cont_predictor_dist_plot_" + col + ".html"
                self.generate_dist_plot(col, filename)

                uw, w, mr_fn = self.calculate_mean_uw_w(col, pop_prop_1)
            else:
                filename = "cat_response_cat_predictor_heatmap_" + col + ".html"
                self.generate_heatmap(col, filename)

                uw, w, mr_fn = self.calculate_mean_uw_w(col, pop_prop_1)
            f_path.append(
                "<a href=" + self.file_path + filename + ">" + filename + "</a>"
            )
            msd_uw.append(uw)
            msd_w.append(w)
            plot_lk.append("<a href= " + mr_fn + "> Plot Link </a>")

        output_df["Predictor_Plot_Link"] = f_path
        output_df["t-value"] = t_val
        output_df["p-value"] = p_val
        output_df["t/p-value-plot"] = m_plot
        output_df["MeanSqDiff"] = msd_uw
        output_df["MeanSqDiffWeighted"] = msd_w
        output_df["MeanSqDiffWPlots"] = plot_lk
        return output_df
