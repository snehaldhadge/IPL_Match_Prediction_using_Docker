#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plotly import graph_objects as go
from scipy import stats


class Brute_Force_Plots:
    def __init__(self, inp_data):
        self.df = inp_data
        if not os.path.exists("../Output-File/Final_Plots"):
            print("Creating plots")
            os.makedirs("../Output-File/Final_Plots")
        self.file_path = "../Output-File/Final_Plots/"
        self.y = self.df["Home_team_Win"]
        self.X = self.df.loc[:, self.df.columns != "Home_team_Win"]

    def print_heading(self, title):
        print("*" * 80)
        print(title)
        print("*" * 80)
        return

    # Calculate the correlation for cont-cont data using Pearsonr
    # It also creates the correlation matrix to display the correlation heatmap
    # It also calculates the Difference in mean weighted and unweighted
    def get_cont_cont_cor(self):

        df_cols = ["Continuous1", "Continuous2", "Correlation"]
        bf_cols = [
            "Continuous1",
            "Continuous2",
            "Correlation",
            "MeanSqDiff",
            "MeanSqDiffW",
            "PLot Link",
        ]
        d2_cont_cont = pd.DataFrame(columns=bf_cols)
        cont_cont_corr = pd.DataFrame(columns=df_cols)
        cont_cont_matrix = pd.DataFrame(index=self.X.columns, columns=self.X.columns)
        pop_prop_1 = self.y.sum() / len(self.X)
        if len(self.X.columns) > 1:
            for i in range(len(self.X.columns)):
                for j in range(i, len(self.X.columns)):
                    if self.X.columns[i] != self.X.columns[j]:
                        val, _ = stats.pearsonr(
                            self.X[self.X.columns[i]], self.X[self.X.columns[j]]
                        )
                        cont_cont_matrix.loc[self.X.columns[i]][self.X.columns[j]] = val
                        cont_cont_matrix.loc[self.X.columns[j]][self.X.columns[i]] = val
                        cont_cont_corr = cont_cont_corr.append(
                            dict(
                                zip(
                                    df_cols,
                                    [self.X.columns[i], self.X.columns[j], val],
                                )
                            ),
                            ignore_index=True,
                        )
                        w, uw, fname = self.get_w_uw_msd(
                            self.X.columns[i], self.X.columns[j], pop_prop_1, 1
                        )
                        d2_cont_cont = d2_cont_cont.append(
                            dict(
                                zip(
                                    bf_cols,
                                    [
                                        self.X.columns[i],
                                        self.X.columns[j],
                                        val,
                                        w,
                                        uw,
                                        fname,
                                    ],
                                )
                            ),
                            ignore_index=True,
                        )
                    else:
                        cont_cont_matrix[self.X.columns[i]][self.X.columns[j]] = 1.0
        return cont_cont_corr, cont_cont_matrix, d2_cont_cont

    def get_feature_correlation(self):
        # All variable are Continuous
        cont_cont_corr, cont_cont_matrix, d2_cont_cont = self.get_cont_cont_cor()
        self.print_heading("Continuous-Continuous Correlation metrics")
        cont_cont_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
        print(cont_cont_corr)
        # Part 2: Correlation Matrices
        cont_cont_matrix = cont_cont_matrix.astype(float)
        sns_plot = sns.heatmap(cont_cont_matrix, annot=True)
        fig1 = sns_plot.get_figure()
        fig1.savefig("../Output-File/Final_Plots/cont-cont-corr.png")
        plt.clf()
        # Brute-Force
        # Cont-Cont Diff of mean
        self.print_heading("Cont-Cont Brute-Force")
        d2_cont_cont = d2_cont_cont.sort_values(by="MeanSqDiffW", ascending=False)
        print(d2_cont_cont)
        with open("../Output-File/BruteForce.html", "w") as _file:
            _file.write(d2_cont_cont.to_html(render_links=True, escape=False))

    # def pct_rank_qcut(self, series, n):
    #    edges = pd.Series([float(i) / n for i in range(n + 1)])
    #    f = lambda x: (edges >= x).values.argmax()
    #    return series.rank(pct=1).apply(f)

    # Get DataFrame for Cont-Cont MSD Calculation
    def get_meansqdiff_cont_cont(self, col1, col2):
        n = 10
        d1 = pd.DataFrame(
            {
                "X1": self.df[col1],
                "X2": self.df[col2],
                "Y": self.df["Home_team_Win"],
                "Bucket1": pd.qcut(self.df[col1].rank(method="first"), n),
                "Bucket2": pd.qcut(self.df[col2].rank(method="first"), n)
                #           "Bucket1": self.pct_rank_qcut(col1, n),
                #                "Bucket2": self.pct_rank_qcut(col2, n),
            }
        )
        d2 = (
            d1.groupby(["Bucket1", "Bucket2"])
            .agg({"Y": ["count", "mean"]})
            .reset_index()
        )

        return d2

    def get_w_uw_msd(self, col1, col2, pop_prop_1, type):

        d2_c_c = self.get_meansqdiff_cont_cont(col1, col2)

        # Calculate the Bincount and Binmean and also the mean of columns
        d2_c_c.columns = [col1, col2, "BinCount", "BinMean"]
        pop_prop = d2_c_c.BinCount / len(self.df)

        # Calculate MeansqDiff weighted and unweighted
        d2_c_c["Mean_sq_diff"] = (d2_c_c["BinMean"] - pop_prop_1) ** 2
        d2_c_c["Mean_sq_diffW"] = d2_c_c.Mean_sq_diff * pop_prop

        # MSd Plot
        d_mat = d2_c_c.pivot(index=col1, columns=col2, values="Mean_sq_diffW")
        fig = go.Figure(data=[go.Surface(z=d_mat.values)])
        fig.update_layout(
            title=col1 + " " + col2 + " Plot",
            autosize=True,
            scene=dict(xaxis_title=col2, yaxis_title=col1, zaxis_title="target"),
        )
        sns_plot = sns.heatmap(d_mat, annot=True)
        fig1 = sns_plot.get_figure()
        fig1.savefig(
            "../Output-File/Final_Plots/BruteForce_Plot_" + col1 + "_" + col2 + ".png"
        )
        plt.clf()
        filename = (
            "../Output-File/Final_Plots/BruteForce_Plot_" + col1 + "_" + col2 + ".html"
        )
        fig.write_html(
            file=filename,
            include_plotlyjs="cdn",
        )

        file_n = (
            "../Output-File/Final_Plots/BruteForce_Plot_" + col1 + "_" + col2 + ".png"
        )
        fname = "<a href=" + file_n + ">Plot Link"
        return d2_c_c["Mean_sq_diff"].sum(), d2_c_c["Mean_sq_diffW"].sum(), fname
