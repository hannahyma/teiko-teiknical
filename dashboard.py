# dashboard_app.py

import sqlite3
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import plotly.express as px
from dash import Dash, dcc, html, dash_table

DB_PATH = "cell-counts.db"


# -----------------------------
# Helpers
# -----------------------------
def fetch_df(query: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def df_to_datatable(df: pd.DataFrame, page_size: int = 10) -> dash_table.DataTable:
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "6px"},
        style_header={"fontWeight": "bold"},
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
    )


def compute_results(melanoma_miraclib: pd.DataFrame, non_responders: pd.DataFrame) -> pd.DataFrame:
    pops = sorted(set(melanoma_miraclib["population"]).intersection(non_responders["population"]))
    rows = []

    for pop in pops:
        x = melanoma_miraclib.loc[melanoma_miraclib["population"] == pop, "percentage"].dropna().to_numpy()
        y = non_responders.loc[non_responders["population"] == pop, "percentage"].dropna().to_numpy()

        if len(x) < 2 or len(y) < 2:
            rows.append(
                {
                    "population": pop,
                    "n_responders": len(x),
                    "n_nonresponders": len(y),
                    "median_responders": np.median(x) if len(x) else np.nan,
                    "median_nonresponders": np.median(y) if len(y) else np.nan,
                    "U": np.nan,
                    "p": np.nan,
                }
            )
            continue

        U, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        rows.append(
            {
                "population": pop,
                "n_responders": len(x),
                "n_nonresponders": len(y),
                "median_responders": float(np.median(x)),
                "median_nonresponders": float(np.median(y)),
                "U": float(U),
                "p": float(p),
            }
        )

    results = pd.DataFrame(rows)

    # BH-FDR correction
    mask = results["p"].notna()
    results.loc[mask, "q"] = multipletests(results.loc[mask, "p"], method="fdr_bh")[1]
    results["significant_q05"] = results["q"] < 0.05

    # median difference
    results["median_diff"] = results["median_responders"] - results["median_nonresponders"]

    # sort
    results = results.sort_values(["q", "p"], na_position="last").reset_index(drop=True)
    return results


# -----------------------------
# Load / compute "Parts"
# -----------------------------

# Part 2: summary table of relative_frequencies dataframe
relative_frequencies = fetch_df("SELECT * FROM relative_frequencies;")

def pop_describe(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (
        df.groupby("population")[col]
        .describe()
        .reset_index()
        .rename(columns={"count": "n"})  # describe()'s count -> n
    )
    return out

relfreq_summary_total = pop_describe(relative_frequencies, "total_count")
relfreq_summary_count = pop_describe(relative_frequencies, "count")
relfreq_summary_pct   = pop_describe(relative_frequencies, "percentage")


# Part 3: data subsets + plotly boxplots + results table + description
melanoma_miraclib = fetch_df(
    """
    SELECT rf.*
    FROM relative_frequencies rf
    JOIN sample_data sd ON rf.sample = sd.sample
    JOIN subject_data subj ON sd.subject = subj.subject
    WHERE subj.treatment = 'miraclib'
      AND subj.response = 'yes'
      AND sd.sample_type = 'PBMC';
    """
)

non_responders = fetch_df(
    """
    SELECT rf.*
    FROM relative_frequencies rf
    JOIN sample_data sd ON rf.sample = sd.sample
    JOIN subject_data subj ON sd.subject = subj.subject
    WHERE subj.response = 'no'
      AND sd.sample_type = 'PBMC';
    """
)

results = compute_results(melanoma_miraclib, non_responders)

fig_non = px.box(
    non_responders,
    x="population",
    y="percentage",
    title="Amongst Non-Responders",
)
fig_non.update_yaxes(ticksuffix="%")
fig_non.update_layout(margin=dict(l=10, r=10, t=50, b=10))

fig_mira = px.box(
    melanoma_miraclib,
    x="population",
    y="percentage",
    title="Amongst Melanoma Patients Receiving Miraclib (Responders)",
)
fig_mira.update_yaxes(ticksuffix="%")
fig_mira.update_layout(margin=dict(l=10, r=10, t=50, b=10))

description = """ Because relative immune cell frequencies are bounded, \
               non-normally distributed, heteroskedastic, and measured \
               in independent groups with moderate sample sizes, a nonparametric \
               Wilcoxon rank-sum test was used in place of a Student's t-test. \
               This approach avoids violations of normality and variance assumptions \
               while providing a robust comparison of distributions between responders \
               and non-responders.\
               Multiple hypothesis testing was corrected using the Benjamini–Hochberg false discovery rate procedure. \
               
               Using this method, it was found that CD4 T cells showed a significantly \
               higher relative frequency in responders compared to non-responders\
               (median = 30.3 vs 29.6%, Wilcoxon p = 0.000034, FDR-adjusted q = 0.000171).\
               B cells (p=0.000261, q=0.000652) were significantly enriched in responders and \
               monocytes in non-responders (p=0.015165, q=0.025274).\
               No significant differences were observed for CD8 T cells or NK cells after correction (q > 0.05). """ 


# Part 4: subset + 3 derived tables, with toggle
subset = fetch_df(
    """
    SELECT subj.*, sd.sample
    FROM subject_data subj
    JOIN sample_data sd ON subj.subject = sd.subject
    WHERE subj.condition = 'melanoma'
      AND subj.treatment = 'miraclib'
      AND sd.sample_type = 'PBMC'
      AND sd.time_from_treatment_start = 0;
    """
)

# Your script produced Series; convert to dataframes for display
samples_by_project = subset.groupby("project").size().reset_index(name="n_samples")
subject_response = subset.groupby("response").size().reset_index(name="n_subjects")
subject_sex = subset.groupby("sex").size().reset_index(name="n_subjects")


# -----------------------------
# Build dashboard
# -----------------------------
app = Dash(__name__)
app.title = "Bob's Cell Count Analysis Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "20px auto", "fontFamily": "Arial"},
    children=[
        html.H1("Bob's Cell Count Analysis Dashboard"),

        dcc.Tabs(
            value="part2",
            children=[
                # ---------------- Part 2 ----------------
                dcc.Tab(
                    label="Part 2 — Relative Frequencies Summary",
                    value="part2",
                    children=[
                        html.H2("Part 2: Summary Statistics by Cell Population"),
                        html.P("Toggle between summary views for total_count, count, and percentage."),

                        dcc.Tabs(
                            value="sum_total",
                            children=[
                                dcc.Tab(
                                    label="total_count",
                                    value="sum_total",
                                    children=[df_to_datatable(relfreq_summary_total, page_size=12)],
                                ),
                                dcc.Tab(
                                    label="count",
                                    value="sum_count",
                                    children=[df_to_datatable(relfreq_summary_count, page_size=12)],
                                ),
                                dcc.Tab(
                                    label="percentage",
                                    value="sum_pct",
                                    children=[df_to_datatable(relfreq_summary_pct, page_size=12)],
                                ),
                            ],
                        ),
                    ]
                ),

                # ---------------- Part 3 ----------------
                dcc.Tab(
                    label="Part 3 — Boxplots + Stats",
                    value="part3",
                    children=[
                        html.H2("Part 3: Cell Population Relative Frequencies Between Responders vs Non-Responders"),

                        html.Div(
                            style={"display": "flex", "gap": "16px"},
                            children=[
                                html.Div(style={"flex": "1"}, children=[dcc.Graph(figure=fig_non)]),
                                html.Div(style={"flex": "1"}, children=[dcc.Graph(figure=fig_mira)]),
                            ],
                        ),

                        html.H3("Statistical Results (Mann–Whitney U + BH-FDR)"),
                        df_to_datatable(results, page_size=10),

                        html.Div(
                            style={
                                "marginTop": "16px",
                                "padding": "12px",
                                "border": "1px solid #ddd",
                                "borderRadius": "8px",
                                "background": "#fafafa",
                            },
                            children=[
                                html.H4("Results"),
                                html.Div(description),
                            ],
                        ),
                    ],
                ),

                # ---------------- Part 4 ----------------
                dcc.Tab(
                    label="Part 4 — Subset Tables",
                    value="part4",
                    children=[
                        html.H2("Part 4: Data Subset Analysis"),
                        html.P("Use the tabs below to toggle between the tables."),

                        dcc.Tabs(
                            value="subset",
                            children=[
                                dcc.Tab(
                                    label="subset",
                                    value="subset",
                                    children=[
                                        html.H3("subset"),
                                        df_to_datatable(subset, page_size=10),
                                    ],
                                ),
                                dcc.Tab(
                                    label="samples_by_project",
                                    value="samples_by_project",
                                    children=[
                                        html.H3("samples_by_project"),
                                        df_to_datatable(samples_by_project, page_size=10),
                                    ],
                                ),
                                dcc.Tab(
                                    label="subject_response",
                                    value="subject_response",
                                    children=[
                                        html.H3("subject_response"),
                                        df_to_datatable(subject_response, page_size=10),
                                    ],
                                ),
                                dcc.Tab(
                                    label="subject_sex",
                                    value="subject_sex",
                                    children=[
                                        html.H3("subject_sex"),
                                        df_to_datatable(subject_sex, page_size=10),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

if __name__ == "__main__":
    # Visit: http://127.0.0.1:8050
    app.run(debug=True)


