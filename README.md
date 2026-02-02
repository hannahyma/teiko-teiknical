This repository contains all work completed for the Teiko Teiknical assessment by Hannah Ma.

Code structure:
* `cell-counts.csv` - raw data provided for the assessment
* `cell-counts.db` - SQLite database created as Part 1 of the analysis.
* `dashboard.py` - Plotly dashboard containing visualizations and summaries of Part 2-4 of Bob's analysis.
* `program.ipynb` - Python program written for Parts 1-4 of the analysis, without dashboard considerations.
* `README.md` - This README

Dependencies:
numpy           2.3.3
pandas          2.3.3
scipy           1.17.0
statsmodels     0.14.6
plotly          6.5.2
dash            3.4.0

SQL Schema:
I decided to segment the master datasheet into subject-level and sample-level data, taking inspiration from some prior work with clinical datasets following ADaM specifications. Subject-level and sample-level data are often queried separately and independently of one another, so segmenting the data this way and joining via a subject identifier variable allows one to avoid retrieving information irrelevant to the query of interest. This reduces query and latency times in the case that a project like this were to be reproduced on a much larger scale. 

Dashboard:
The only python platform I have experience developing dashboards in is with dash by plotly, so I chose to use it in this project. The dashboard is designed to run locally using the `dashboard.py` script.

