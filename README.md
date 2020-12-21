# tds-kaggle-7

This repo contains my solutions to the [Challenge #25 of the Datascience.net platform](https://www.datascience.net/fr/challenge/25/details) - **October 2016**.

My best solution (`sklearn/sklearn_optim`) was ranked 25th. Part of this work has been presented at Toulouse Data Science Meetup (TDS).

## Jupyter Notebooks & Scikit-learn
For my main work, I coded Jupyter Notebooks using python and scikit-learn libs.

 - `notebooks-scikitlearn/sklearn_basic.ipynb`: simple model.
 - `notebooks-scikitlearn/sklearn_optim.ipynb`: advanced model. Ranked 25th.

## PySpark & MLLib

For fun, I recoded the solutions using Spark MLLib instead of Scikit-learn. 
Of course, since Spark MLLib is designed for distributed systems, the solutions based on Scikit-learn are more efficient in a standalone mode. But my main objectif here was to compare MLLib and Scikit-learn APIs. 
 - `pyspark/spark_basic`: Spark version of `sklearn_basic.ipynb`
 - `pyspark/spark_optim`: Optimized version of `spark_basic` (includes OHE).