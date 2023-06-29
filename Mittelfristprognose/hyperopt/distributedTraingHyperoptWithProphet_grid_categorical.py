# Databricks notebook source
# MAGIC %md
# MAGIC ## Intro
# MAGIC
# MAGIC The main purpose of this notebook is to show how to use Spark to perform distributed training of ML models on the cluster (here we'll train a few Prophet models). The notebook also shows how to set up Hyperopt to orchestrate the hyperparameter search. Hyperopt is a general purpose library that can be used for the optimisation of any function that has parameters. Here it is used to optimise the prediction loss function (rmse). Additionally, some metrics will be tracked using MLflow. 
# MAGIC
# MAGIC We will to work with daily aggregated energy consumption data becausem the data shows a nice weekly and annual seasonality.

# COMMAND ----------

# MAGIC %pip install prophet==1.1.4 hyperopt==0.2.7 sklearn mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid

from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK
from prophet import Prophet

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow

# COMMAND ----------

# You must create the expirement by hand in the Databricks Machine Learning Experiments GUI
# Then copy the name of the experiment and paste it here

experimentPath = "/Users/simon.buse@ewz.ch/MittelfristPrognoseTest"

if mlflow.get_experiment_by_name(experimentPath) != None:
    print(f"Experiment {experimentPath} exists, setting it as the active experiemnt")
    mlflow.set_experiment(experimentPath)
else:
    raise Exception(
        "You must first create the experiment in the Databricks Machine Learning GUI"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Functions

# COMMAND ----------

def logMetrics(test, predictions):
    """
    Simple function to compute MAPE and RMSE metrics from test predictions and the test data for our models.
    The error metrics will be logged with MLflow.
    """
    mapeValue = mean_absolute_percentage_error(test, predictions)
    rmseValue = mean_squared_error(test, predictions, squared=False)
    metrics = {"mape": mapeValue, "rmse": rmseValue}

    # MLflow will log the metrics to the currently active run
    mlflow.log_metrics(metrics)


def logParams(parameters):
    """
    Parameters should be a dictionary with the form {"parameterName": "parameterValue"}
    """
    for par in parameters:
        mlflow.log_params(par)


def resample_fix_ends(pdf, frequency):
    """
    The function resamples the data according to the sampling frequency.
    Often the first and the last data-point are deviating after a resampling and as a simple fix i will just delete
    the first and the last value if they deviate more than 20% from their neighbour.
    """

    pdf = pdf.resample(frequency).sum(min_count=1)  # "D,W,M"

    for column in pdf.columns:
        if pdf[column].iloc[0] < 0.8 * pdf[column].iloc[1]:
            pdf = pdf.drop(pdf.index[0])

        if pdf[column].iloc[-1] < 0.8 * pdf[column].iloc[-2]:
            pdf = pdf.drop(pdf.index[-1])

    return pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the Data

# COMMAND ----------

url = "https://data.stadt-zuerich.ch/dataset/ewz_stromabgabe_netzebenen_stadt_zuerich/download/ewz_stromabgabe_netzebenen_stadt_zuerich.csv"
pdf = pd.read_csv(url, index_col=None)

pdf["Timestamp"] = pd.to_datetime(pdf["Timestamp"], utc=True)

#set timestamp as index to do a daily sampling
pdf = pdf.set_index(pdf["Timestamp"])  
pdf = resample_fix_ends(pdf, "D")

#Drop the timezone to avoid warnings
pdf.index = pdf.index.tz_localize(None)  

#rename the columns into y and ds. needed by prophet
pdf["ds"] = pdf.index
pdf["y"] = pdf["Value_NE5"].values + pdf["Value_NE7"].values
pdf = pdf.drop(columns=["Value_NE5", "Value_NE7"])

# put aside some data for evaluation
split = int(len(pdf) * 0.9)
pdfTrain, pdfTest = pdf.iloc[:split], pdf.iloc[split:]
pdfTrain

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Model and the Search Space

# COMMAND ----------

def trainProphet(pdfTrain, pdfTest, trackedHyperparams, maxEvals=10, timeoutSec=5*60):
    """This function is just a wrapper for the hyperopt procedure."""

    def train(params):
        """
        This is our main training function which we pass to Hyperopt.
        It takes in hyperparameter settings, fits a model based on those settings,
        evaluates the model, and returns the loss.
        """


        forecaster = Prophet(
            seasonality_mode=trackedHyperparams["seasonality_mode"],
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            holidays_prior_scale=params["holidays_prior_scale"],
            changepoint_range=params["changepoint_range"],
        )

        if trackedHyperparams["holidays"] != None:
            forecaster.add_country_holidays(country_name=trackedHyperparams["holidays"])

        forecaster.fit(pdfTrain)
        ypred = forecaster.predict(pdfTest)

        rmse = mean_squared_error(
            y_true=pdfTest.y.values, y_pred=ypred.yhat.values, squared=False
        )

        return {"loss": rmse, "status": STATUS_OK, "Trained_Model": forecaster}

    # Define the search space for Hyperopt. Prophets main parameters where found here
    # https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning

    search_space = {"changepoint_prior_scale":hp.loguniform("changepoint_prior_scale", -6.9, -0.69),  # according to recom. same as [0.001,0.5]
        "seasonality_prior_scale":hp.loguniform("seasonality_prior_scale", -6.9, 2.3),  # according to recom. same as [0.001, 10]
        "holidays_prior_scale":hp.loguniform("holidays_prior_scale", -6.9, 2.3),  # according to recom. same as [0.001, 10]
        "changepoint_range":hp.uniform("changepoint_range", 0.8, 0.95),  # optional according to docs, default = 0.8
    }

    # Select a search algorithm for Hyperopt to use.
    algo = tpe.suggest  # Tree of Parzen Estimators, a Bayesian method

    # Distribute tuning across our Spark cluster
    spark_trials = SparkTrials(parallelism=4)


    bestHyperparameters = fmin(
        fn=train,
        space=search_space,
        algo=algo,
        trials=spark_trials,
        max_evals=maxEvals,
        timeout=timeoutSec,
    )
    bestModel = spark_trials.results[np.argmin([r["loss"] for r in spark_trials.results])]["Trained_Model"]

    print(bestHyperparameters)

    return bestModel, bestHyperparameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model

# COMMAND ----------

# define which hyperparams should be tracked in MLflow e.g. track the setting of the holidays,
# additive or multiplicative, adding corona by hand or not

trackedHyperparams = {
    "seasonality_mode": ["multiplicative", "additive"],
    "corona": [False, True],
    "holidays": [None, "Switzerland"],
}
grid = ParameterGrid(trackedHyperparams)

models = []
metrics = []

# restrict total run time of the search
searchTimeSec = 15 * 60
timePerGridPoint = int(searchTimeSec / len(grid))

for evaluationParams in grid:

    # Create an MLflow run for each point in the grid space.
    # The "with" syntax means the run will be closed once the following block of code is finished.
    with mlflow.start_run(run_name="Prophet Model") as run:

        # Set a tag with the model type
        mlflow.set_tags({"Model": "Prophet"})

        # create, train and return model
        print(evaluationParams)
        model, hyperparameters = trainProphet(
            pdfTrain, pdfTest, evaluationParams, timeoutSec=timePerGridPoint
        )

        prediction = model.predict(pdfTest)

        metric = mean_squared_error(
            y_true=pdfTest.y.values, y_pred=prediction.yhat.values, squared=False
        )
        metrics.append(metric)
        models.append(model)

        # Log the metrics in MLflow
        logMetrics(pdfTest.y.values, prediction.yhat.values)

        # Log hyperparameters
        logParams([hyperparameters, evaluationParams])

bestModel = models[np.argmin(metrics)]

# COMMAND ----------

# plot the outcome of the model on the summed data.
f, axes = plt.subplots(2, 1, figsize=(18, 8))

ypred = bestModel.predict(pdfTest)

axes[0].plot(ypred.ds.values, ypred.yhat.values, color="tab:red", label="forcast")
# axes[0].plot(pdfTrain.ds.values, pdfTrain.y.values, color="tab:blue", label="train")
axes[0].plot(pdfTest.ds.values, pdfTest.y.values, color="tab:orange", label="truth", alpha=0.5)
axes[0].legend()
axes[0].set_title("NE5 + NE7")
axes[0].set_ylabel("Last [kWh]")

xmin, xmax = axes[0].get_xlim()
axes[1].plot(pdfTest.ds,(pdfTest.y.values - ypred.yhat.values)/(pdfTest.y.values)*100)
axes[1].set_xlim(xmin,xmax)
axes[1].set_ylabel("residual: True-Pred/True [%]")

plt.show()

# COMMAND ----------

# plot the induvidual components of the prophet model

forecast = bestModel.predict(pdf)
bestModel.plot(forecast)
fig = bestModel.plot_components(forecast)
