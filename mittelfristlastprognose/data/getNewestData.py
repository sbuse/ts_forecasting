import pandas as pd
import numpy as np
import sys
import warnings
import ssl

warnings.filterwarnings("ignore")

def resample_fix_ends(pdf,frequency):
  """
  The function resamples the data according to the sampling frequency. 
  Often the first and the last data-point are deviating a lot from the rest of the series.
  As a simple fix i will just delete the first and the last value if they deviate more than 20% to their neighbour. 
  """
  #if input pdf is not a pandas dataframe, try to convert it to one
  if not isinstance(pdf, pd.DataFrame):
    try: 
      pdf = pd.DataFrame(pdf)
    except: 
      raise ValueError("The input pdf can not be converted to a pandas dataframe.")
    
  pdf = pdf.resample(frequency.upper()).sum(min_count=1) #"D,W,M"
  for column in pdf.columns:
    if pdf[column].iloc[0] < 0.8*pdf[column].iloc[1]:
      pdf = pdf.drop(pdf.index[0]) 
    if pdf[column].iloc[-1] < 0.8*pdf[column].iloc[-2]:
      pdf = pdf.drop(pdf.index[-1]) 
  return pdf

def reassign_outliers(pdf):
  """
  There is an extrem outlier in the data which is probably a mistake. 
  I will reassign the value to the mean of the column.
  """
  for column in pdf.columns:
    outlier_loc = np.where(pdf[column] < np.mean(pdf[column])-3*np.std(pdf[column]))
    (pdf[column].values)[outlier_loc] = np.mean(pdf[column]) 
    #print(f"Reassigned {len(outlier_loc)} values in the column {column}. These values where more than 3 sigma away from the mean.")
  return pdf


if __name__ == "__main__":
   
  #read in flags given to this file at execution
  frequency = sys.argv[1] #expects a string like "d","m" for daily or monthly

  if frequency.lower() == "d":
      folder = "daily"
  elif frequency.lower() == "m":
      folder = "monthly"
  else:
      raise ValueError("The frequency given is not valid. Please use 'd' for daily or 'm' for monthly.")

  ssl._create_default_https_context = ssl._create_unverified_context
  url = "https://data.stadt-zuerich.ch/dataset/ewz_stromabgabe_netzebenen_stadt_zuerich/download/ewz_stromabgabe_netzebenen_stadt_zuerich.csv"
  pdf = pd.read_csv(url,index_col=None)


  pdf["Timestamp"] =  pd.to_datetime(pdf['Timestamp'],utc=True)
  pdf = pdf.set_index(pdf["Timestamp"])
  pdf = pdf.drop(columns=["Timestamp"])
  pdf = resample_fix_ends(pdf,frequency)
  pdf = reassign_outliers(pdf)

  pdf.index = pdf.index.tz_localize(None)  #Let's drop the timezone info to avoid warnings


  pdf["NE5_GWh"] = pdf["Value_NE5"].values/1e6 #in GWh
  pdf["NE7_GWh"] = pdf["Value_NE7"].values/1e6 #in GWh

  pdf = pdf.drop(columns=["Value_NE5","Value_NE7"])

  pdf.to_csv(f"{folder}/ewz_stromabgabe_netzebenen_stadt_zuerich.csv")

  print("downloaded and saved the newest data.The last data-point is from: ",pdf.index[-1].strftime("%Y-%m-%d %H:%M:%S"))