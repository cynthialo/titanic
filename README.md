# titanic
Titanic: Machine Learning from Disaster (Kaggle Competition)

https://www.kaggle.com/c/titanic

This Kaggle competition asks users to "apply the tools of machine learning to predict which passengers survived the [sinking of the RMS Titanic] tragedy".  

Data science involves three main steps:
* Data curation
* Data cleaning and integration
* Data analytics

The models in this repository include:
* Logistic regression: [genderclasslogisticregressionmodel.py](genderclasslogisticregressionmodel.py)

  A simple classification model to get started, using the passengers' socio-economic status, sex, and fare as the features.  Here, the model assumptions are: 
  * Passengers with incomplete data are removed 
  * No feature scaling is performed  
  * All features are assumed to be independent

The raw data can be downloaded directly from Kaggle.
