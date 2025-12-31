Project Overview
This project develops a predictive model and an analytical framework to assist archaeologists in discovering new sites. It integrates various geospatial data features (elevation, slope, soil type, proximity to water, historical records, satellite imagery features) to predict the probability of archaeological site presence. The system also includes modules for spatial analysis, advanced machine learning, and interactive visualization of high-potential areas.

Key Features and Modules
Data Processing (ArchaeologicalDataProcessor):

Synthetic Data Generation: Initially generates synthetic data to simulate a diverse range of geospatial features and site presence for development and demonstration.
Real Data Integration (Placeholder): Designed to load and preprocess real-world geospatial datasets including Digital Elevation Models (DEM), water bodies, soil maps, and known archaeological sites (requires actual files).
Feature Engineering: Extracts and creates relevant features such as elevation, slope, distance to water, soil type, historical indicators, and spatial cluster information.
Exploratory Data Analysis (FeatureAnalyzer):

Analyzes correlations between features and archaeological site presence.
Performs spatial analysis to visualize site distribution patterns relative to environmental factors like elevation, distance to water, and slope.
Machine Learning Prediction (ArchaeologicalSitePredictor):

Trains and evaluates predictive models (initially Random Forest Classifier) to forecast archaeological site presence.
Provides standard evaluation metrics including accuracy, classification reports, ROC AUC scores, and feature importance analysis.
Satellite Image Analysis (SatelliteImageAnalyzer):

Simulates satellite image patches representing archaeological and natural features.
Trains a Convolutional Neural Network (CNN) to detect subtle patterns indicative of archaeological remains in these simulated images.
Prediction Mapping (ArchaeologicalSiteMapper):

Generates a probability map highlighting areas with high potential for archaeological sites based on ML model outputs.
Classifies locations into risk categories (e.g., Very Low, Low, Medium, High, Very High) and provides summary statistics for these areas.
Survey Recommendation (SurveyRecommendationSystem):

Ranks high-potential areas based on a multi-factor survey priority score, incorporating site probability, accessibility, and potential historical significance.
Recommends optimal locations for field surveys to maximize discovery efficiency.
Model Validation (ModelValidator):

Performs spatial cross-validation to assess the model's robustness across different geographic partitions.
Conducts confidence analysis to understand the certainty and reliability of the model's predictions.
Advanced Modeling Techniques:

Explores and compares performance of advanced ensemble models like Gradient Boosting and XGBoost for improved predictive accuracy.
Spatial Modeling:

Implements Geographically Weighted Regression (GWR) to account for spatial non-stationarity, analyzing how the relationships between features and site presence vary across different locations.
Uncertainty Quantification:

Analyzes the distribution and variance of predicted probabilities, particularly within spatial clusters, to provide insights into prediction confidence and areas of higher uncertainty.
Interactive Mapping and Visualization:

Utilizes Folium to create interactive web maps displaying site probability, risk categories, and recommended survey locations, enhancing accessibility for non-programmers.
Technologies Used
Data Manipulation: pandas, numpy, geopandas, rasterio
Machine Learning: scikit-learn, xgboost, tensorflow/keras, spreg, mgwr
Visualization: matplotlib, seaborn, folium
Utility: warnings
Current Limitations & Next Steps
The project primarily uses simulated data for demonstration. Full functionality relies on providing real-world geospatial data files (DEM, water bodies, soil, known sites) for a chosen study region.
Manual acquisition and upload of these geospatial datasets are currently required due to environment limitations.
Further refinement of feature engineering and integration of actual satellite imagery analysis (beyond simulation) would enhance the model's real-world applicability.
Expanding the Streamlit UI to be fully functional with user-uploaded data or selection of predefined regions would greatly improve usability.
