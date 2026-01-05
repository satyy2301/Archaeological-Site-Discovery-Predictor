# 🏺 Archaeological Site Discovery Predictor

> **AI-Powered Predictive Modeling for Archaeological Site Discovery**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%2CXGBoost%2CCNN-orange)](https://scikit-learn.org)
[![Geospatial](https://img.shields.io/badge/Geospatial-Analysis-green)](https://geopandas.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📋 Overview

**Archaeological-Site-Discovery-Predictor** is an advanced predictive analytics framework designed to assist archaeologists in discovering new sites. By integrating geospatial data, machine learning, and satellite imagery analysis, 
the system identifies high-potential areas for archaeological exploration, optimizing survey efforts and resource allocation.

<img width="1596" height="867" alt="Screenshot 2026-01-05 155733" src="https://github.com/user-attachments/assets/1aa89144-5539-4b13-af64-445c35860974" />
<img width="1315" height="472" alt="Screenshot 2026-01-05 155835" src="https://github.com/user-attachments/assets/c8fc22d7-305d-41e8-8037-1c86fe90bad2" />
<img width="874" height="687" alt="Screenshot 2026-01-05 155801" src="https://github.com/user-attachments/assets/911529af-9313-478c-8cab-a7ed9fe00bcc" />
## ✨ Key Feature

### 🔍 **Data Processing & Feature Engineering**
- **Synthetic Data Generation**: Simulates diverse geospatial features for development
- **Real Data Integration**: Supports DEM, water bodies, soil maps, historical sites
- **Feature Extraction**: Elevation, slope, distance to water, soil type, spatial clusters

### 📊 **Exploratory Spatial Analysis**
- Correlation analysis between environmental factors and site presence
- Visualization of site distribution patterns
- Spatial autocorrelation assessment

### 🤖 **Machine Learning Prediction**
- **Multiple Models**: Random Forest, Gradient Boosting, XGBoost
- **CNN for Satellite Imagery**: Detects archaeological patterns in simulated imagery
- **Model Evaluation**: ROC-AUC, accuracy, precision-recall, feature importance

### 🗺️ **Spatial Modeling & Mapping**
- **Probability Mapping**: Heatmaps of archaeological potential
- **Risk Classification**: Very Low to Very High potential categories
- **Geographically Weighted Regression**: Accounts for spatial non-stationarity

### 🎯 **Survey Optimization**
- **Priority Scoring**: Multi-factor ranking (probability + accessibility + significance)
- **Field Survey Recommendations**: Optimized locations for field exploration
- **Resource Allocation Guidance**: Maximizes discovery efficiency

## 📁 Project Structure

```
Archaeological-Site-Discovery-Predictor/
├── data/
│   ├── synthetic_data.csv      # Generated training data
│   ├── geospatial/             # Placeholder for real data
│   └── satellite_patches/      # Simulated imagery
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_spatial_analysis.ipynb
├── src/
│   ├── data_processing.py      # ArchaeologicalDataProcessor
│   ├── feature_analysis.py     # FeatureAnalyzer
│   ├── prediction_model.py     # ArchaeologicalSitePredictor
│   ├── satellite_analysis.py   # SatelliteImageAnalyzer
│   ├── mapping.py             # ArchaeologicalSiteMapper
│   ├── survey_recommendation.py # SurveyRecommendationSystem
│   └── validation.py          # ModelValidator
├── models/                     # Trained model files
├── outputs/                    # Generated maps and visualizations
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Archaeological-Site-Discovery-Predictor.git
   cd Archaeological-Site-Discovery-Predictor
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the demo notebook**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

## 📊 Usage Examples

### 1. **Generate Synthetic Data**
```python
from src.data_processing import ArchaeologicalDataProcessor

processor = ArchaeologicalDataProcessor()
data = processor.generate_synthetic_data(n_samples=1000)
features, labels = processor.extract_features(data)
```

### 2. **Train Prediction Model**
```python
from src.prediction_model import ArchaeologicalSitePredictor

predictor = ArchaeologicalSitePredictor()
model, metrics = predictor.train_random_forest(features, labels)
print(f"Model Accuracy: {metrics['accuracy']:.2%}")
```

### 3. **Generate Probability Map**
```python
from src.mapping import ArchaeologicalSiteMapper

mapper = ArchaeologicalSiteMapper()
probability_map = mapper.generate_probability_map(model, geospatial_data)
mapper.plot_risk_categories(probability_map)
```

### 4. **Get Survey Recommendations**
```python
from src.survey_recommendation import SurveyRecommendationSystem

survey_system = SurveyRecommendationSystem()
recommendations = survey_system.recommend_survey_locations(
    probability_map, 
    accessibility_data,
    top_n=10
)
```

## 🧪 Technologies Used

### **Data Science & Machine Learning**
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Traditional ML models
- `xgboost` - Gradient boosting framework
- `tensorflow/keras` - Deep learning (CNN for imagery)

### **Geospatial Analysis**
- `geopandas` - Geographic data manipulation
- `rasterio` - Raster/geospatial data processing
- `spreg`, `mgwr` - Spatial regression modeling

### **Visualization**
- `matplotlib`, `seaborn` - Statistical plotting
- `folium` - Interactive web mapping

## 📈 Results & Outputs

The system produces several key outputs:

1. **Probability Maps**: Interactive heatmaps showing archaeological potential
2. **Risk Classification**: Categorical maps (Very Low to Very High potential)
3. **Feature Importance**: Charts showing most predictive factors
4. **Survey Priority List**: Ranked locations for field exploration
5. **Model Metrics**: Performance evaluation across different regions

## 🚧 Current Limitations

- **Simulated Data**: Currently uses synthetic data for demonstration
- **Data Dependency**: Requires real geospatial datasets for production use
- **Manual Setup**: Geospatial files need manual acquisition and upload
- **Satellite Imagery**: Uses simulated patches rather than real imagery

## 🔮 Future Enhancements

1. **Real Data Integration**
   - APIs for automatic geospatial data retrieval
   - Support for common archaeological data formats
   - Pre-configured datasets for key regions

2. **Advanced Features**
   - Integration with actual satellite imagery (Sentinel, Landsat)
   - Time-series analysis for landscape change detection
   - Multi-modal data fusion (LiDAR, ground-penetrating radar)

3. **User Interface**
   - Fully functional Streamlit web application
   - Interactive map-based data exploration
   - Export functionality for field teams

4. **Model Improvements**
   - Transfer learning for different geographical regions
   - Ensemble methods combining multiple data sources
   - Uncertainty quantification for predictions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{ArchaeologicalSiteDiscovery2024,
  title = {Archaeological Site Discovery Predictor},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Archaeological-Site-Discovery-Predictor}
}
```

## 🙏 Acknowledgments

- Inspired by archaeological predictive modeling research
- Built upon open-source geospatial and ML libraries
- Special thanks to contributors and testers

---

**🔍 Discover the Past, Predict the Future**

*For questions or collaboration, please open an issue or contact the maintainer.*
