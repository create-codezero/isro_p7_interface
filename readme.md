# 🚀 Exoplanet Intelligence Architecture (ISRO P7 Challenge)

An advanced, production-ready astrophysical data pipeline and machine learning interface built to ingest Kepler/TESS stellar photometry telemetry (`.fits` formats), extract deep mathematical structures, optimize transit geometry via analytical physics modeling, and classify targets into distinct cosmological structures.

**🔗 Live Deployment Link:** [Exoplanet Intelligence Portal](https://isro-p7-interface.onrender.com)  
**Author:** Amit Kumar Tiwari  

---

## 🖥️ User Interface Overview

Below is the active layout of the dynamic telemetry dashboard and scientific inference platform:

![Exoplanet Intelligence Architecture Interface](static/ss/Screenshot%202026-06-28%20000935.png)

---

## 🧬 Core Engine & Architecture

The framework is cleanly decoupled into three mission-critical layers:

### 1. Data Ingestion & Signal Detrending
* **Zero-Disk Clutter Buffer:** Light curve files (`.fits`) are streamed entirely within system volatile memory structures using `astropy.io.fits` to maximize processing throughput on cloud nodes.
* **Photometric Flattening:** Raw time-series arrays filter cosmic ray anomalies, quality control flags, and systemic stellar trends via a custom `lightkurve` window-slider optimization layout.

### 2. Multi-Class Feature Extraction & Machine Learning
The analytical backbone runs an optimized, hyperparameter-tuned multi-class `XGBoostClassifier` built to isolate faint planetary transit dips from complex astrophysical background noise. It maps signals across four structural designations:
* `Class 0`: Normal / Stellar / Detector Artifacts
* `Class 1`: Confirmed Transit (Exoplanet Candidate)
* `Class 2`: Eclipsing Binary System (EB)
* `Class 3`: Background Field Blend Contaminant

#### Engineered Mathematical Features Include:
* **BLS Spectrum Mapping:** Periodogram maximization using Box Least Squares (`scipy` / `astropy`).
* **Higher-Order Statistical Moments:** Evaluates skewness, kurtosis, and Median Absolute Deviation (`MAD`) to isolate asymmetric blend profiles.
* **Dynamic Physics Ratios:** Computes depth-to-duration ratios ($\delta / T_{\text{dur}}$) and range-to-depth parameters to accurately differentiate $V$-shaped stellar eclipses from clean $U$-shaped planetary signatures.

### 3. Keplerian Geometry Solver (`batman`)
Once a primary period matches candidate criteria, a non-linear analytical optimizer engine utilizes Levenberg-Marquardt / L-BFGS-B minimizers to evaluate explicit orbital parameters via the `batman` transit framework:
* Planet-to-star radius ratio ($R_p / R_s$)
* Semi-major axis normalized by stellar radius ($a / R_s$)
* Orbital Inclination ($i$)

### 4. Asynchronous Chat Inference Layer
Integrates the state-of-the-art **Llama-3.3 70B Versatile** engine over a low-latency Groq cluster. The LLM acts as an automated Senior Lead Research Astrophysicist, securely holding runtime metrics, analytical model residuals, and NASA Exoplanet Archive cross-reference parameters inside its transient system instruction frame for rapid diagnostic exploration.

---

## 📦 Project Directory Layout

```text
├── app.py                      # Production Flask application engine & API handlers
├── templates/
│   └── index.html              # TailwindCSS-driven responsive responsive dashboard
├── requirements.txt            # Strictly pinned production dependencies
├── exoplanet_classifier.pkl    # Serialized model deployment pipeline artifact
├── best_classifier.joblib      # Hyperparameter tuned multi-class model weights
├── feature_names.joblib        # Ordered column naming alignments
└── README.md                   # System documentation core