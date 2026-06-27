import os
import io
import base64
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib
matplotlib.use('Agg') # Headless safety for web-servers
import matplotlib.pyplot as plt
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from astropy.io import fits
from lightkurve import LightCurve
import batman
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, median_abs_deviation
from astropy.timeseries import BoxLeastSquares
from groq import Groq

app = Flask(__name__)
app.secret_key = "isro_p7_deep_space_secret_key"

# Artifact file names matched directly from your training script output
MODEL_FILE = "best_classifier.joblib"
FEATURES_FILE = "feature_names.joblib"

# Global lazy-loading configuration
if Path(MODEL_FILE).exists() and Path(FEATURES_FILE).exists():
    try:
        TRAINED_MODEL = joblib.load(MODEL_FILE)
        MODEL_FEATURES = joblib.load(FEATURES_FILE)
    except Exception as e:
        print(f"Error loading system ML artifacts: {str(e)}")
        TRAINED_MODEL, MODEL_FEATURES = None, None
else:
    TRAINED_MODEL, MODEL_FEATURES = None, None

def check_nasa_archive(tic_id):
    query = f"select pl_name, discoverymethod, disc_year from ps where tic_id='TIC {tic_id}'"
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=json"
    try:
        r = requests.get(url, timeout=4)
        if r.status_code == 200 and len(r.json()) > 0:
            d = r.json()[0]
            return f"CONFIRMED PLANET: {d['pl_name']} ({d['discoverymethod']}, {d['disc_year']})"
        return "UNCONFIRMED CANDIDATE / FALSE POSITIVE FIELD"
    except:
        return "NASA Database Unreachable"

def fit_batman_transit(time, flux, period, depth_guess):
    def calc_residuals(params):
        rp, a, inc = params
        m = batman.TransitParams()
        m.t0, m.per, m.rp, m.a, m.inc, m.ecc, m.w = 0.0, period, rp, a, inc, 0.0, 90.0
        m.u, m.limb_dark = [0.1, 0.3], "quadratic"
        try:
            flux_model = batman.TransitModel(m, time).light_curve(m)
            return np.sum((flux - flux_model)**2)
        except:
            return 1e10

    res = minimize(calc_residuals, x0=[np.sqrt(max(depth_guess, 0.0001)), 10.0, 90.0], 
                   bounds=((0.001, 0.5), (2.0, 100.0), (70.0, 90.0)), method='L-BFGS-B')
    return res.x

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    base64_str = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    return base64_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if not TRAINED_MODEL or not MODEL_FEATURES:
        return jsonify({"error": "ML Pipeline payload components ('best_classifier.joblib' or 'feature_names.joblib') missing from root server directory."}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file payload sent"}), 400
        
    file = request.files['file']
    groq_key = (
    os.getenv("GROQ_API_KEY", "").strip()
    or request.form.get("api_key", "").strip()
    )
    
    if not groq_key:
        return jsonify({"error": "Groq API Key initialization required to lock telemetry pipeline."}), 400

    try:
        # 1. Stream file buffer from memory
        file_bytes = file.read()
        with fits.open(io.BytesIO(file_bytes), memmap=False) as hdul:
            header = hdul[0].header
            tic_id = str(header.get("TICID", "Unknown"))
            sector = str(header.get("SECTOR", "Unknown"))
            
            data = hdul[1].data
            mask = np.isfinite(data["TIME"]) & np.isfinite(data["PDCSAP_FLUX"]) & (data["QUALITY"] == 0)
            raw_time, raw_flux = data["TIME"][mask], data["PDCSAP_FLUX"][mask]

        if len(raw_time) < 100:
            return jsonify({"error": "Insufficient baseline telemetry available in chosen target's FITS file."}), 400

        # 2. Extract & Detrend
        normalized_flux = raw_flux / np.nanmedian(raw_flux)
        lc = LightCurve(time=raw_time, flux=normalized_flux).remove_nans().flatten(window_length=401)
        
        # 3. Calculate Periodogram Engine Base
        bls = BoxLeastSquares(lc.time.value, lc.flux.value)
        periodogram = bls.autopower(duration=0.1)
        best_idx = np.argmax(periodogram.power)
        
        # Intermediate baseline feature parameters
        raw_rms = float(np.std(lc.flux.value))
        raw_depth = float(periodogram.depth[best_idx])
        raw_duration = float(periodogram.duration[best_idx])
        raw_period = float(periodogram.period[best_idx])
        raw_skew = float(skew(lc.flux.value))

        # 4. Synthesize Advanced Feature Vectors matched directly with your new training loops
        features_dict = {
            "rms_flux": raw_rms,
            "bls_power": float(periodogram.power[best_idx]),
            "depth": raw_depth,
            "period": raw_period,
            "duration": raw_duration,
            "transit_time": float(periodogram.transit_time[best_idx]),
            "depth_snr": raw_depth / (raw_rms + 1e-6),
            "skew": raw_skew,
            "kurtosis": float(kurtosis(lc.flux.value)),
            "mad": float(median_abs_deviation(lc.flux.value)),
            "duration_to_period": raw_duration / (raw_period + 1e-6),
            "depth_to_duration": raw_depth / (raw_duration + 1e-6),
            "flux_skew_abs": abs(raw_skew),
            # Default fallbacks for physical metrics extracted down at full catalog cross-match levels
            "centroid_motion": 0.0,
            "motion_noise_ratio": 0.0,
            "flux_range": float(np.max(lc.flux.value) - np.min(lc.flux.value)),
            "flux_range_to_depth": float(np.max(lc.flux.value) - np.min(lc.flux.value)) / (raw_depth + 1e-6)
        }

        # 5. Build dynamic prediction frame aligned to feature columns
        X_eval = pd.DataFrame([features_dict])
        
        # Populate missing tracking metrics safely
        for col in MODEL_FEATURES:
            if col not in X_eval.columns:
                X_eval[col] = 0.0
                
        X_eval = X_eval[MODEL_FEATURES] # Force strict tabular sorting
        
        # Calculate Multiclass array probabilities
        raw_probs = TRAINED_MODEL.predict_proba(X_eval)
        
        # Map indices explicitly to your new training configurations
        class_labels_map = {
            0: "Stellar / Detector Noise",
            1: "Transit Candidate",
            2: "Eclipsing Binary",
            3: "Blend"
        }
        
        prob_matrix = []
        planet_probability = 0.0

        for idx in range(raw_probs.shape[1]):
            prob_val = float(raw_probs[0, idx])
            label_name = class_labels_map.get(idx, f"Class {idx}")
            prob_matrix.append({"label": label_name, "confidence": prob_val})
            if idx == 1:
                planet_probability = prob_val

        prob_matrix = sorted(prob_matrix, key=lambda x: x["confidence"], reverse=True)
        top_prediction = prob_matrix[0]["label"]

        # 6. Batman Geometry Fitting
        folded = lc.fold(period=raw_period, epoch_time=features_dict["transit_time"])
        sort_idx = np.argsort(folded.time.value)
        fit_time, fit_flux = folded.time.value[sort_idx], folded.flux.value[sort_idx]
        best_rp, best_a, best_inc = fit_batman_transit(fit_time, fit_flux, raw_period, raw_depth)
        
        params = batman.TransitParams()
        params.t0, params.per, params.rp, params.a, params.inc, params.ecc, params.w = 0.0, raw_period, best_rp, best_a, best_inc, 0.0, 90.0
        params.u, params.limb_dark = [0.1, 0.3], "quadratic"
        batman_flux = batman.TransitModel(params, fit_time).light_curve(params)

        duration_hours = float(raw_duration * 24.0)
        nasa_status = check_nasa_archive(tic_id)

        # 7. Asynchronous Plot Construction
        fig1, ax1 = plt.subplots(figsize=(6, 3.2))
        ax1.scatter(lc.time.value, lc.flux.value, s=1, color="gray", alpha=0.4)
        ax1.set_xlabel("Time (BJD - 2457000)", fontsize=8)
        ax1.set_ylabel("Normalized Flux", fontsize=8)
        ax1.set_title(f"TIC {tic_id} Detrended Light Curve", fontsize=10, fontweight='bold', color='#38bdf8')
        ax1.grid(True, alpha=0.15)
        plot_lc_b64 = fig_to_base64(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        ax2.scatter(folded.time.value, folded.flux.value, s=2, color="#3b82f6", alpha=0.3, label="Data")
        ax2.plot(fit_time, batman_flux, color="#ef4444", linewidth=2.5, label="Batman Analytical Model")
        ax2.set_xlabel("Phase (Days)", fontsize=8)
        ax2.set_ylabel("Relative Flux", fontsize=8)
        ax2.set_title("Phase Folded Geometric Fit Analysis", fontsize=10, fontweight='bold', color='#a855f7')
        ax2.legend(loc="lower right", prop={'size': 8})
        ax2.grid(True, alpha=0.15)
        plot_fit_b64 = fig_to_base64(fig2)

        # 8. Save session parameters for LLM Context Engine
        session["target_metadata"] = {
            "tic_id": tic_id, "sector": sector, "prob": f"{planet_probability*100:.2f}%",
            "top_pred": top_prediction,
            "period": f"{raw_period:.4f} days", "depth": f"{raw_depth:.5f}",
            "duration": f"{duration_hours:.2f} hours", "snr": f"{features_dict['depth_snr']:.2f}", 
            "rp": f"{best_rp:.4f}", "a": f"{best_a:.2f}", "inc": f"{best_inc:.2f} deg", 
            "nasa": nasa_status, "groq_api_key": groq_key
        }
        
        return jsonify({
            "success": True, "tic_id": tic_id, "sector": sector,
            "prob": f"{planet_probability*100:.1f}%", "top_pred": top_prediction, "nasa": nasa_status,
            "fit_params": {
                "rp": f"{best_rp:.4f}", 
                "a": f"{best_a:.2f}", 
                "inc": f"{best_inc:.1f}°",
                "duration": f"{duration_hours:.2f} hours"
            },
            "prob_matrix": prob_matrix,
            "features_json": features_dict, "plot_lc": plot_lc_b64, "plot_fit": plot_fit_b64
        })

    except Exception as err:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Pipeline processing error: {str(err)}"}), 500


@app.route('/chat', methods=['POST'])
def chat_with_inference_engine():
    meta = session.get("target_metadata")
    if not meta:
        return jsonify({"error": "No target photometry profile has been processed yet."}), 400
        
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty queries are rejected."}), 400

    system_instruction = (
        f"You are a Senior Lead Research Astrophysicist evaluating stellar lightcurves.\n"
        f"Context: Analyzing target TIC {meta['tic_id']} from Sector {meta['sector']}.\n"
        f"Physical Diagnostics calculated by our software pipeline:\n"
        f"- Automated XGBoost Classifier Core Ruling: {meta['top_pred']} (Transit Class Probability: {meta['prob']})\n"
        f"- Box Least Squares Period: {meta['period']} | Transit Depth: {meta['depth']} | SNR: {meta['snr']}\n"
        f"- Calculated Transit Duration: {meta['duration']}\n"
        f"- Analytical Batman Optimization parameters: Rp/Rs={meta['rp']}, Axis a/Rs={meta['a']}, Inclination={meta['inc']}.\n"
        f"- Official NASA Database Cross-Reference: {meta['nasa']}.\n\n"
        f"Be strictly analytical, quantitative, objective, and concise. Explain anomalies, evaluate signal significance, or confirm true transit validation vs false positive blend/eclipses based exclusively on these metrics."
    )

    try:
        api_key = meta.get("groq_api_key")
        if not api_key:
            return jsonify({"error": "Missing API Authentication: Pass a valid Groq Key."}), 401
            
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return jsonify({"reply": completion.choices[0].message.content})
    except Exception as e:
        print(f"Inference Engine Failure: {str(e)}")
        return jsonify({"error": f"Pipeline Chat Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)