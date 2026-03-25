"""
CreditWise ML Loan Prediction — Real ML Models
SecureTrust Bank — Logistic Regression · KNN · Naive Bayes
Trained on: loan_approval_data schema from credit_wise notebook
"""

import os, time, json, random
import numpy as np

# ── Load artefacts once at startup ────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "ml")
_MODELS_LOADED = False
_lr = _knn = _nb = _scaler = _ohe = _meta = None

def _load_models():
    global _lr, _knn, _nb, _scaler, _ohe, _meta, _MODELS_LOADED
    try:
        import joblib
        _lr     = joblib.load(os.path.join(_BASE, "logistic_model.pkl"))
        _knn    = joblib.load(os.path.join(_BASE, "knn_model.pkl"))
        _nb     = joblib.load(os.path.join(_BASE, "naive_bayes_model.pkl"))
        _scaler = joblib.load(os.path.join(_BASE, "scaler.pkl"))
        _ohe    = joblib.load(os.path.join(_BASE, "ohe.pkl"))
        with open(os.path.join(_BASE, "meta.json")) as f:
            _meta = json.load(f)
        _MODELS_LOADED = True
        print("[CreditWise] ML models loaded successfully ✓")
    except Exception as e:
        print(f"[CreditWise] ML models not found — falling back to rule engine: {e}")
        _MODELS_LOADED = False

_load_models()


def predict_loan(data: dict) -> dict:
    if _MODELS_LOADED:
        return _predict_ml(data)
    return _predict_rules(data)


# ── ML Prediction ─────────────────────────────────────────────────────────────

def _predict_ml(data: dict) -> dict:
    import pandas as pd

    # Parse inputs
    income       = _to_float(data.get("income", 0))
    coincome     = _to_float(data.get("coincome", 0))
    loan_amt     = _to_float(data.get("loanamt", 0))
    term         = _to_float(data.get("term", 360))
    credit_score = _to_float(data.get("credit_score", 650))
    dependents   = _to_int(data.get("dependents", 0).replace("3+", "3") if isinstance(data.get("dependents"), str) else str(data.get("dependents", 0)))
    education    = str(data.get("education", "Not Graduate"))   # "Graduate" | "Not Graduate"
    emp_status   = str(data.get("employment_status", "Employed"))  # Employed | Self-Employed | Unemployed
    emp_cat      = str(data.get("employer_category", "Private"))   # Private | Government | NGO | Self
    marital      = "Married" if str(data.get("married","no")).lower() == "yes" else "Single"
    gender       = str(data.get("gender", "Male")).capitalize()
    loan_purpose = str(data.get("type", "Home")).capitalize()
    prop_area    = str(data.get("area", "Urban")).capitalize()

    # DTI Ratio
    monthly_income = income + coincome
    emi = loan_amt / term if term > 0 else loan_amt
    dti_ratio = emi / monthly_income if monthly_income > 0 else 2.0
    dti_ratio = min(dti_ratio, 2.0)

    # Education label encode (Graduate=0, Not Graduate=1 OR reverse — match training)
    edu_classes = _meta.get("edu_classes", ["Graduate", "Not Graduate"])
    edu_encoded = edu_classes.index(education) if education in edu_classes else 0

    # Normalise values to match training categories
    loan_purpose_map = {"home": "Home", "personal": "Personal", "education": "Education",
                        "vehicle": "Vehicle", "business": "Business"}
    loan_purpose = loan_purpose_map.get(data.get("type","home").lower(), "Home")

    area_map = {"urban": "Urban", "semiurban": "Semiurban", "rural": "Rural"}
    prop_area = area_map.get(data.get("area","urban").lower(), "Urban")

    # Build OHE input
    ohe_cols    = _meta["ohe_cols"]  # Employment_Status, Marital_Status, Loan_Purpose, Property_Area, Gender, Employer_Category
    ohe_input   = pd.DataFrame([[emp_status, marital, loan_purpose, prop_area, gender, emp_cat]],
                                columns=ohe_cols)
    ohe_encoded = _ohe.transform(ohe_input)

    # Feature engineering
    dti_sq = dti_ratio ** 2
    cs_sq  = credit_score ** 2

    # Base numerical features (order must match training)
    num_feats = [dependents, edu_encoded, income, coincome, loan_amt, term]

    # Full feature vector
    feature_vec = np.array(num_feats + list(ohe_encoded[0]) + [dti_sq, cs_sq]).reshape(1, -1)

    # Scale
    feature_scaled = _scaler.transform(feature_vec)

    # Run all 3 models
    lr_pred  = int(_lr.predict(feature_scaled)[0])
    knn_pred = int(_knn.predict(feature_scaled)[0])
    nb_pred  = int(_nb.predict(feature_scaled)[0])

    lr_prob  = float(_lr.predict_proba(feature_scaled)[0][1])
    nb_prob  = float(_nb.predict_proba(feature_scaled)[0][1])

    # Majority vote
    votes = lr_pred + knn_pred + nb_pred
    approved = votes >= 2

    # Ensemble probability
    ensemble_prob = (lr_prob + nb_prob) / 2  # KNN doesn't give calibrated probs
    confidence = int(min(97, max(62, ensemble_prob * 100 if approved else (1 - ensemble_prob) * 100)))

    # Scoring factors for display (0–100 visual bars)
    factors = _compute_factors(credit_score, dti_ratio, education, prop_area, dependents, emp_status, data)

    # Score (visual 0–100 derived from ensemble prob)
    score = int(min(99, max(5, ensemble_prob * 100)))

    # Interest rate
    rate_map = {
        "Home":      {"Urban": 8.50, "Semiurban": 8.25, "Rural": 8.75},
        "Personal":  {"Urban": 11.0, "Semiurban": 10.5, "Rural": 11.5},
        "Education": {"Urban": 8.75, "Semiurban": 8.50, "Rural": 9.00},
        "Vehicle":   {"Urban": 9.50, "Semiurban": 9.25, "Rural": 9.75},
        "Business":  {"Urban": 10.5, "Semiurban": 10.0, "Rural": 11.0},
    }
    interest_rate = rate_map.get(loan_purpose, {}).get(prop_area, 9.0) if approved else None

    max_loan = int(monthly_income * term * 0.45)

    return {
        "approved":     approved,
        "score":        score,
        "confidence":   confidence,
        "factors":      factors,
        "emi":          int(emi),
        "maxLoan":      max_loan,
        "rate":         interest_rate,
        "id":           f"STB-{int(time.time()) % 1000000:06d}",
        "monthly":      int(monthly_income),
        "loanamt":      int(loan_amt),
        "term":         int(term),
        "model_votes":  {"logistic": lr_pred, "knn": knn_pred, "naive_bayes": nb_pred},
        "ensemble_prob": round(ensemble_prob, 3),
        "ml_powered":   True,
    }


def _compute_factors(credit_score, dti_ratio, education, prop_area, dependents, emp_status, data):
    """Visual scoring factors for the result card."""
    factors = []

    # 1. Credit score
    cs = int(credit_score)
    if cs >= 750:   cs_v, cs_c = 95, "#0a7c4e"
    elif cs >= 700: cs_v, cs_c = 80, "#0a7c4e"
    elif cs >= 650: cs_v, cs_c = 62, "#92600a"
    elif cs >= 600: cs_v, cs_c = 42, "#92600a"
    else:           cs_v, cs_c = 18, "#c0392b"
    factors.append({"n": f"Credit score ({cs})", "v": cs_v, "c": cs_c})

    # 2. DTI ratio
    if dti_ratio < 0.30:   dti_v, dti_c = 92, "#0a7c4e"
    elif dti_ratio < 0.45: dti_v, dti_c = 72, "#92600a"
    elif dti_ratio < 0.60: dti_v, dti_c = 50, "#92600a"
    elif dti_ratio < 0.75: dti_v, dti_c = 30, "#c0392b"
    else:                  dti_v, dti_c = 10, "#c0392b"
    factors.append({"n": f"Debt-to-Income ({dti_ratio:.2f})", "v": dti_v, "c": dti_c})

    # 3. Education
    edu_v = 82 if education == "Graduate" else 52
    factors.append({"n": "Education level", "v": edu_v, "c": "#0a7c4e" if edu_v >= 70 else "#92600a"})

    # 4. Property area
    area_map = {"Semiurban": 88, "Urban": 76, "Rural": 56}
    area_v = area_map.get(prop_area, 60)
    factors.append({"n": "Property area", "v": area_v, "c": "#0a7c4e" if area_v >= 75 else "#92600a"})

    # 5. Employment
    emp_map = {"Employed": 88, "Self-Employed": 60, "Unemployed": 20}
    emp_v = emp_map.get(emp_status, 60)
    factors.append({"n": "Employment status", "v": emp_v, "c": "#0a7c4e" if emp_v >= 70 else ("#92600a" if emp_v >= 40 else "#c0392b")})

    # 6. Dependents
    dep_map = {0: 90, 1: 78, 2: 62, 3: 40}
    dep = min(int(str(dependents).replace("3+", "3")), 3)
    dep_v = dep_map.get(dep, 40)
    factors.append({"n": f"Dependents ({dep})", "v": dep_v, "c": "#0a7c4e" if dep_v >= 70 else "#92600a"})

    return factors


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _predict_rules(data: dict) -> dict:
    income      = _to_float(data.get("income", 0))
    coincome    = _to_float(data.get("coincome", 0))
    loan_amt    = _to_float(data.get("loanamt", 0))
    term        = _to_float(data.get("term", 360))
    credit_raw  = data.get("credit_score", data.get("credit", "0"))
    try:
        credit_score = float(credit_raw)
        if credit_score <= 1:   # old binary format
            credit_score = 750 if credit_score == 1 else 500
    except Exception:
        credit_score = 550

    monthly = income + coincome
    emi     = loan_amt / term if term > 0 else loan_amt
    dti     = emi / monthly if monthly > 0 else 999

    score = 0
    factors = []

    # Credit score
    if credit_score >= 750:   cs_v = 95; score += 35
    elif credit_score >= 700: cs_v = 80; score += 28
    elif credit_score >= 650: cs_v = 62; score += 18
    elif credit_score >= 600: cs_v = 42; score += 10
    else:                     cs_v = 18; score += 0
    cs_c = "#0a7c4e" if credit_score >= 700 else ("#92600a" if credit_score >= 600 else "#c0392b")
    factors.append({"n": f"Credit score ({int(credit_score)})", "v": cs_v, "c": cs_c})

    # DTI
    if dti < 0.30:   ir_v = 92; score += 25
    elif dti < 0.45: ir_v = 72; score += 18
    elif dti < 0.60: ir_v = 50; score += 10
    elif dti < 0.75: ir_v = 30; score += 4
    else:            ir_v = 10; score += 0
    factors.append({"n": f"Debt-to-Income ({dti:.2f})", "v": ir_v, "c": _color(ir_v)})

    education = str(data.get("education","")).lower()
    edu_v = 82 if education == "graduate" else 52
    score += 10 if education == "graduate" else 5
    factors.append({"n": "Education level", "v": edu_v, "c": _color(edu_v)})

    area = str(data.get("area","urban")).lower()
    area_scores = {"semiurban": 88, "urban": 76, "rural": 56}
    area_v = area_scores.get(area, 60)
    score += {"semiurban":12,"urban":9,"rural":5}.get(area,5)
    factors.append({"n": "Property area", "v": area_v, "c": _color(area_v)})

    emp = str(data.get("employment_status", "employed")).lower()
    emp_scores = {"employed": 88, "self-employed": 60, "unemployed": 20}
    emp_v = emp_scores.get(emp, 60)
    score += {"employed":5,"self-employed":2,"unemployed":-5}.get(emp,2)
    factors.append({"n": "Employment status", "v": emp_v, "c": _color(emp_v)})

    approved = score >= 55 and credit_score >= 600
    confidence = min(97, max(62, 60 + abs(score - 50) + random.randint(-3, 3)))
    max_loan = int(monthly * term * 0.45)
    loan_type = str(data.get("type","home")).lower()
    rate_map = {"home": 8.5, "personal": 11.0, "education": 8.75, "vehicle": 9.5, "business": 10.5}
    interest_rate = rate_map.get(loan_type, 9.0) if approved else None

    return {
        "approved": approved, "score": min(100, max(0, score)),
        "confidence": confidence, "factors": factors,
        "emi": int(emi), "maxLoan": max_loan, "rate": interest_rate,
        "id": f"STB-{int(time.time()) % 1000000:06d}",
        "monthly": int(monthly), "loanamt": int(loan_amt), "term": int(term),
        "ml_powered": False,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(val, default=0.0):
    try:  return float(str(val).replace(",", ""))
    except: return default

def _to_int(val, default=0):
    try:  return int(float(str(val).replace(",", "")))
    except: return default

def _color(v):
    if v >= 70: return "#0a7c4e"
    if v >= 45: return "#92600a"
    return "#c0392b"
