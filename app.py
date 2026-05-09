"""
Spaceship Titanic — AutoML  |  Streamlit UI
============================================
5-step guided wizard:
  1. Upload Data
  2. Select Target
  3. Task Detection
  4. Train & Tune
  5. Evaluate & Download
"""

import io
import json
import joblib
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.task_detector import TaskDetector
from src.automl_engine import AutoMLEngine
from src.evaluator import full_report

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AutoML Pipeline",
    page_icon   = "🚀",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Session state defaults ─────────────────────────────────────
for key, default in {
    "step":       1,
    "df":         None,
    "target_col": None,
    "task":       None,
    "detection":  None,
    "engine":     None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════
# Sidebar — progress tracker
# ══════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.title("🚀 AutoML Pipeline")
        st.markdown("---")
        steps = [
            ("📂", "Upload Data"),
            ("🎯", "Select Target"),
            ("🔍", "Task Detection"),
            ("🤖", "Train & Tune"),
            ("📊", "Evaluate & Download"),
        ]
        for i, (icon, label) in enumerate(steps, 1):
            if i < st.session_state.step:
                st.markdown(f"✅ **Step {i}:** {label}")
            elif i == st.session_state.step:
                st.markdown(f"▶️ **Step {i}: {label}** ← *current*")
            else:
                st.markdown(f"⬜ Step {i}: {label}")

        st.markdown("---")
        if st.button("🔄 Reset All", use_container_width=True):
            for key in ["step","df","target_col","task","detection","engine"]:
                st.session_state[key] = 1 if key == "step" else None
            st.rerun()


# ══════════════════════════════════════════════════════════════
# Step 1 — Upload Data
# ══════════════════════════════════════════════════════════════

def step_upload():
    st.header("📂 Step 1 — Upload Your Data")
    st.markdown("Upload a **CSV or Excel** file. Data is processed in-memory and never saved to disk.")

    uploaded = st.file_uploader(
        "Drop your file here", type=["csv", "xlsx", "xls"], label_visibility="collapsed"
    )

    if uploaded is None:
        st.info("👆 Upload a file to get started.")
        return

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    st.success(f"✅ Loaded **{df.shape[0]:,} rows × {df.shape[1]} columns**")

    # Quality dashboard
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows",        f"{df.shape[0]:,}")
    col2.metric("Columns",     df.shape[1])
    col3.metric("Missing (%)", f"{df.isnull().mean().mean()*100:.1f}%")
    col4.metric("Duplicates",  df.duplicated().sum())

    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📊 Column Info"):
        info = pd.DataFrame({
            "dtype":   df.dtypes,
            "non_null":df.notnull().sum(),
            "null_%":  (df.isnull().mean() * 100).round(1),
            "nunique": df.nunique(),
        })
        st.dataframe(info, use_container_width=True)

    if st.button("▶️ Continue to Target Selection", type="primary", use_container_width=True):
        st.session_state.df   = df
        st.session_state.step = 2
        st.rerun()


# ══════════════════════════════════════════════════════════════
# Step 2 — Select Target
# ══════════════════════════════════════════════════════════════

def step_target():
    st.header("🎯 Step 2 — Select Target Column")
    df = st.session_state.df

    # Suggest last column as default (common Kaggle convention)
    default_idx = len(df.columns) - 1
    target_col  = st.selectbox(
        "Which column do you want to predict?",
        options   = df.columns.tolist(),
        index     = default_idx,
    )

    # Stats preview
    col = df[target_col]
    c1, c2, c3 = st.columns(3)
    c1.metric("dtype",   str(col.dtype))
    c2.metric("Unique",  col.nunique())
    c3.metric("Missing", f"{col.isnull().mean()*100:.1f}%")

    # Distribution chart
    fig, ax = plt.subplots(figsize=(6, 2.5))
    if col.dtype in ("object", "category", "bool") or col.nunique() <= 20:
        vc = col.value_counts().head(15)
        vc.plot(kind="bar", ax=ax, color="#4F8EF7")
        ax.set_title(f"Distribution of '{target_col}'")
        ax.set_xlabel("")
    else:
        col.dropna().plot(kind="hist", bins=30, ax=ax, color="#4F8EF7")
        ax.set_title(f"Distribution of '{target_col}'")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    if st.button("▶️ Detect Task Type", type="primary", use_container_width=True):
        st.session_state.target_col = target_col
        st.session_state.step       = 3
        st.rerun()


# ══════════════════════════════════════════════════════════════
# Step 3 — Task Detection
# ══════════════════════════════════════════════════════════════

def step_detect():
    st.header("🔍 Step 3 — Task Detection")
    df         = st.session_state.df
    target_col = st.session_state.target_col
    y          = df[target_col]

    # Run detection
    detector  = TaskDetector()
    detection = detector.detect(y)

    # Show result
    task_emoji = "🔵" if detection.task == "classification" else "🟠"
    st.markdown(f"### {task_emoji} Detected: **{detection.task.upper()}**")

    # Confidence bar
    conf_pct = int(detection.confidence * 100)
    color    = "#2ecc71" if conf_pct >= 80 else "#f39c12" if conf_pct >= 60 else "#e74c3c"
    st.markdown(
        f"**Confidence: {conf_pct}%**"
    )
    st.progress(conf_pct)

    # Rules fired
    with st.expander("📋 Detection Rules Fired", expanded=True):
        for rule in detection.rules_fired:
            st.markdown(f"- {rule}")

    # Override option
    st.markdown("---")
    st.markdown("**Override task type (optional):**")
    override = st.radio(
        "Task type",
        options       = ["Use detected", "classification", "regression"],
        horizontal    = True,
        label_visibility = "collapsed",
    )
    final_task = detection.task if override == "Use detected" else override

    if st.button("▶️ Proceed to Training", type="primary", use_container_width=True):
        st.session_state.detection = detection
        st.session_state.task      = final_task
        st.session_state.step      = 4
        st.rerun()


# ══════════════════════════════════════════════════════════════
# Step 4 — Train & Tune
# ══════════════════════════════════════════════════════════════

def step_train():
    st.header("🤖 Step 4 — Train & Tune")
    df         = st.session_state.df
    target_col = st.session_state.target_col
    task       = st.session_state.task

    c1, c2 = st.columns(2)
    n_trials = c1.slider("Optuna trials per model", min_value=5, max_value=100, value=20, step=5)
    cv_folds = c2.slider("Cross-validation folds",  min_value=3, max_value=10,  value=5)

    st.info(
        f"ℹ️  Will search all **{task}** models with **{n_trials} Optuna trials** each, "
        f"then auto-build 3 ensembles."
    )

    if st.button("🚀 Start AutoML", type="primary", use_container_width=True):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        log_area  = st.empty()
        progress  = st.progress(0)
        status    = st.status("Running AutoML...", expanded=True)

        with status:
            engine = AutoMLEngine(
                task         = task,
                n_trials     = n_trials,
                cv_folds     = cv_folds,
                val_size     = 0.2,
            )

            # Monkey-patch print to stream to Streamlit
            import sys
            class StreamlitWriter:
                def write(self, msg):
                    if msg.strip():
                        st.write(msg.strip())
                def flush(self): pass

            old_stdout      = sys.stdout
            sys.stdout      = StreamlitWriter()
            try:
                engine.fit(X, y)
            finally:
                sys.stdout = old_stdout

            progress.progress(100)
            status.update(label="✅ AutoML complete!", state="complete")

        st.session_state.engine = engine
        st.session_state.step   = 5
        st.rerun()


# ══════════════════════════════════════════════════════════════
# Step 5 — Evaluate & Download
# ══════════════════════════════════════════════════════════════

def step_evaluate():
    st.header("📊 Step 5 — Evaluate & Download")
    engine     = st.session_state.engine
    task       = st.session_state.task
    target_col = st.session_state.target_col

    # Champion banner
    primary_key = "val_accuracy" if task == "classification" else "val_rmse"
    best_val    = engine.results[engine.champion_name].get(primary_key, "N/A")
    st.success(f"🏆 **Champion: {engine.champion_name}**  —  {primary_key} = **{best_val}**")

    # Leaderboard
    st.subheader("📋 Leaderboard")
    lb = engine.leaderboard().reset_index().rename(columns={"index": "Model"})
    st.dataframe(lb, use_container_width=True)

    # Bar chart
    st.subheader("📈 Model Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_df = lb.set_index("Model")[primary_key].dropna().astype(float)
    colors  = ["#2ecc71" if m == engine.champion_name else "#4F8EF7" for m in plot_df.index]
    plot_df.plot(kind="bar", ax=ax, color=colors)
    ax.set_title(f"Models ranked by {primary_key}")
    ax.set_xlabel("")
    ax.set_ylabel(primary_key)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Downloads
    st.subheader("💾 Downloads")
    col1, col2, col3 = st.columns(3)

    # 1. Model card JSON
    card = {
        "champion":   engine.champion_name,
        "task":       engine._task,
        "target_col": target_col,
        "timestamp":  datetime.now().strftime("%Y%m%d_%H%M%S"),
        "leaderboard": lb.to_dict(orient="records"),
    }
    col1.download_button(
        label     = "📄 Download Model Card (JSON)",
        data      = json.dumps(card, indent=2, default=str),
        file_name = "model_card.json",
        mime      = "application/json",
        use_container_width = True,
    )

    # 2. Leaderboard CSV
    col2.download_button(
        label     = "📊 Download Leaderboard (CSV)",
        data      = lb.to_csv(index=False),
        file_name = "leaderboard.csv",
        mime      = "text/csv",
        use_container_width = True,
    )

    # 3. Model PKL
    buf = io.BytesIO()
    joblib.dump({"model": engine.champion, "engineer": engine.engineer}, buf)
    buf.seek(0)
    col3.download_button(
        label     = "🤖 Download Champion Model (.pkl)",
        data      = buf,
        file_name = "champion_model.pkl",
        mime      = "application/octet-stream",
        use_container_width = True,
    )

    # ── Predictions on new data ──────────────────────────────
    st.markdown("---")
    st.subheader("🔮 Predict on New Data")
    st.markdown("Upload a new CSV (without the target column) to get predictions.")

    new_file = st.file_uploader("Upload new data (CSV)", type=["csv"], key="predict_upload")
    if new_file is not None:
        try:
            new_df   = pd.read_csv(new_file)
            preds    = engine.predict(new_df)
            out_df   = new_df.copy()
            out_df[f"{target_col}_predicted"] = preds
            st.success(f"✅ {len(preds):,} predictions generated.")
            st.dataframe(out_df.head(10), use_container_width=True)
            st.download_button(
                label     = "⬇️ Download Predictions (CSV)",
                data      = out_df.to_csv(index=False),
                file_name = "predictions.csv",
                mime      = "text/csv",
                use_container_width = True,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════

sidebar()

step = st.session_state.step
if   step == 1: step_upload()
elif step == 2: step_target()
elif step == 3: step_detect()
elif step == 4: step_train()
elif step == 5: step_evaluate()
