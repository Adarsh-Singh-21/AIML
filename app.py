"""Numerical lung cancer project with grouped sidebar navigation."""
from pathlib import Path

import streamlit as st

from lung_tabular import ARTIFACTS, REPORTS, ROOT
from lung_tabular.model import load_model, resolve_run
from lung_tabular.ui import DEVELOPERS, PROFESSOR, EVALUATOR

st.set_page_config(page_title="Numerical Lung Cancer Project", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {max-width:1220px; padding-top:2.4rem; padding-bottom:2.5rem;}
h1 {font-size:2.25rem!important; letter-spacing:-.035em; color:#111318;}
h2 {font-size:1.65rem!important; letter-spacing:-.025em;}
h3 {font-size:1.3rem!important;}
[data-testid="stSidebar"] {background:#F1F0EC; min-width:265px; max-width:285px; border-right:1px solid #E8E7E2;}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] {padding-top:0;}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] li {margin-bottom:0;}
[data-testid="stSidebarUserContent"] .stVerticalBlock {gap:.65rem;}
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
    display:grid; grid-template-columns:20px minmax(0,1fr); align-items:center;
    column-gap:10px; min-height:34px; height:auto; padding:5px 10px;
    font-size:.82rem; line-height:20px; border-radius:6px; box-sizing:border-box;
}
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] > span:first-child {
    display:flex; align-items:center; justify-content:center;
    width:20px; height:20px; line-height:20px;
}
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] [data-testid="stIconEmoji"] {
    display:flex; align-items:center; justify-content:center;
    width:20px; height:20px; font-size:16px; line-height:20px;
}
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] > span[label],
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] [data-testid="stMarkdownContainer"] {
    display:block; min-width:0; line-height:20px;
}
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] p {
    display:block; margin:0; padding:0; font-size:.82rem; line-height:20px;
    white-space:normal; overflow-wrap:break-word;
}
[data-testid="stSidebar"] a[aria-current="page"] {background:#DEDFD7; font-weight:600; color:#111318;}
[data-testid="stSidebar"] h3 {font-size:1.05rem!important;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {font-size:.79rem;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {font-size:.75rem;}
.project-credits {border-top:1px solid #CFCFC6; margin-top:.8rem; padding-top:.8rem; color:#6D6D66; font-size:.75rem; line-height:1.85;}
.project-credits strong {color:#575850; font-weight:600;}
[data-testid="stMetricValue"] {font-size:1.95rem; font-weight:500;}
button[data-baseweb="tab"] {font-size:.84rem;}
@media(max-width:760px) {h1 {font-size:1.85rem!important;} .block-container {padding-top:1.7rem;}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def saved_model(path, digest):
    return load_model(Path(path))


try:
    run, metadata = resolve_run(ARTIFACTS)
    model, metadata = saved_model(str(run), metadata["model_sha256"])
except FileNotFoundError:
    st.warning("No numerical model is installed yet.")
    st.code("python -m lung_tabular download-data\npython -m lung_tabular prepare-data\npython -m lung_tabular train")
    st.stop()
except (OSError, ValueError, KeyError) as error:
    st.error(f"Numerical model could not be loaded: {error}")
    st.stop()

st.session_state["_project_context"] = {"root": ROOT, "reports": REPORTS, "run": run, "model": model, "metadata": metadata}

page = st.navigation({
    "Overview": [st.Page("ui_pages/home.py", title="Home", icon="🏠", default=True)],
    "Build the model": [
        st.Page("ui_pages/data.py", title="1 · Data", icon="📥", url_path="data"),
        st.Page("ui_pages/preprocessing.py", title="2 · Preprocessing & EDA", icon="🧹", url_path="preprocessing"),
        st.Page("ui_pages/training.py", title="3 · Training & Tuning", icon="🧠", url_path="training"),
        st.Page("ui_pages/evaluation.py", title="4 · Evaluation", icon="📊", url_path="evaluation"),
        st.Page("ui_pages/insights.py", title="5 · Model Insights", icon="🔎", url_path="insights"),
    ],
    "Use the model": [
        st.Page("ui_pages/predict.py", title="Single-record prediction", icon="🩺", url_path="predict"),
        st.Page("ui_pages/batch.py", title="CSV batch prediction", icon="📁", url_path="batch"),
    ],
}, position="sidebar", expanded=True)

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.selectbox("Dataset config", ["akashnath29_lung_cancer"],
                 help="This project uses the user-selected Kaggle dataset, version 2.")
    st.markdown("**Prediction target**")
    st.caption("LUNG_CANCER · NO / YES")
    st.radio("Evaluation focus", ["YES class", "Macro average"], key="evaluation_focus",
             help="Controls precision, recall and F1 on the Evaluation page. Accuracy is unchanged.")
    st.markdown(f'<div class="project-credits"><strong>Developed by:</strong> {DEVELOPERS}<br>'
                f'<strong>Submitted to:</strong> {PROFESSOR}<br>'
                f'<strong>Evaluator:</strong> {EVALUATOR}</div>', unsafe_allow_html=True)

page.run()
