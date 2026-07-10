import requests
import streamlit as st

API_URL = "https://dp-production-2c54.up.railway.app/predict"

st.set_page_config(
    page_title="EuroSAT Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #8a8a8a;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: rgba(46, 139, 87, 0.08);
        border: 1px solid rgba(46, 139, 87, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-class {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-confidence {
        font-size: 1.1rem;
        color: #4a4a4a;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("🛰️ About")
    st.write(
        "This app classifies satellite images into land cover types "
        "using the **EuroSAT** dataset categories."
    )
    st.markdown("---")
    st.subheader("How to use")
    st.markdown(
        """
        1. Upload a satellite image (JPG/PNG)
        2. Click **Predict**
        3. View the predicted land cover class
        """
    )
    st.markdown("---")
    st.caption("Built with FastAPI + Streamlit")

st.markdown('<div class="main-title">🛰️ EuroSAT Land Cover Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a satellite image and let the model classify its land cover type</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        predict_clicked = st.button("🔍 Predict", type="primary")
    else:
        st.info("👆 Upload an image to get started")
        predict_clicked = False

with right_col:
    st.subheader("📊 Result")

    if uploaded_file is None:
        st.markdown(
            """
            <div style="text-align:center; color:#8a8a8a; padding-top: 3rem;">
                No image uploaded yet.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif predict_clicked:
        with st.spinner("🛰️ Analyzing satellite image..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }
            response = requests.post(API_URL, files=files, timeout=60)

        if response.status_code == 200:
            result = response.json()

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size: 2rem;">✅</div>
                    <div class="result-class">{result['predicted_class']}</div>
                    <div class="result-confidence">Confidence: {result['confidence']:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(min(result['confidence'], 1.0))

        else:
            st.error("Prediction Failed")
            st.text(response.text)

    else:
        st.markdown(
            """
            <div style="text-align:center; color:#8a8a8a; padding-top: 3rem;">
                Click <b>Predict</b> to see the result here.
            </div>
            """,
            unsafe_allow_html=True
        )