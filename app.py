import streamlit as st
import tensorflow as tf
import joblib as jb
import pandas as pd
import numpy as np
import os



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FrostMatch | AI Ice Cream Recommender",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# THEME CUSTOMIZER INJECTION
# ============================================================

if "app_theme" not in st.session_state:
    st.session_state.app_theme = "🌙 Midnight Velvet (Dark)"

if st.session_state.app_theme == "🍧 Summer Sorbet (Light)":
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(255, 126, 95, 0.18), transparent 35%), radial-gradient(circle at 85% 20%, rgba(254, 180, 123, 0.22), transparent 35%), radial-gradient(circle at 50% 85%, rgba(255, 235, 170, 0.25), transparent 40%), #fffcf8;
        color: #2d3748;
    }
    .hero-title { font-family: 'Outfit', sans-serif; font-size: 4.8rem; font-weight: 800; text-align: center; background: linear-gradient(135deg, #ff512f 0%, #dd2476 50%, #ff8c00 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { text-align: center; color: #718096; font-size: 1.15rem; margin-bottom: 2.8rem; }
    .badge { display: inline-block; padding: 6px 18px; border-radius: 999px; background: rgba(255, 255, 255, 0.9); border: 1px solid rgba(255, 225, 215, 0.8); color: #dd2476; font-size: 0.8rem; font-weight: 700; }
    .glass-card { background: rgba(255, 255, 255, 0.88); backdrop-filter: blur(20px); border: 1px solid rgba(255, 235, 225, 0.9); border-radius: 24px; padding: 30px; box-shadow: 0 20px 40px rgba(221, 36, 118, 0.06); margin-bottom: 20px; }
    .section-title { font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; color: #1a202c; margin-bottom: 4px; }
    .section-description { color: #718096; font-size: 0.92rem; margin-bottom: 22px; }
    label { font-weight: 600 !important; color: #2d3748 !important; }
    div[data-baseweb="select"] > div { border-radius: 14px !important; border: 1px solid #fed7d7 !important; background: rgba(255, 255, 255, 0.95) !important; color: #1a202c !important; }
    .stButton > button { width: 100%; border: none; border-radius: 16px; padding: 0.9rem 1rem; font-size: 1.05rem; font-weight: 700; color: white; background: linear-gradient(135deg, #ff512f 0%, #dd2476 100%); box-shadow: 0 10px 25px rgba(255, 81, 47, 0.3); transition: all 0.25s ease; }
    .stat-card { text-align: center; background: rgba(255, 245, 245, 0.8); border-radius: 18px; padding: 16px; border: 1px solid #ffe3e3; }
    .stat-number { font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #ff512f, #dd2476); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stat-label { color: #718096; font-size: 0.75rem; font-weight: 700; }
    .rec-card { background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(255, 225, 215, 0.9); border-radius: 20px; padding: 24px; margin-top: 16px; box-shadow: 0 10px 30px rgba(255, 81, 47, 0.05); }
    .rec-rank { font-size: 0.78rem; font-weight: 800; color: #dd2476; text-transform: uppercase; }
    .rec-flavor { font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; color: #1a202c; }
    .star-row { font-size: 1.1rem; color: #d69e2e; }
    .progress-bg { background: #fff5f5; border-radius: 999px; height: 8px; width: 100%; margin-top: 14px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ff512f, #dd2476); }
    .footer { text-align: center; color: #a0aec0; margin-top: 50px; font-size: 0.85rem; }
    .score-txt { font-weight: 700; color: #4a5568; font-size: 0.95rem; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(168, 85, 247, 0.18), transparent 40%), radial-gradient(circle at 85% 20%, rgba(255, 67, 112, 0.18), transparent 40%), radial-gradient(circle at 50% 85%, rgba(59, 130, 246, 0.15), transparent 45%), #0b0f19;
        color: #f8fafc;
    }
    .hero-title { font-family: 'Outfit', sans-serif; font-size: 4.8rem; font-weight: 800; text-align: center; background: linear-gradient(135deg, #ff4370 0%, #c084fc 50%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { text-align: center; color: #94a3b8; font-size: 1.15rem; margin-bottom: 2.8rem; }
    .badge { display: inline-block; padding: 6px 18px; border-radius: 999px; background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.12); color: #c084fc; font-size: 0.8rem; font-weight: 700; }
    .glass-card { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 24px; padding: 30px; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4); margin-bottom: 20px; }
    .section-title { font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .section-description { color: #94a3b8; font-size: 0.92rem; margin-bottom: 22px; }
    label { font-weight: 600 !important; color: #cbd5e1 !important; }
    div[data-baseweb="select"] > div { border-radius: 14px !important; border: 1px solid rgba(255, 255, 255, 0.12) !important; background: rgba(30, 41, 59, 0.9) !important; color: #f8fafc !important; }
    .stButton > button { width: 100%; border: none; border-radius: 16px; padding: 0.9rem 1rem; font-size: 1.05rem; font-weight: 700; color: white; background: linear-gradient(135deg, #ff4370 0%, #a855f7 100%); box-shadow: 0 10px 30px rgba(255, 67, 112, 0.35); transition: all 0.25s ease; }
    .stat-card { text-align: center; background: rgba(30, 41, 59, 0.6); border-radius: 18px; padding: 16px; border: 1px solid rgba(255, 255, 255, 0.08); }
    .stat-number { font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #c084fc, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stat-label { color: #94a3b8; font-size: 0.75rem; font-weight: 700; }
    .rec-card { background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 24px; margin-top: 16px; box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35); }
    .rec-rank { font-size: 0.78rem; font-weight: 800; color: #c084fc; text-transform: uppercase; }
    .rec-flavor { font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; color: #f8fafc; }
    .star-row { font-size: 1.1rem; color: #fbbf24; }
    .progress-bg { background: rgba(30, 41, 59, 0.8); border-radius: 999px; height: 8px; width: 100%; margin-top: 14px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ff4370, #a855f7); }
    .footer { text-align: center; color: #64748b; margin-top: 50px; font-size: 0.85rem; }
    .score-txt { font-weight: 700; color: #cbd5e1; font-size: 0.95rem; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & DATA
# ============================================================

@st.cache_resource
def load_system():
    
    # Load trained model
    try:
        model = tf.keras.models.load_model("ice_cream_model.keras")
    except Exception:
        st.error(
            "Could not load 'ice_cream_model.keras'. "
            "Did you run the save code in your notebook?"
        )
        st.stop()

    # Load original flavor features
    item_features = pd.read_csv("ice_cream_flavors.csv")
    item_features_indexed = item_features.set_index("flavor")

    # Load custom flavors
    if os.path.exists("custom_flavors.csv"):
      

      custom_flavors = pd.read_csv("custom_flavors.csv")

      if not custom_flavors.empty:

        custom_flavors = custom_flavors.set_index("flavor")

        # Keep only the exact columns the original model uses
        custom_flavors = custom_flavors[
            item_features_indexed.columns

        ]

        item_features_indexed = pd.concat(
            [
                item_features_indexed,
                custom_flavors
            ]
        )
    return model, item_features_indexed

model, item_features_indexed = load_system()




# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

FEATURE_COLUMNS = [
    "sweetness",
    "chocolate_content",
    "fruity",
    "nutty",
    "creaminess",
    "has_mix_ins",
    "intensity",
    "coldness",
    "tartness",
    "spice_warmth",
    "floral_herbal",
    "texture_smoothness"
]


def add_custom_flavor(flavor_name, values):

    flavor_name = flavor_name.strip()

    if not flavor_name:
        return False, "Please enter a flavor name."

    # Check if flavor already exists
    if flavor_name in item_features_indexed.index:
        return False, "That flavor already exists."

    new_flavor = {
        "flavor": flavor_name,
        **values
    }

    # Add to custom CSV
    pd.DataFrame([new_flavor]).to_csv(
        "custom_flavors.csv",
        mode="a",
        header=False,
        index=False
    )

    return True, f"{flavor_name} added successfully."

def build_user_vector(flavor_rating_pair):
    total_weighted_vector = np.zeros(item_features_indexed.shape[1])
    total_abs_weight = 0

    for f, r in flavor_rating_pair:
        flavor_vec = item_features_indexed.loc[f].values
        weight = r - 3
        total_weighted_vector += flavor_vec * weight
        total_abs_weight += abs(weight)

        if total_abs_weight == 0:
           

        # all ratings were neutral (3) — fall back to a simple average
           rating = np.mean([item_features_indexed.loc[f].values for f, r in flavor_rating_pair], axis=0)
        else:
           rating = total_weighted_vector / total_abs_weight

    return rating.astype(np.float32)


def recommend_flavors(model, flavor_ratings_pairs, top_n=3):
    new_user_vector = build_user_vector(flavor_ratings_pairs)

    flavors_list = [flavor for flavor, rating in flavor_ratings_pairs]
    all_flavors = item_features_indexed.index.tolist()

    candidate = [f for f in all_flavors if f not in flavors_list]

    user_input = np.tile(new_user_vector, (len(candidate), 1))
    item_input = np.array([item_features_indexed.loc[f].values for f in candidate], dtype=np.float32)


    predictions = model.predict(
    [user_input, item_input],
    verbose=0).flatten()

    st.write("===== RAW MODEL PREDICTIONS =====")
    st.write(predictions)
    st.write("=================================")

    global_min, global_max = jb.load('prediction_range.pkl')

    if global_max != global_min:
        predictions = 1 + 4 * ((predictions - global_min) / (global_max - global_min))
    else:
        predictions = np.full_like(predictions, 3.0, dtype=np.float32)
    predictions = np.clip(predictions, 1, 5)

    results = list(zip(candidate, predictions))
    results.sort(key=lambda pair: pair[1], reverse=True)

    return results[:top_n]


# ============================================================
# TOP BAR / THEME SELECTOR
# ============================================================

top_col1, top_col2 = st.columns([3, 1])
with top_col2:
    theme_choice = st.selectbox(
        "🎨 UI Theme",
        options=["🌙 Midnight Velvet (Dark)", "🍧 Summer Sorbet (Light)"],
        index=0 if st.session_state.app_theme == "🌙 Midnight Velvet (Dark)" else 1,
        label_visibility="collapsed"
    )

if theme_choice != st.session_state.app_theme:
    st.session_state.app_theme = theme_choice
    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="text-align:center;">
        <span class="badge">✦ AI-POWERED FLAVOR DISCOVERY</span>
    </div>
    <div class="hero-title">FrostMatch</div>
    <div class="hero-subtitle">Tell us what you love. Our neural network finds your next favorite scoop.</div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN SECTION
# ============================================================

left, right = st.columns([1.25, 0.75], gap="large")


# ============================================================
# FLAVOR INPUT
# ============================================================

with left:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">🍨 Build your flavor profile</div>
            <div class="section-description">Choose flavors you have tried before, then tell us how much you liked each one.</div>
            """,
            unsafe_allow_html=True
        )

        all_flavors = item_features_indexed.index.tolist()

        selected_flavors = st.multiselect(
    "Flavors you've tried",
    options=all_flavors,
    key="flavor_multiselect",
    placeholder="Search and select flavors...")

        ratings = {}

        if selected_flavors:


            st.markdown(
        "<br><b>How did you rate them?</b>",
        unsafe_allow_html=True)

            for flavor in selected_flavors:

                ratings[flavor] = st.slider(
            flavor,
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.5,
            key=f"rating_{flavor}"
        )

        else:

            st.info(
        "Select at least one flavor to create your taste profile.")

# ============================================================
# CUSTOM FLAVOR
# ============================================================

with st.expander("🍨 Can't find your flavor? Add your own"):

    st.markdown(
        """
        <div class="section-title">Create a custom flavor</div>
        <div class="section-description">
        Add a flavor that isn't already in FrostMatch.
        </div>
        """,
        unsafe_allow_html=True
    )

    custom_name = st.text_input(
        "Flavor name",
        placeholder="e.g. Pistachio Caramel Crunch"
    )

    st.markdown("### Flavor characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:

        sweetness = st.slider(
            "Sweetness",
            0.0, 5.0, 3.0, 0.5
        )

        chocolate_content = st.slider(
            "Chocolate content",
            0.0, 5.0, 0.0, 0.5
        )

        fruity = st.slider(
            "Fruity",
            0.0, 5.0, 0.0, 0.5
        )

        nutty = st.slider(
            "Nutty",
            0.0, 5.0, 0.0, 0.5
        )

    with col2:

        creaminess = st.slider(
            "Creaminess",
            0.0, 5.0, 3.0, 0.5
        )

        has_mix_ins = st.checkbox(
            "Contains mix-ins"
        )

        intensity = st.slider(
            "Intensity",
            0.0, 5.0, 3.0, 0.5
        )

        coldness = st.slider(
            "Coldness",
            0.0, 5.0, 3.0, 0.5
        )

    with col3:

        tartness = st.slider(
            "Tartness",
            0.0, 5.0, 0.0, 0.5
        )

        spice_warmth = st.slider(
            "Spice / warmth",
            0.0, 5.0, 0.0, 0.5
        )

        floral_herbal = st.slider(
            "Floral / herbal",
            0.0, 5.0, 0.0, 0.5
        )

        texture_smoothness = st.slider(
            "Texture smoothness",
            0.0, 5.0, 4.0, 0.5
        )

    add_flavor = st.button(
        "➕ Add Flavor",
        key="add_custom_flavor"
    )

    if add_flavor:

        values = {
            "sweetness": sweetness,
            "chocolate_content": chocolate_content,
            "fruity": fruity,
            "nutty": nutty,
            "creaminess": creaminess,
            "has_mix_ins": int(has_mix_ins),
            "intensity": intensity,
            "coldness": coldness,
            "tartness": tartness,
            "spice_warmth": spice_warmth,
            "floral_herbal": floral_herbal,
            "texture_smoothness": texture_smoothness
        }

        success, message = add_custom_flavor(
            custom_name,
            values
        )

        if success:
            st.success(message)

            # Clear cached model/data
            st.cache_resource.clear()

            # Reload everything
            st.rerun()

        else:
            st.error(message)

# ============================================================
# SETTINGS
# ============================================================

with right:
    with st.container(border=True):
        st.markdown(
            """
            <div class="section-title">⚙️ Recommendation settings</div>
            <div class="section-description">Choose how many flavors FrostMatch should discover.</div>
            """,
            unsafe_allow_html=True
        )

        top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5, step=1)
        st.markdown("<br>", unsafe_allow_html=True)

        selected_count = len(selected_flavors)
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{selected_count}</div>
                <div class="stat-label">FLAVORS ANALYZED</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        generate = st.button(" Find My Perfect Flavors")


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

if generate:
    if not selected_flavors:
        st.error("Please select at least one flavor before generating recommendations.")
    else:
        flavor_rating_pairs = [(flavor, ratings[flavor]) for flavor in selected_flavors]

        with st.spinner("🧠 Analyzing your taste profile..."):
            recommendations = recommend_flavors(model, flavor_rating_pairs, top_n=top_n)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="text-align:center;">✨ Your Perfect Flavor Matches</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-description" style="text-align:center;">Based on your ratings, these flavors are predicted to match your taste best.</div>', unsafe_allow_html=True)

        for rank, (flavor, predicted_rating) in enumerate(recommendations, start=1):
            predicted_rating = float(predicted_rating)
            display_rating = max(0, min(5, predicted_rating))

            progress_width = (display_rating / 5.0) * 100

            full_stars = int(display_rating)
            half_star = (display_rating - full_stars >= 0.5)
            
            stars = "★" * full_stars
            if half_star: stars += "½"
            stars += "☆" * (5 - full_stars - (1 if half_star else 0))

            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-header">
                        <span class="rec-rank">MATCH #{rank}</span>
                    </div>
                    <div class="rec-flavor">🍦 {flavor}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="star-row">{stars}</span>
                        <span class="score-txt">{predicted_rating:.2f} / 5.0</span>
                    </div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width: {progress_width}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

else:
    st.markdown(
        """
        <div style="text-align:center; margin-top:50px; color:#64748b;">
            <div style="font-size:4rem; margin-bottom:10px;">🍨</div>
            <div style="font-family:'Outfit',sans-serif; font-size:1.5rem; font-weight:700; color:#cbd5e1;">Your next favorite flavor is waiting.</div>
            <div style="margin-top:6px;">Select some flavors above to begin discovery.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="footer">
        FrostMatch · Neural Recommendation Engine · Built with TensorFlow & Streamlit<br>
        <span style="font-weight:700; color:#a855f7; margin-top:6px; display:inline-block;">
            ✨ Created by <b>Varad Gaikwad</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)