# streamlit_app.py
import streamlit as st
import pandas as pd
from app import (
    load_and_preprocess,
    compute_top_k_matches,
    INTERNAL_NAME_COL,
    INTERNAL_TEXT_COL,
    INTERNAL_DEPT_COL,
    EXTERNAL_NAME_COL,
    EXTERNAL_TEXT_COL,
)

# Page config (use relative path for icon)
st.set_page_config(
    page_title="Office of Research & Innovation",
    page_icon="Humber.png",  # place this image in your repo folder
    layout="wide"
)

# Humber colors
HUMBER_DARK_BLUE = "#000033"
HUMBER_LIGHT_GREEN = "#B4C800"
HUMBER_LIGHT_GREY = "#F5F5F5"

# UI Header
st.markdown(f"""
<div style="
    text-align:center; padding:30px; 
    background-color:{HUMBER_DARK_BLUE}; color:white; border-radius:10px;
    margin-bottom: 30px;">
    <h1 style="margin-bottom:10px;">Bridging Minds, Building Futures</h1>
    <p style="margin-top:0; font-size:18px;">Discover top research matches powered by Sentence-Transformers embeddings</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: Upload Files
with st.sidebar:
    st.header("Upload Your Files")
    internal_file = st.file_uploader("📂 Internal Researchers", type="csv")
    external_file = st.file_uploader("📂 External Researchers", type="csv")


# --- Cache heavy computation ---
@st.cache_data(show_spinner=False)
def cached_compute(internal_df, external_df, in_name, in_text, in_dept, ex_name, ex_text, k):
    return compute_top_k_matches(
        internal_df, external_df,
        in_name, in_text, in_dept,
        ex_name, ex_text,
        k=k
    )


def _map_column(requested_name: str, df: pd.DataFrame) -> str:
    if requested_name in df.columns:
        return requested_name
    req = requested_name.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == req:
            return col
    raise ValueError(
        f"Could not find column '{requested_name}' in uploaded file. Available columns: {list(df.columns)}")


# Main panel
if internal_file and external_file:
    try:
        internal_df = pd.read_csv(internal_file)
        external_df = pd.read_csv(external_file)

        internal_df.columns = [str(c).strip() for c in internal_df.columns]
        external_df.columns = [str(c).strip() for c in external_df.columns]

        in_name_col = _map_column(INTERNAL_NAME_COL, internal_df)
        in_text_col = _map_column(INTERNAL_TEXT_COL, internal_df)
        try:
            in_dept_col = _map_column(INTERNAL_DEPT_COL, internal_df)
        except ValueError:
            in_dept_col = None

        ex_name_col = _map_column(EXTERNAL_NAME_COL, external_df)
        ex_text_col = _map_column(EXTERNAL_TEXT_COL, external_df)

        internal_df, in_name, in_text, in_dept = load_and_preprocess(
            internal_df, in_name_col, in_text_col, in_dept_col
        )
        external_df, ex_name, ex_text, _ = load_and_preprocess(
            external_df, ex_name_col, ex_text_col
        )

        if st.button("Compute Matches"):
            with st.spinner("Computing matches..."):
                large_k = max(1, len(internal_df))
                full_result_df = cached_compute(
                    internal_df, external_df,
                    in_name, in_text, in_dept,
                    ex_name, ex_text,
                    k=large_k
                )
            if full_result_df.empty:
                st.warning("No matches found.")
            else:
                st.session_state['full_result_df'] = full_result_df

        if 'full_result_df' in st.session_state:
            result_df = st.session_state['full_result_df']

            col1, col2 = st.columns([1, 1])
            with col1:
                top_k = st.number_input("Top K Matches", min_value=1, max_value=5, value=3, step=1)
            with col2:
                sim_threshold = st.number_input("Minimum Similarity", min_value=0.0, max_value=1.0, value=0.0,
                                                step=0.01, format="%.2f")

            filtered_df = result_df[result_df['similarity_score'] >= sim_threshold]
            filtered_df = filtered_df.groupby('external_name').head(top_k).reset_index(drop=True)

            display_df = filtered_df.copy()
            display_df['external_name'] = display_df['external_name'].str.title()
            display_df['best_internal_match'] = display_df['best_internal_match'].str.title()
            display_df['internal_department'] = display_df['internal_department'].str.title()
            display_df['similarity_score'] = display_df['similarity_score'].round(4)

    

            try:
                st.dataframe(styled_df, use_container_width=True)
            except:
                st.write(display_df)

            csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 Download Matches as CSV", data=csv, file_name="top_matches.csv", mime="text/csv")

            st.markdown(f"""
                <div style="text-align:center; padding:15px; margin-top:40px; 
                            background-color:{HUMBER_LIGHT_GREY}; color:#555555; border-radius:5px;">
                    Office of Research & Innovation
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload both internal and external researcher CSV files to proceed.")
