import streamlit as st

st.set_page_config(
    page_title="Bemessungsapp",
    page_icon="🏗️",
    layout="wide"
)

st.title("Bemessung eines Einfeldträgers")

spannweite = st.number_input(
    "Spannweite L [m]",
    min_value=1.0,
    value=15.0,
    step=0.5
)

if st.button("Berechnen"):
    st.success(f"Die Spannweite beträgt {spannweite:.2f} m.")
