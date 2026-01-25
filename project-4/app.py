import streamlit as st
from PIL import Image
from src.indexer import SearchEngine

# Page configuration
st.set_page_config(layout="wide", page_title="CLIP Semantic Search")

@st.cache_resource
def load_engine():
    return SearchEngine()

engine = load_engine()

st.title("CLIP-based Semantic Search System")
st.markdown("Search images using Text or Image queries without retraining.")

# Sidebar for options
st.sidebar.header("Search Options")
top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=10, value=5)
mode = st.sidebar.radio("Query Mode", ["Text to Image", "Image to Image"])

query_content = None

# UI for input
if mode == "Text to Image":
    text_query = st.text_input("Enter a text description:", placeholder="e.g., a dog playing in the grass")
    if text_query:
        query_content = text_query
        
elif mode == "Image to Image":
    uploaded_file = st.file_uploader("Upload an query image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_query = Image.open(uploaded_file).convert("RGB")
        st.image(image_query, caption="Query Image", width=250)
        query_content = image_query

# Perform search and display results
if query_content:
    if st.button("Search") or (mode == "Text to Image" and query_content):
        with st.spinner("Searching..."):
            # Search
            search_mode = 'text' if mode == "Text to Image" else 'image'
            results = engine.search(query_content, top_k=top_k, mode=search_mode)

        st.subheader(f"Top {top_k} Search Results")
        
        # Display results in columns
        cols = st.columns(top_k)
        for i, res in enumerate(results):
            with cols[i]:
                img = Image.open(res['path'])
                st.image(img, use_container_width=True)
                st.caption(f"Score: {res['score']:.4f}")