import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Smart Tile AI", layout="wide")
st.title("Smart Tile AI: Discovery & Visualization")

# Professional Tabbed Navigation
tab1, tab2 = st.tabs(["🔍 Best Match Search", "🖼️ Multi-Surface Visualizer"])

# --- TAB 1: RAW IMAGE SEARCH ---
with tab1:
    st.header("1. Find Best Matching Tile")
    st.write("Upload a room photo to see the single best matching tile image from your catalog.")
    
    col1_search, col2_search = st.columns(2)
    
    with col1_search:
        search_file = st.file_uploader("Upload Room Photo", type=["jpg", "png", "jpeg"], key="tab1_upload")
        if search_file:
            st.image(search_file, caption="Uploaded Room", use_container_width=True)

    with col2_search:
        if search_file:
            if st.button("Find Matching Tile", type="primary"):
                with st.spinner("AI is analyzing..."):
                    files = {"query_image": (search_file.name, search_file.getvalue(), search_file.type)}
                    # This hits the endpoint that returns raw image bytes
                    res = requests.post("http://localhost:8000/search-tiles/", files=files)
                    
                    if res.status_code == 200:
                        st.success("Best Match Found!")
                        st.image(res.content, caption="Best Match from Catalog", use_container_width=True)
                    else:
                        st.error(f"Error {res.status_code}: Check your backend terminal.")

# --- TAB 2: MULTI-SURFACE VISUALIZER ---
with tab2:
    st.header("2. Professional 3D Visualizer")
    st.write("Upload a room and choose separate tiles for different surfaces.")
    
    col1_vis, col2_vis = st.columns([1, 2])
    
    with col1_vis:
        st.subheader("Upload Parameters")
        # Matches Swagger UI parameters: room, floor_tile, wall_tile
        room_f = st.file_uploader("Base Room Image", type=["jpg", "png", "jpeg"], key="tab2_room")
        floor_f = st.file_uploader("Floor Tile (Optional)", type=["jpg", "png", "jpeg"], key="tab2_floor")
        wall_f = st.file_uploader("Wall Tile (Optional)", type=["jpg", "png", "jpeg"], key="tab2_wall")

        if st.button("Generate 3D Visual", type="primary"):
            if not room_f:
                st.error("Please upload a base room image.")
            else:
                with st.spinner("Applying tiles..."):
                    # Prepare multi-part form data exactly like Swagger
                    vis_files = {"room": (room_f.name, room_f.getvalue(), room_f.type)}
                    if floor_f: 
                        vis_files["floor_tile"] = (floor_f.name, floor_f.getvalue(), floor_f.type)
                    if wall_f: 
                        vis_files["wall_tile"] = (wall_f.name, wall_f.getvalue(), wall_f.type)
                    
                    # Hit the visualize endpoint
                    res = requests.post("http://localhost:8000/visualize/", files=vis_files)
                    
                    if res.status_code == 200:
                        st.session_state['vis_result'] = res.content
                    else:
                        st.error(f"Visualization failed (Error {res.status_code}).")

    with col2_vis:
        st.subheader("Visualized Result")
        if 'vis_result' in st.session_state:
            # Display raw image bytes directly
            st.image(st.session_state['vis_result'], use_container_width=True)
            st.download_button(
                label="Download Visualized Room",
                data=st.session_state['vis_result'],
                file_name="visualized_room.png",
                mime="image/png"
            )
        else:
            st.info("Set your parameters on the left and click 'Generate'.")