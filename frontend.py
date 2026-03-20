import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Smart Tile AI", layout="wide")
st.title("Smart Tile AI: Discovery & Visualization")

# Professional Tabbed Navigation
tab1, tab2 = st.tabs(["🔍 Best Match Search", "🖼️ Multi-Surface Visualizer"])

# --- TAB 1: RAW IMAGE SEARCH (FIXED JSON PARSER) ---
with tab1:
    st.header("1. Find Best Matching Tile")
    st.write("Upload a room photo to see the AI analyze surfaces and fetch matching tiles from your database.")
    
    col1_search, col2_search = st.columns([1, 2])
    
    with col1_search:
        search_file = st.file_uploader("Upload Room Photo", type=["jpg", "png", "jpeg"], key="tab1_upload")
        
        if search_file:
            st.image(search_file, caption="Uploaded Room", use_container_width=True)

    with col2_search:
        if search_file:
            if st.button("Find Matching Tiles", type="primary"):
                with st.spinner("AI is analyzing surfaces and querying the database..."):
                    files = {"query_image": (search_file.name, search_file.getvalue(), search_file.type)}
                    
                    try:
                        res = requests.post("http://localhost:8000/search-tiles/", files=files)
                        
                        if res.status_code == 200:
                            data = res.json()
                            st.success("Matches Found!")
                            
                            # UPDATED HELPER: Now correctly reads the lists from the new JSON!
                            def display_match_card(title, matches_list):
                                if matches_list and len(matches_list) > 0:
                                    st.subheader(title)
                                    cols = st.columns(len(matches_list))
                                    
                                    for i, match in enumerate(matches_list):
                                        with cols[i]:
                                            img_url = match.get("image_url", "")
                                            if img_url:
                                                st.image(img_url, use_container_width=True)
                                            
                                            name = match.get("name", "Unknown Product")
                                            st.write(f"**{name}**")
                                            
                                            price = match.get("price_per_sqft", "Price Not Available")
                                            price_text = f"₹{price}/Sq.Ft." if isinstance(price, (int, float)) else price
                                            st.write(f"**Price:** {price_text}")
                                            
                                            score = match.get("match_score", 0)
                                            st.caption(f"Match: {score}%")
                                    
                                    st.divider()

                            # Grab the lists using the NEW keys we set in routes.py
                            wall_matches = data.get("wall_matches", [])
                            floor_matches = data.get("floor_matches", [])
                            overall_matches = data.get("overall_matches", [])

                            # Draw the UI!
                            display_match_card("🧱 Wall Tile Matches", wall_matches)
                            display_match_card("🪵 Floor Tile Matches", floor_matches)
                            
                            # Fallback if SAM AI failed to find any specific walls or floors
                            if not wall_matches and not floor_matches:
                                display_match_card("🎯 Overall Image Matches", overall_matches)

                        else:
                            st.error(f"Error {res.status_code}: Check your backend terminal.")
                    
                    except Exception as e:
                        st.error(f"Failed to connect to AI Server. Is Uvicorn running? Error: {e}")


# --- TAB 2: MULTI-SURFACE VISUALIZER (UNCHANGED & SAFE) ---
with tab2:
    st.header("2. Professional 3D Visualizer")
    st.write("Upload a room and choose separate tiles for different surfaces.")
    
    col1_vis, col2_vis = st.columns([1, 2])
    
    with col1_vis:
        st.subheader("Upload Parameters")
        room_f = st.file_uploader("Base Room Image", type=["jpg", "png", "jpeg"], key="tab2_room")
        floor_f = st.file_uploader("Floor Tile (Optional)", type=["jpg", "png", "jpeg"], key="tab2_floor")
        wall_f = st.file_uploader("Wall Tile (Optional)", type=["jpg", "png", "jpeg"], key="tab2_wall")

        if st.button("Generate 3D Visual", type="primary"):
            if not room_f:
                st.error("Please upload a base room image.")
            else:
                with st.spinner("Applying tiles..."):
                    vis_files = {"room": (room_f.name, room_f.getvalue(), room_f.type)}
                    if floor_f: 
                        vis_files["floor_tile"] = (floor_f.name, floor_f.getvalue(), floor_f.type)
                    if wall_f: 
                        vis_files["wall_tile"] = (wall_f.name, wall_f.getvalue(), wall_f.type)
                    
                    res = requests.post("http://localhost:8000/visualize/", files=vis_files)
                    
                    if res.status_code == 200:
                        st.session_state['vis_result'] = res.content
                    else:
                        st.error(f"Visualization failed (Error {res.status_code}).")

    with col2_vis:
        st.subheader("Visualized Result")
        if 'vis_result' in st.session_state:
            st.image(st.session_state['vis_result'], use_container_width=True)
            st.download_button(
                label="Download Visualized Room",
                data=st.session_state['vis_result'],
                file_name="visualized_room.png",
                mime="image/png"
            )
        else:
            st.info("Set your parameters on the left and click 'Generate'.")