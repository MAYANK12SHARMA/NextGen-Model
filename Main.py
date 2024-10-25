import streamlit as st
from streamlit_option_menu import option_menu
import json
from streamlit_lottie import st_lottie

from Visualization.Home import MainPage, MainPageNavigation
from Visualization.Modelling import Modelling


#? ================================== Additional part  ==================================================
 
st.set_page_config(page_title="Visualization Tool", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .css-18e3th9 {
        padding-top: 0 !important;
    }
    .css-1d391kg {
        padding-top: 0 !important;
    }
    .st-emotion-cache-13ln4jf{
        padding-top: 0 !important;
    }
    .st-emotion-cache-1jicfl2{
        padding-top: 1rem !important;
    }
    .st-emotion-cache-kgpedg{
        padding: 0 !important;
    }
    # header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

def load_lottie_file(file_path: str):
    with open(file_path, "r") as f:
        lottie_json = json.load(f)
        return lottie_json


#? =============================== Navbar ================================== 
def top_nav_menu():
    selected = option_menu(
        menu_title=None,  # No title for the horizontal menu
        options=["Home", "No-Code Modelling"],  # menu options
        icons=["house", "robot"],  # icons for each option (using example names)
        menu_icon="cast",  # optional menu icon
        default_index=0,  # default selected option
        orientation="horizontal",  # horizontal navigation bar at the top
        styles={
            "container": {
                "padding": "0", "margin": "0",
                "width": "100%",  # Set navbar width to 100% of the page
                "height": "40px",  # Set navbar height
                "top": "0", "position": "sticky", "z-index": "999",
            },
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#404040"},
            "icon": {"color": "orange", "font-size": "20px","padding-bottom": "2px"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
    )
    return selected

#! ============================================= Pages =============================================

#? ========================================= Code Generation Page ================================== 



def main():
    st.write("<h1 style='text-align: center;'>No-Code Modelling</h1>", unsafe_allow_html=True)
    
    selected = top_nav_menu()  

    # Route to different pages based on user selection
    if selected == "Home":
        MainPageNavigation()   
        MainPage()
 
    elif selected == "No-Code Modelling":
        Modelling()
        
# Run the main function
if __name__ == "__main__":
    main()
