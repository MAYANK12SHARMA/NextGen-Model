import streamlit as st
import json
from streamlit_lottie import st_lottie

def load_lottie_file(file_path: str):
    with open(file_path, "r") as f:
        lottie_json = json.load(f)
        return lottie_json


def MainPageNavigation():
    
    with st.sidebar:
        # Apply the logo image
        st.image("./Visualization/Images/logo.png", width=200)
        

        # Navigation with selectbox
        st.sidebar.markdown("""
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
            <style>
                .dropdown {
                    position: relative;
                    display: inline-block;
                    margin-bottom: 10px;
                }
                
                .dropbtn {
                    background-color: #08ac24; /* Dark background */
                    color: white;
                    padding: 10px 20px;
                    font-size: 16px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    width: 100%;
                }

                .arrow {
                    margin-left: 100px;
                    font-size: 20px;
                    
                }

                .dropbtn:hover {
                    background-color: #1a1b24; /* Slightly lighter on hover */
                }
                .custom {
                    color: black;
                    font-size: 20px;
                }
            </style>
            <div class="dropdown">
                <a href="https://next-gen-hyf2gyg6cfb2cfar.canadacentral-01.azurewebsites.net" target="_blank">
                    <button class="dropbtn">Go to PlotWizard <span class="arrow"><i class="fas fa-up-right-from-square custom"></i></span></button>
                </a>
            </div>
            """, unsafe_allow_html=True)

                # Short description or instructions
        
        # Lottie Animation
        lottile_json = load_lottie_file("./Visualization/FilesJson/Animation2.json")
        st_lottie(lottile_json, speed=1, width=250, height=250, key="initial")


import streamlit as st
from streamlit_lottie import st_lottie

def load_lottie_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def MainPage():
    # Introduction Section
    Col1, col2 = st.columns([3, 1])
    with Col1:    
        st.markdown(
            """
            <style>
            @font-face {
                font-family: 'Algerian';
                src: url('font-family/Algerian-Regular.ttf') format('truetype');
            }

            .centered , h2 {
                text-align: center;
                font-family: 'Algerian', sans-serif;
                color: #F0F0FF;
                font-weight: 300;
            }
            
            .justified {
                text-align: justify;
                font-family: Arial, sans-serif;
            }
            </style>
            
            <div style="text-align: center;">
                <h2>👋 <b>Introduction</b></h2>
            </div>
            
            <div class="justified">
                Welcome to the No-Code Model Platform, a powerful and intuitive platform designed for predicting outputs from any tabular data, including time series!
                </br></br>
                Whether you’re a novice or an experienced data scientist, our Platform offers automated features and a rich library of models tailored for both classification and regression tasks.
                </br>
                
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        lottile_json = load_lottie_file("./Visualization/FilesJson/Animation.json")
        st_lottie(lottile_json, speed=52, width=300, height=200)

    # Divider
    st.markdown("---")

    # Features Section
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>🌟 <b>Features of the No-Code Model Platform</b></h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add the CSS styles for each section
    st.markdown(
        """
        <style>
        /* Title style */
        .title {
            text-align: center;
            font-size: 2em;
            font-weight: bold;
        }

        /* Subtitle style */
        .subtitle {
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;  /* Optional color styling */
            margin-bottom: 10px;
        }

        /* Content style */
        .content {
            text-align: justify;
            font-size: 1.1em;
            line-height: 1.6;
            margin: 0 0;
            width: 100%;  /* Optional width styling to make the content narrower */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Section 1: Model Flexibility
    with st.container():
        st.markdown(
            """
            <div class="subtitle">
                1. 🛠️ Model Flexibility (Classification & Regression)
            </div>
            <div class="content">
                ✅ <b>Predictive Power:</b> Capable of handling any tabular data, including time series, with support for both classification and regression tasks.
                <br><br>
                ⚙️ <b>Automated Customization:</b> Features automated model customization that allows users to fine-tune their models effortlessly.
                <br><br>
                📁 <b>Data Input:</b> Upload <b>CSV, Excel, JSON</b>, or use sample datasets.
                <br><br>
                
            </div>
            """,
            unsafe_allow_html=True
        )
          
    st.markdown("---")

    # Section 2: Visualization Options
    with st.container():
        st.markdown(
            '''
            <div class="subtitle">
                2. 📈 Advanced Modeling Techniques
            </div>
            ''', unsafe_allow_html=True) 
        col1, col2 = st.columns([2, 1]) 
             
        with col1:     
            st.markdown(
            """
            <div class="content">
                📊 <b>Optimized Performance:</b> Incorporates customized optimization techniques for enhanced model performance
                <br><br>
                🔄 <b>Bootstrapping & Stacking:</b> Automates bootstrapping and stacking of multiple models for robust predictions.
                <br><br>
            </div>
            """,
            unsafe_allow_html=True
        )
        with col2:
            st.write(" ")

    # Divider
    st.markdown("---")

    # Section 3: Dynamic Code Generation
    with st.container():
        st.markdown(
            """
            <div class="subtitle">
                3. 📊 Insightful Visualizations
            </div>
            <div class="content">
                🎨 <b>Visual Insights:</b> Generate insightful visualizations that help in understanding model performance and feature importance.
                <br><br>
                📊<b> Comprehensive Metrics:</b> Access detailed metrics such as AUC-ROC and precision-recall curves for thorough evaluation.
                <br><br>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Divider
    st.markdown("---")

    # Section 4: Advanced Features
    with st.container():
        st.markdown(
            '''
            <div class="subtitle">
                🚀 Deployment-Ready Solutions
            </div>
            ''', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(
            """
            <div class="content">
                ✅ <b>Best-Performing Model:</b> Automatically identifies the best-performing model, ensuring deployment readiness for real-world applications.
                <br><br>
                🧩 <b>User-Friendly Interface:</b> Simplifies the modeling process, allowing users to achieve accurate predictions without hassle.
                <br><br>
            </div>
            """,
            unsafe_allow_html=True
        )
            
        with col2:
           st.write(" ")

    # Conclusion / Footer
    st.markdown(
        """
        <div class="title" style="font-size:25px;">
            ✨ Start exploring the power of our No-Code Model Platform to achieve accurate predictions effortlessly today!
        """,
        unsafe_allow_html=True
    )


