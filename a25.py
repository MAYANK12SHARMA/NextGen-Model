import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas.plotting import parallel_coordinates
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
from PIL import Image
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import base64
from reportlab.lib.utils import ImageReader
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf




import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression,ridge_regression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor,AdaBoostClassifier,AdaBoostRegressor,StackingClassifier,StackingRegressor

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,LocalOutlierFactor
from sklearn.svm import SVC,SVR,OneClassSVM
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error,r2_score,confusion_matrix,classification_report,precision_recall_curve,precision_score,recall_score,roc_auc_score
import pickle
import io
from sklearn.model_selection import GridSearchCV
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from matplotlib.backends.backend_pdf import PdfPages
import PyPDF2
import shap
shap.initjs()

from PyPDF2 import PdfMerger


from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, roc_curve, precision_recall_curve
)

import streamlit as st

def main():
    st.sidebar.header("Automated ML")
    data = st.file_uploader("Upload a Dataset", type=["csv", "xlsx"])
    if data is not None:
                # Handle Excel files
                if data.name.endswith('xlsx'):
                    df = pd.read_excel(data)
                else:
                    df = pd.read_csv(data)
                st.dataframe(df.head())
                pdf_list = []
                info=""
                def create_pdf(info_text):
                            pdf_buffer = io.BytesIO()
                            c = canvas.Canvas(pdf_buffer, pagesize=letter)
                            width, height = letter

                            # Draw text on PDF
                            text = c.beginText(40, height - 40)  # Start position
                            text.setFont("Helvetica", 12)

                            for line in info_text.split('\n'):
                                if line.strip():  # Ensure line is not empty
                                    text.textLine(line)

                            c.drawText(text)
                            c.showPage()
                            c.save()

    # Reset the buffer's position to the beginning
                            pdf_buffer.seek(0)

                            return pdf_buffer
                        

                        
                        
                
                
                
                
                
                
                
                
                
                
                
                
                    
                    
                if st.sidebar.checkbox("Remove Columns"):
                
                    selected_columns = st.sidebar.multiselect("Select Columns to Remove", df.columns.to_list())
                    df = df.drop(selected_columns, axis=1)
                    st.write(df.head())
                # if st.sidebar.checkbox("Encode Selected Categorical Columns"):
                #     categorical_columns = st.sidebar.multiselect("Select Categorical Columns to Encode", df.select_dtypes(include=['object']).columns.to_list())
                
                
                
                def encode_categorical_columns(df, categorical_columns):
        # Check if any columns are selected for encoding
                    if len(categorical_columns) > 0:
                        label_encoder = LabelEncoder()
                        for col in categorical_columns:
                            df[col] = label_encoder.fit_transform(df[col])

            # Show success message and display the head of the encoded DataFrame
                        st.success(f"Successfully encoded columns: {categorical_columns}")
                        st.write(df.head())
                    else:
                        st.error("Please select at least one column to encode.")
        
                    return df

    
        #         if st.sidebar.checkbox("Encode Selected Categorical Columns"):
        
        #             categorical_columns = st.sidebar.multiselect(
        #     "Select Categorical Columns to Encode", 
        #     df.select_dtypes(include=['object']).columns.to_list()
        # )
        
        # # If the user clicks the "Encode" button, perform encoding
        #             if st.sidebar.button("Encode"):
        #                 df = encode_categorical_columns(df, categorical_columns)
        #                 a=df
                        
                        
                       
                
                st.sidebar.write("---------------------------------------------------------------------------")
                if st.sidebar.checkbox("EDA"):
                        pdf_list=[]
                        plots=[]
                        
                        
                                                
                     
                   
                        if st.checkbox("Show Shape"):
                            st.write(f"Shape of the dataset: {df.shape}")
                            shape_info = f"Shape of the dataset: {df.shape}\n"
                            info += shape_info
                            info+="-----------------------------------------------------------------------------"+"\n"
                            
                            fig=df.shape
    #               

                        if st.checkbox("Show Columns"):
                            all_columns = df.columns.to_list()
                            columns_info = f"Columns: {', '.join(all_columns)}\n"
                            info += columns_info
                            info+="-----------------------------------------------------------------------------"+"\n"
                            st.write(all_columns)
    #                       
                        
                        if st.checkbox("Summary"):
                            summary_info = df.describe().to_string() + "\n"
                            info += "Summary Statistics:\n" + summary_info
                            info+="-----------------------------------------------------------------------------"+"\n"
                            st.write(df.describe())
    #                      

                        if st.checkbox("Show Null Values"):
                            null_info = df.isnull().sum().to_string() + "\n"
                            info += "Null Values:\n" + null_info
                            info+="-----------------------------------------------------------------------------"+"\n"
                            st.write(df.isnull().sum())
                        
                        
    #                       
                        if st.checkbox("Show Selected Columns"):
                            selected_columns = st.multiselect("Select Columns", df.columns.to_list())
                            new_df = df[selected_columns]
                            st.dataframe(new_df)
        #                   

                        if st.checkbox("Show Value Counts"):
                            value_counts_info = df.iloc[:, -1].value_counts().to_string() + "\n"
                            info += "Value Counts:\n" + value_counts_info
                            info+="-----------------------------------------------------------------------------"+"\n"
                            st.write(df.iloc[:, -1].value_counts())
                        
                        if st.checkbox("Show unique value"):
                           
                            info+="-----------------------------------------------------------------------------"+"\n"
                        
                            st.write(df.nunique())
                        
                        def create_pdf(info_text):
                            pdf_buffer = io.BytesIO()
                            c = canvas.Canvas(pdf_buffer, pagesize=letter)
                            width, height = letter

                            # Draw text on PDF
                            text = c.beginText(40, height - 40)  # Start position
                            text.setFont("Helvetica", 12)

                            for line in info_text.split('\n'):
                                if line.strip():  # Ensure line is not empty
                                    text.textLine(line)

                            c.drawText(text)
                            c.showPage()
                            c.save()

    # Reset the buffer's position to the beginning
                            pdf_buffer.seek(0)

                            return pdf_buffer
                        

                        
                        # if st.button("Generate PDF"):
                               
                        #       # Ensure there's content to write to the PDF
                        #         pdf_buffer = create_pdf(info)
                                
                        #         # Create a download button for the generated PDF
                        #         st.download_button(
                        #             label="Download PDF",
                        #             data=pdf_buffer,
                        #             file_name="report.pdf",
                        #             mime="application/pdf"
                        #         )
                                     
                        
                    
                            
                        
                        
                        
                        def create_correlation_plot(df):
                            for col in df.columns:
                                label_encoder=LabelEncoder()
                                df[col] = label_encoder.fit_transform(df[col])
                                
                            fig, ax = plt.subplots(figsize=((len(df.columns)), (len(df.columns))))
                            sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
                            st.pyplot(fig)
                            return fig
                        
                        def create_pie_plot(df, column_to_plot):
                            pie_data = df[column_to_plot].value_counts()
                            fig, ax = plt.subplots()
                            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=140)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                            st.pyplot(fig)
                            return fig
                        def save_fig_to_pdf(fig):
                            buf = BytesIO()
                            fig.savefig(buf, format="pdf")  # Save the figure as PDF to buffer
                            buf.seek(0)  # Move to the beginning of the buffer
                            return buf

                        
                        
                        
                        
                        def merge_pdfs(pdf_files):
                            merger = PdfMerger()

                            for pdf_file in pdf_files:
                                merger.append(pdf_file)

                            merged_pdf = BytesIO()
                            merger.write(merged_pdf)
                            merger.close()
                              # Reset buffer position to the beginning
                            return merged_pdf
                        
                        
                       
                        
                        
                        if st.checkbox("Correlation Plot (Seaborn)"):
                            correlation_fig = create_correlation_plot(df)
                            st.write(df.head())
                            pdf_list.append(save_fig_to_pdf(correlation_fig))
                            
                        
                        
                        if st.checkbox("Pie Plot"):
                            unique_value_counts = df.nunique()
                            filtered_columns = unique_value_counts[unique_value_counts < 12].index.tolist()
                            all_columns = df.columns.to_list()
                            column_to_plot = st.selectbox("Select 1 Column for Pie Plot",filtered_columns)
                            pie_fig = create_pie_plot(df, column_to_plot)
                            
                            pdf_list.append(save_fig_to_pdf(pie_fig))
                       
                        
                        merged_pdf_data = merge_pdfs(pdf_list)
                            
                        
                        
                        
        #                 if st.checkbox("Correlation Plot (Seaborn)"):
        #                     fig, ax = plt.subplots(figsize=(10,8))
        #                     sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
        #                     st.pyplot(fig)
        #                     buf = BytesIO()
        #                     fig.savefig(buf, format="pdf")  # Save the figure as PDF to buffer
        #                     buf.seek(0)  # Move to the beginning of the buffer

        # # Append the buffer to the list
        #                     pdf_list.append(buf)
                        
        #                 if st.checkbox("Pie Plot"):
        #                     all_columns = df.columns.to_list()
        #                     column_to_plot = st.selectbox("Select 1 Column for Pie Plot", all_columns)
        #                     pie_data = df[column_to_plot].value_counts()
        #                     fig, ax = plt.subplots()
        #                     ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=140)
        #                     ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        #                     st.pyplot(fig)
        #                     buf = BytesIO()
        #                     fig.savefig(buf, format="pdf")  # Save the figure as PDF to buffer
        #                     buf.seek(0)  # Move to the beginning of the buffer

        # # Append the buffer to the list
        #                     pdf_list.append(buf)
                            
                            
        #                 # if st.checkbox("imputations"):
        #                 #     st.write(df.isnull().sum())
        #                 #     i=st.selectbox("mean","median","mode")
                        
                        
        #                 def merge_pdfs(pdf_files):
        #                     merger = PyPDF2.PdfMerger()

        #                     # Iterate through the list of PDF files
        #                     for pdf_file in pdf_files:
        #                         # For each PDF, append it to the merger object
        #                         merger.append(pdf_file)

        #                     # Output the merged PDF into a buffer
        #                     merged_pdf = BytesIO()
        #                     merger.write(merged_pdf)
        #                     merger.close()

        #                     # Reset the buffer's position to the beginning
        #                     merged_pdf.seek(0)

        #                     return merged_pdf

                        
        #                 merged_pdf_data = merge_pdfs([pdf.read() for pdf in pdf_list])

    # Provide a download button for the merged PDF
                        # st.download_button(
                        #     label="Download Merged PDF",
                        #     data=merged_pdf_data,
                        #     file_name="merged_output.pdf",
                        #     mime="application/pdf"
                        # )
                        
                        
                    
                            
                
        
                                
                    
                st.sidebar.write("---------------------------------------------------------------------------")
                
                if st.sidebar.checkbox("Generate Plot"):
                    plots=["histogram","scatterplot","cumulative distribution plots","density plot"]
                    a=st.selectbox("choose any plot",plots)
                    st.write(a)
                    numerical_cols=df.select_dtypes(include=np.number).columns.to_list()
                    if(a=="histogram"):
                        
                        
                        if st.checkbox("custom plots"):
                            a=st.selectbox("selct numerical columns",numerical_cols)
                            fig,ax=plt.subplots()
                            sns.histplot(df[a],kde=True,ax=ax)
                            ax.set_title(f"distribution of{a}")
                            st.pyplot(fig)
                        if st.checkbox("automatic plot"):
                            
                            for col in numerical_cols:
                                fig, ax = plt.subplots()
                                sns.histplot(df[col], kde=True, ax=ax)
                                ax.set_title(f"Distribution of {col}")
                                st.pyplot(fig)
                                # pdf_list.append(save_fig_to_pdf(fig))
                                # merged_pdf_data = merge_pdfs(pdf_list)
                            

                        
                    
                        
                       
                    if(a=="scatterplot"):
                        if st.checkbox("custom plots"):
                            
                            xaxis=st.selectbox("selct numerical columns",numerical_cols,key="Xaxisselct")
                            yaxis=st.selectbox("selct numerical columns",numerical_cols)
                            fig,ax=plt.subplots()
                            sns.scatterplot(x=df[xaxis],y=df[yaxis],ax=ax,hue=df[xaxis],palette='viridis')
                            ax.set_title(f"scatterplot betwwen the{xaxis} and {yaxis}")
                            st.pyplot(fig)
                            # pdf_list.append(save_fig_to_pdf(fig))
                            # merged_pdf_data = merge_pdfs(pdf_list)
                            
                            
                            
                            
                            
                            
                        if st.checkbox("automated plot"):
                            for i, col1 in enumerate(numerical_cols):
                                for col2 in numerical_cols[i+1:]:
                                    fig, ax = plt.subplots()
                                    sns.scatterplot(x=df[col1], y=df[col2], ax=ax, hue=df[col1], palette='viridis')
                                    ax.set_title(f"Scatterplot between {col1} and {col2}")
                                    st.pyplot(fig)
                    if(a=='density plot'):
                        for col in numerical_cols:
                            fig, ax = plt.subplots()
                            sns.kdeplot(df[col], shade=True, ax=ax, color='orange')
                            ax.set_title(f"Density Plot of {col}")
                            st.pyplot(fig)
                    if(a=="cumulative distribution plots"):
                        for col in numerical_cols:
                            fig, ax = plt.subplots()
                            sns.ecdfplot(df[col], ax=ax, color='purple')
                            ax.set_title(f"CDF of {col}")
                            st.pyplot(fig)
                            
                            
                    # st.download_button(
                    #         label="Download Merged PDF",
                    #         data=merged_pdf_data,
                    #         file_name="merged_output.pdf",
                    #         mime="application/pdf",
                    #         key="a"
                    #     )
                            
                            
                            
                            
                            
                                
                               
                            
                         
                
                        
                
                st.sidebar.write("---------------------------------------------------------------------------")
                if st.sidebar.checkbox("important features"):
                    try:
                        all_columns = df.columns.to_list()
                        target= st.sidebar.selectbox("Select Target Column", all_columns)
                        feature_columns =  [col for col in all_columns if col != target]
                        label_encoder=LabelEncoder()
                        for col in feature_columns:
                            df[col] = label_encoder.fit_transform(df[col])
                        X = df[feature_columns]
                        df[target] = label_encoder.fit_transform(df[target])
                        y = df[target]
                        imp_feature={"random forest","correlation"}
                        a=st.selectbox("features selction",imp_feature)
                        if (a=="random forest"):
                            # X = pd.get_dummies(X, drop_first=True)

                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
                            randomforest=RandomForestClassifier(n_estimators=100)
                            randomforest.fit(X_train,y_train)
                            selected_features= pd.Series(randomforest.feature_importances_, index=X_train.columns).sort_values(ascending=False).index
                            abcd=selected_features= pd.Series(randomforest.feature_importances_, index=X_train.columns).sort_values(ascending=False).index
                            info+="important features"+abcd
                            st.write(selected_features)
                            

                            
                            
                            st.write("---------------------------------------------------------------------------")
                            if st.checkbox("shap value"):
                                explainer = shap.Explainer(randomforest)  # Create SHAP explainer
                                shap_values = explainer(X_train)  # Get the SHAP values

                            

                                # Create and display the waterfall plot
                                fig, ax = plt.subplots() 
                                shap.plots.waterfall(shap_values[0, :, 0])
                                st.pyplot(fig)
                                
                                
                                # Option 1: Select SHAP values for a specific class
                                fig, ax = plt.subplots() 
                                shap.summary_plot(shap_values[:, :, 0], X_train)  # SHAP values for class 0

                                st.pyplot(fig)
                        if(a=="correlation"):
                            def create_correlation_plot(df):
                                for col in df.columns:
                                    label_encoder=LabelEncoder()
                                    df[col] = label_encoder.fit_transform(df[col])
                                    
                                fig, ax = plt.subplots(figsize=((len(df.columns)), (len(df.columns))))
                                sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
                                st.pyplot(fig)
                                return fig
                            correlation_fig = create_correlation_plot(df)
                            
                            
                        
                        
                        
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                          
                        
                        
                        
                        

                        
                        
        #                 if len(selected_features) > 0:
        #                     X_train_selected = X_train[selected_features]
        #                     X_test_selected = X_test[selected_features]
        #                     explainer=shap.Explainer(randomforest,X_train_selected)
        #                     shap_values_test=explainer(X_test_selected)
                            
        #                     def plot_shap_summary_and_bar(shap_values, X, feature_names, title_suffix):
        # # SHAP Summary Plot
        #                         st.write(f"SHAP Summary Plot - {title_suffix}")
        #                         fig_summary, ax = plt.subplots(figsize=(10, 6))
        #                         shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        #                         st.pyplot(fig_summary)

        #                         # SHAP Bar Plot
        #                         st.write(f"SHAP Bar Plot - {title_suffix}")
        #                         fig_bar, ax = plt.subplots(figsize=(10, 6))
        #                         shap.bar_plot(shap_values, feature_names=feature_names, show=False)
        #                         st.pyplot(fig_bar)
                            
        #                     with st.expander("View SHAP Plots for Test Data"):
        #                         plot_shap_summary_and_bar(shap_values_test, X_test_selected,selected_features, "Test Data")
                                                    
        #                 else:
        #                     st.write("no featur selcted on base of threshold")
                        
                        
                        
                    except:
                        st.write("no")
                        
                    
              
                
                
                
                
                
                # st.write(df.head(5))
                all_columns = df.columns.to_list()
                # st.write(all_columns)
                # all_outlier={
                #     'lof':LocalOutlierFactor(),
                #     'isoaltion forest':IsolationForest(),
                    
                # }
                
                
                
                    
                
                
                # numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                # selected_features = st.sidebar.multiselect("Select Features (Numerical Columns Only)", options=numerical_columns,
                # default=numerical_columns[:5] if len(numerical_columns) > 5 else numerical_columns)
                try:
                    st.sidebar.write("---------------------------------------------------------------------------")
                    target= st.sidebar.selectbox("Select Target Column", all_columns)
                
                    default_columns = [col for col in all_columns if col != target]

                    # Let the user select or modify the pre-selected columns
                    st.sidebar.write("---------------------------------------------------------------------------")
                    feature_columns = st.sidebar.multiselect(
                        "Select Feature Columns", 
                        all_columns,  # All columns as the options
                        default=default_columns  # Default selection (all except target)
                    )
                    
                    rate=st.sidebar.multiselect("select the learning rate",lrate)
                    history=[]
                except:
                    st.write()
                
                if len(feature_columns) == 0:
                        st.error("Please select at least one feature column.")
                else:
                        
                        label_encoder=LabelEncoder()
                        for col in feature_columns:
                            df[col] = label_encoder.fit_transform(df[col])
                        X = df[feature_columns]
                        df[target] = label_encoder.fit_transform(df[target])
                        y = df[target]
                        # X = pd.get_dummies(X, drop_first=True)

                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                all_models = {
                            # 'Stacking Classifer':'StackingClassi',
                            'Logistic Regression': LogisticRegression(),
                            'Linear Regression': LinearRegression(),
                            'decison tree classifier':DecisionTreeClassifier(),
                            'gradient boosting classifer':GradientBoostingClassifier(),
                            'Adabbost classifer':AdaBoostClassifier(),
                            'SVC':SVC(),
                            'random forest classifer':RandomForestClassifier(),
                            'Kneighbor Classifier':KNeighborsClassifier(),
                            
                            'k neighbour regressor':KNeighborsRegressor(),
                            'decison tree regressor':DecisionTreeRegressor(),
                            'adaboost regressor':AdaBoostRegressor(),
                            'gradient boost regressor':GradientBoostingRegressor(),
                            'SVR':SVR(),
                            'Random forest regressor':RandomForestRegressor()
                            
                            
                        } 
                st.sidebar.write("---------------------------------------------------------------------------")
                if st.sidebar.checkbox("Enable Anomaly Detection"): 
                    method = st.selectbox("Choose an anomaly detection method", 
                                  ["Isolation Forest"]
                                       )
                    if method == "Isolation Forest":
                        try:
                            n_estimators = st.number_input("Number of estimators", value=100)
                            contamination = st.number_input("Contamination (proportion of outliers)", value=0.1)
                            model = IsolationForest(n_estimators=n_estimators, contamination=contamination)
                            st.write("Running Isolation Forest...")
                            
                            predictions = model.fit_predict(X)
                            df['Anomaly'] = predictions  # -1 for anomalies, 1 for normal data
                            
                            st.title("Isolation Forest Anomaly Detection on Diabetes Dataset")

                            # Choose two features to plot using Streamlit's selectbox
                            feature1 = st.selectbox("Select first feature to plot", df.columns[:-2])
                            feature2 = st.selectbox("Select second feature to plot", df.columns[:-2])

                            # Scatter plot with anomaly and normal data
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = {1: 'blue', -1: 'red'}  # Blue for normal, red for anomalies

                            sns.scatterplot(x=feature1, y=feature2, data=df, hue='Anomaly', palette=colors, ax=ax)

                            # Add labels and title
                            plt.title("Isolation Forest Anomaly Detection")
                            plt.xlabel(feature1)
                            plt.ylabel(feature2)

                            # Show plot in Streamlit
                            st.pyplot(fig)
                        except:
                            st.write("")
                st.sidebar.write("---------------------------------------------------------------------------")            
                        
                if st.sidebar.checkbox("timeseries data"):
                    index_column = st.selectbox("Select a column to set as index (usually a date/time column)", df.columns)

                    # Set the selected column as the index
                    df.set_index(index_column, inplace=True)
                    df.index = pd.to_datetime(df.index)

                    # Display the modified DataFrame
                    st.write("Data with Selected Index:")
                    st.write(df.head())

                    # Step 4: Select the numerical column for ARIMA
                    value_column = st.selectbox("Select a numerical column for analysis", df.select_dtypes(include=[np.number]).columns)

                    # Step 5: Visualize the time series data
                    plt.figure(figsize=(12, 6))
                    plt.plot(df.index, df[value_column], label='Historical Data', color='blue')
                    plt.title('Time Series Data')
                    plt.xlabel('Date/Time')
                    plt.ylabel(value_column)
                    plt.legend()
                    st.pyplot(plt)

                    # Step 6: User inputs for ARIMA parameters
                    p = st.number_input("Select ARIMA parameter p (lag order)", min_value=0, max_value=5, value=1)
                    d = st.number_input("Select ARIMA parameter d (degree of differencing)", min_value=0, max_value=2, value=1)
                    q = st.number_input("Select ARIMA parameter q (order of moving average)", min_value=0, max_value=5, value=1)

                    # Step 7: Fit the ARIMA model
                    if st.button("Fit ARIMA Model"):
                        # Fit the ARIMA model
                        model = ARIMA(df[value_column], order=(p, d, q))
                        model_fit = model.fit()

                        # Step 8: Forecast the next 10 steps
                        forecast = model_fit.forecast(steps=10)

                        # Step 9: Create a DataFrame for the forecast
                        forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
                        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

                        # Step 10: Plot historical data and forecast
                        plt.figure(figsize=(12, 6))
                        plt.plot(df.index, df[value_column], label='Historical Data', color='blue')
                        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='orange')
                        plt.title('ARIMA Forecast')
                        plt.xlabel('Date/Time')
                        plt.ylabel(value_column)
                        plt.legend()
                        st.pyplot(plt)

                        # Step 11: Show model summary
                        st.write("ARIMA Model Summary:")
                        st.text(model_fit.summary())
                
                st.sidebar.write("---------------------------------------------------------------------------")
                
                
                if st.sidebar.checkbox("ensemble"):
                    models=['custom bagging','custom boosting']
                    option=["regression","classification"]
                    st.subheader("Boothstrap Aggreation")
                    m=st.selectbox("target type",options=["target"]+option)
                    if (m=="classification"):
                        
                        
                        modeling = [
                            DecisionTreeClassifier(),
                            SVC(probability=True),
                            LogisticRegression(),
                            RandomForestClassifier(),
                            KNeighborsClassifier()
                        ]
                      
                        selct_models = st.multiselect("Select models for ", options=modeling, format_func=lambda x: type(x).__name__)
                        st.write(selct_models)
                        def train_bagging_models(selct_models, X_train, y_train,X_test):
                            trained_models = []
                            predictions = []
                            
                            for model in selct_models:
                                # Bootstrap sampling (sampling with replacement)
                                X_resampled, y_resampled = resample(X_train, y_train, random_state=42)
        
                                
                                # Fit the model on the bootstrapped data
                                model.fit(X_resampled,y_resampled)
                                
                                # Add the trained model to the list
                                predictions.append(model.predict(X_test))
                                trained_models.append(model)
                            
                        
                        
                        
                                
                            predictions = np.array(predictions)
                            majority_vote_predictions = []
                            for i in range(predictions.shape[1]):
                                m = mode(predictions[:, i])
        # Check if mode is scalar and handle accordingly
                                majority_vote_predictions.append(m.mode[0] if isinstance(m.mode, np.ndarray) else m.mode)
                                    
                           
                            
                            return majority_vote_predictions
                        if selct_models:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # Train the models
                            predictions = train_bagging_models(selct_models, X_train, y_train,X_test)

                            

                            # Evaluate the accuracy
                            accuracy = accuracy_score(y_test, predictions)
                            
                            st.write(f"Accuracy: {accuracy:.4f}")
                            # probabilities = model.predict_proba(X_test)[:,1] 
                            acc = accuracy_score(y_test, predictions)
                            conf_matrix = confusion_matrix(y_test, predictions)
                           
                            report = classification_report(y_test, predictions, output_dict=True)
                            
                            precision = precision_score(y_test, predictions)
                           
                            recall = recall_score(y_test, predictions)
                           
                            f1 = f1_score(y_test, predictions)
                           
                                
                        
                          
                            info = ""
                            info += "Bootstrap Sampling\n"
                            info += f"Selected Models: {selct_models}\n"
                            info += f"Accuracy: {accuracy:.4f}\n"
                            info += f"Confusion Matrix:\n{conf_matrix}\n"
                            info += f"Classification Report:\n{report}\n"
                            info += f"Precision: {precision:.4f}\n"
                            info += f"Recall: {recall:.4f}\n"
                            info += f"F1 Score: {f1:.4f}\n"
                            
                            
                            # try:
                                
                            #     st.subheader("ROC Curve")
                            #     fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                            #     fig_roc, ax_roc = plt.subplots()
                            #     ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                            #     ax_roc.plot([0, 1], [0, 1], 'k--')
                            #     ax_roc.set_xlabel('False Positive Rate')
                            #     ax_roc.set_ylabel('True Positive Rate')
                            #     ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                            #     ax_roc.legend(loc='lower right')
                            #     st.pyplot(fig_roc)
                            # except:
                            #     st.write()

                            # try:
                                
                            #     st.subheader("Precision-Recall Curve")
                            #     precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                            #     fig_pr, ax_pr = plt.subplots()
                            #     ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                            #     ax_pr.set_xlabel('Recall')
                            #     ax_pr.set_ylabel('Precision')
                            #     ax_pr.set_title('Precision-Recall Curve')
                            #     ax_pr.legend(loc='upper right')
                            #     st.pyplot(fig_pr)
                            # except:
                            #     st.write()
                            # roc_auc = roc_auc_score(y_test, probabilities)
                            # mcc = matthews_corrcoef(y_test, predictions)
                            if st.button("Generat PDF"):
                               
                              # Ensure there's content to write to the PDF
                                pdf_buffer = create_pdf(info)
                                
                                # Create a download button for the generated PDF
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_buffer,
                                    file_name="report.pdf",
                                    mime="application/pdf"
                                )
                            
                            
    # Streamlit App
                            st.title("Classification Model Evaluation")

    # Display Accuracy
                            try:
                                
                                st.metric("Accuracy", f"{acc:.2f}")

                                # Display Precision, Recall, F1-Score
                                st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write()

    # Display ROC-AUC and MCC
                            # try:
                            #     # st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                            #     # st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                            #     # st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                            #     # st.markdown("---")
                            # except:
                            #     st.write("")

    # Confusion Matrix Heatmap
                            try:
                                st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write("")

    # Classification Report as a DataFrame
                            try:
                                
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write("")
                            
                        # if st.sidebar.button('Save Model',key="succes"):
                    
                        
                        #     model_file = io.BytesIO()
                        #     pickle.dump(model, model_file)
                        #     model_file.seek(0)

                        #     st.success('Model saved successfully as model.pkl!')

                                
                                
                        #     st.download_button(
                        #             label="Download Model",
                        #             data=model_file,
                        #             file_name="model.pkl",
                        #             mime="application/octet-stream"
                        #             )   
                    

    # ROC Curve
                            # try:
                            #     st.subheader("ROC Curve")
                            #     fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                            #     fig_roc, ax_roc = plt.subplots()
                            #     ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                            #     ax_roc.plot([0, 1], [0, 1], 'k--')
                            #     ax_roc.set_xlabel('False Positive Rate')
                            #     ax_roc.set_ylabel('True Positive Rate')
                            #     ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                            #     ax_roc.legend(loc='lower right')
                            #     st.pyplot(fig_roc)
                            # except:
                            #     st.write("")

    # # Precision-Recall Curve
    #                         try:
    #                             st.subheader("Precision-Recall Curve")
    #                             precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
    #                             fig_pr, ax_pr = plt.subplots()
    #                             ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
    #                             ax_pr.set_xlabel('Recall')
    #                             ax_pr.set_ylabel('Precision')
    #                             ax_pr.set_title('Precision-Recall Curve')
    #                             ax_pr.legend(loc='upper right')
    #                             st.pyplot(fig_pr)
    #                         except:
    #                             st.write("")
                            
    #                         try:
    #                             learning_rates=np.logspace(-3,0,10)
    #                             accuracy_history=[]
    #                             for lr in learning_rate:
    #                                 model = DecisionTreeClassifier(learning_rate=lr, n_estimators=100, random_state=42)
    #                                 model.fit(X_train,y_train)
    #                                 y_pred = model.predict(X_test)
    
    # # Calculate accuracy and store the result
    #                                 accuracy = accuracy_score(y_test, y_pred)
    #                                 accuracy_history.append(accuracy)
    #                                 results_df = pd.DataFrame({
    #                                     'Learning Rate': learning_rates,
    #                                     'Accuracy': accuracy_history
    #                                 })
    #                             plt.figure(figsize=(10, 6))
    #                             sns.lineplot(data=results_df, x='Learning Rate', y='Accuracy', marker='o', color='skyblue')
    #                             plt.title('Learning Rate vs. Accuracy')
    #                             plt.xlabel('Learning Rate')
    #                             plt.ylabel('Accuracy')
    #                             plt.xscale('log')  # Use log scale for learning rates
    #                             plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
    #                             plt.grid()
    #                             plt.tight_layout()

    #                             # Display the plot in Streamlit
    #                             st.pyplot(plt)

    #                             # Optionally, display the DataFrame with results
    #                             st.write(results_df)
    #                         except:
    #                             st.write("not")
                           
                                    
                        else:
                            st.write("Please select at least one model.")
                            
                    
                    
                    
                        
                    if m == "regression":
                        modeling = [
                            DecisionTreeRegressor(),
                            RandomForestRegressor(),
                            LinearRegression(),
                            SVR(),
                            KNeighborsRegressor()
                        ]
                        
                        selct_models = st.multiselect("Select models for bagging", options=modeling, format_func=lambda x: type(x).__name__)
                        st.write(selct_models)
                        
                        def train_bagging_models(selct_models, X_train, y_train, X_test):
                            trained_models = []
                            predictions = []
                            
                            for model in selct_models:
                                # Bootstrap sampling (sampling with replacement)
                                X_resampled, y_resampled = resample(X_train, y_train, random_state=42)
                                
                                # Fit the model on the bootstrapped data
                                model.fit(X_resampled, y_resampled)
                                
                                # Predict on the test data
                                predictions.append(model.predict(X_test))
                                trained_models.append(model)
                            
                            # Convert predictions to a numpy array (shape: n_models x n_samples)
                            predictions = np.array(predictions)
                            
                            # Aggregate predictions by averaging (regression task)
                            average_predictions = np.mean(predictions, axis=0)
                            
                            return average_predictions
                        
                        if selct_models:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
                            
                            # Train the models
                            predictions = train_bagging_models(selct_models, X_train, y_train, X_test)
                            mse = mean_squared_error(y_test, predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, predictions)
                            mape = mean_absolute_percentage_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            evs = explained_variance_score(y_test, predictions)
                            n = len(y_test)
                            k = X_test.shape[1]
                            adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                            st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                            metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                            st.markdown("---")

    # Visualization: Predicted vs Actual Values
                            st.header("📊 Predicted vs. Actual Values")

                            fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                            ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                            ax_pv.set_xlabel("Actual Values")
                            ax_pv.set_ylabel("Predicted Values")
                            ax_pv.set_title("Predicted vs. Actual Values")
                            st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                            st.header("📉 Residuals Plot")

                            residuals = y_test - predictions
                            fig_res, ax_res = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                            ax_res.axhline(0, color='r', linestyle='--')
                            ax_res.set_xlabel("Predicted Values")
                            ax_res.set_ylabel("Residuals")
                            ax_res.set_title("Residuals vs. Predicted Values")
                            st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                            st.header("📈 Residuals Distribution")

                            fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                            sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                            ax_res_dist.set_xlabel("Residuals")
                            ax_res_dist.set_ylabel("Frequency")
                            ax_res_dist.set_title("Distribution of Residuals")
                            st.pyplot(fig_res_dist)
                            
                            # if st.sidebar.button('Save Model',key="succesful"):
                        
                            
                            #     model_file = io.BytesIO()
                            #     pickle.dump(model, model_file)
                            #     model_file.seek(0)

                            #     st.success('Model saved successfully as model.pkl!')

                                    
                                    
                            #     st.download_button(
                            #             label="Download Model",
                            #             data=model_file,
                            #             file_name="model.hd5",
                            #             mime="application/octet-stream"
                            #             )   
                        
                            
                            # Evaluate using regression metrics (R-squared or MSE)
                            
                        else:
                            st.write("Please select at least one model.")

                        
                                    
                st.sidebar.write("---------------------------------------------------------------------------")
                if st.sidebar.checkbox('Stacking '):
                    option=["regression","classification"]
                    m=st.selectbox("target type",options=["target"]+option)
                    
                    if(m=="classification"):
                        all_model = {
                            'Logistic Regression': LogisticRegression(),
                            # 'Linear Regression': LinearRegression(),
                            'decison tree classifier':DecisionTreeClassifier(),
                            'gradient boosting classifer':GradientBoostingClassifier(),
                            'Adabbost classifer':AdaBoostClassifier(),
                            'SVC':SVC(),
                            'random forest classifer':RandomForestClassifier(),
                            'Kneighbor Classifier':KNeighborsClassifier(),
                        
                            
                            
                        }
                        selected_base_models = st.multiselect("Select base models to be stacked", list(all_model.keys()))

                        # User selection for final model to use after stacking
                        final_model = st.selectbox("Select the final model to be used", list(all_model.keys()))
                        try:
                            base_models = [(model_name, all_model[model_name]) for model_name in selected_base_models]

                            st.write("Selected base models for stacking:")
                            for model in base_models:
                                st.write(model[0])

                            # Select final model
                            final_model_instance = all_model[final_model]

                            st.write(f"Final model: {final_model}")
                            model=StackingClassifier(estimators=base_models,final_estimator=final_model_instance)
                        
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                            
                            if st.button("run"):
                    
                                probabilities = model.predict_proba(X_test)[:,1] 
                                acc = accuracy_score(y_test, predictions)
                                conf_matrix = confusion_matrix(y_test, predictions)
                                report = classification_report(y_test, predictions, output_dict=True)
                                precision = precision_score(y_test, predictions)
                                recall = recall_score(y_test, predictions)
                                f1 = f1_score(y_test, predictions)
                                roc_auc = roc_auc_score(y_test, probabilities)
                                mcc = matthews_corrcoef(y_test, predictions)
                                
        # Streamlit App
                                st.title("Classification Model Evaluation")

        # Display Accuracy
                                try:
                                    
                                    st.metric("Accuracy", f"{acc:.2f}")

                                    # Display Precision, Recall, F1-Score
                                    st.subheader("Precision, Recall, and F1-Score")
                                    precision_val = report['weighted avg']['precision']
                                    recall_val = report['weighted avg']['recall']
                                    f1_val = report['weighted avg']['f1-score']
                                    st.write(f"**Precision:** {precision_val:.2f}")
                                    st.write(f"**Recall:** {recall_val:.2f}")
                                    st.write(f"**F1-Score:** {f1_val:.2f}")
                                except:
                                    st.write("")

        # Display ROC-AUC and MCC
                                try:
                                    st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                    st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                                    st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                                    st.markdown("---")
                                except:
                                    st.write("")
        # Confusion Matrix Heatmap
                                try:
                                    st.subheader("Confusion Matrix")
                                    fig_cm, ax_cm = plt.subplots()
                                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                    ax_cm.set_xlabel('Predicted')
                                    ax_cm.set_ylabel('Actual')
                                    ax_cm.set_title('Confusion Matrix')
                                    st.pyplot(fig_cm)
                                except:
                                    st.write("")

        # Classification Report as a DataFrame
                                try:
                                    st.subheader("Classification Report")
                                    report_df = pd.DataFrame(report).transpose()
                                    st.dataframe(report_df.style.highlight_max(axis=0))
                                except:
                                    st.write("")
        # ROC Curve  
                                try:
        
                                    st.subheader("ROC Curve")
                                    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                                    fig_roc, ax_roc = plt.subplots()
                                    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                    ax_roc.plot([0, 1], [0, 1], 'k--')
                                    ax_roc.set_xlabel('False Positive Rate')
                                    ax_roc.set_ylabel('True Positive Rate')
                                    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                    ax_roc.legend(loc='lower right')
                                    st.pyplot(fig_roc)
                                except:
                                    st.write("")
        # Precision-Recall Curve
                                try:
                                    st.subheader("Precision-Recall Curve")
                                    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                                    fig_pr, ax_pr = plt.subplots()
                                    ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                    ax_pr.set_xlabel('Recall')
                                    ax_pr.set_ylabel('Precision')
                                    ax_pr.set_title('Precision-Recall Curve')
                                    ax_pr.legend(loc='upper right')
                                    st.pyplot(fig_pr)
                                except:
                                    st.write("")
                            if st.button('Save Model',key='save_modelbutton'):
            
                
                                model_file = io.BytesIO()
                                pickle.dump(model, model_file)
                                
                                model_file.seek(0)

                                st.success('Model saved successfully  model.pkl!')

                                    
                                    
                                st.download_button(
                                        label="Download Model",
                                        data=model_file,
                                        
                                        file_name="model.pkl",
                                        mime="application/octet-stream"
                                        )   
                        
                                
                            
                            

                        except Exception as e:
                            st.write(f"An error occurred: {e}")

                    if(m=="regression"):
                        all_model = {
                          
                            'Linear Regression': LinearRegression(),
                            
                            
                            'k neighbour regressor':KNeighborsRegressor(),
                            'decison tree regressor':DecisionTreeRegressor(),
                            'adaboost regressor':AdaBoostRegressor(),
                            'gradient boost regressor':GradientBoostingRegressor(),
                            'SVR':SVR(),
                            'Random forest regressor':RandomForestRegressor()
                            
                        
                            
                            
                        }
                        selected_base_models = st.multiselect("Select base models to be stacked", list(all_model.keys()))

                        # User selection for final model to use after stacking
                        final_model = st.selectbox("Select the final model to be used", list(all_model.keys()))
                        try:
                            base_models = [(model_name, all_model[model_name]) for model_name in selected_base_models]

                            st.write("Selected base models for stacking:")
                            for model in base_models:
                                st.write(model[0])

                            # Select final model
                            final_model_instance = all_model[final_model]

                            st.write(f"Final model: {final_model}")
                            model=StackingRegressor(estimators=base_models,final_estimator=final_model_instance)
                        
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                            mse = mean_squared_error(y_test, predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, predictions)
                            mape = mean_absolute_percentage_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            evs = explained_variance_score(y_test, predictions)
                            n = len(y_test)
                            k = X_test.shape[1]
                            adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                            st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                            metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                            st.markdown("---")

    # Visualization: Predicted vs Actual Values
                            st.header("📊 Predicted vs. Actual Values")

                            fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                            ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                            ax_pv.set_xlabel("Actual Values")
                            ax_pv.set_ylabel("Predicted Values")
                            ax_pv.set_title("Predicted vs. Actual Values")
                            st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                            st.header("📉 Residuals Plot")

                            residuals = y_test - predictions
                            fig_res, ax_res = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                            ax_res.axhline(0, color='r', linestyle='--')
                            ax_res.set_xlabel("Predicted Values")
                            ax_res.set_ylabel("Residuals")
                            ax_res.set_title("Residuals vs. Predicted Values")
                            st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                            st.header("📈 Residuals Distribution")

                            fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                            sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                            ax_res_dist.set_xlabel("Residuals")
                            ax_res_dist.set_ylabel("Frequency")
                            ax_res_dist.set_title("Distribution of Residuals")
                            st.pyplot(fig_res_dist)
                        except:
                            st.write("")
                        
                       
                    
                    
                        
                            
                
                st.sidebar.write("---------------------------------------------------------------------------")        
                model_name = st.sidebar.selectbox("Select Model",options=["Select a model"] + list(all_models.keys())) 
                        
                        
                    
                
                 # Adding a placeholder as the first option)
                
                if model_name=="Stacking Classifer":
                    all_model = {
                            'Logistic Regression': LogisticRegression(),
                            # 'Linear Regression': LinearRegression(),
                            'decison tree classifier':DecisionTreeClassifier(),
                            'gradient boosting classifer':GradientBoostingClassifier(),
                            'Adabbost classifer':AdaBoostClassifier(),
                            'SVC':SVC(),
                            'random forest classifer':RandomForestClassifier(),
                            'Kneighbor Classifier':KNeighborsClassifier(),
                        
                            
                            
                        }
                   
                    first_model=st.selectbox("selct he first model to be stack",list(all_model.keys()))
                    second_model=st.selectbox("selct he second model to be stack",list(all_model.keys()))
                    final_model=st.selectbox("selct he final model to be stack",list(all_model.keys()))
                    try:

                        base_models = [
                    ('first_model', all_model[first_model]),
                        ('second_model', all_model[second_model]),
                        
                        ]
                    
                    
                    
                    
                        meta_model=all_model[final_model]
                    except:
                        st.write("")
                    model=StackingClassifier(estimators=base_models,final_estimator=meta_model)
                    try:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    except:
                        st.write("")
                    if st.button("run model"):
                    
                            probabilities = model.predict_proba(X_test)[:,1] 
                            acc = accuracy_score(y_test, predictions)
                            conf_matrix = confusion_matrix(y_test, predictions)
                            report = classification_report(y_test, predictions, output_dict=True)
                            precision = precision_score(y_test, predictions)
                            recall = recall_score(y_test, predictions)
                            f1 = f1_score(y_test, predictions)
                            roc_auc = roc_auc_score(y_test, probabilities)
                            mcc = matthews_corrcoef(y_test, predictions)
                            
    # Streamlit App
                            st.title("Classification Model Evaluation")

    # Display Accuracy
                            try:
                                
                                st.metric("Accuracy", f"{acc:.2f}")

                                # Display Precision, Recall, F1-Score
                                st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write("")

    # Display ROC-AUC and MCC
                            try:
                                st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                                st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                                st.markdown("---")
                            except:
                                st.write("")
    # Confusion Matrix Heatmap
                            try:
                                st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write("")

    # Classification Report as a DataFrame
                            try:
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write("")
    # ROC Curve  
                            try:
    
                                st.subheader("ROC Curve")
                                fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                ax_roc.plot([0, 1], [0, 1], 'k--')
                                ax_roc.set_xlabel('False Positive Rate')
                                ax_roc.set_ylabel('True Positive Rate')
                                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax_roc.legend(loc='lower right')
                                st.pyplot(fig_roc)
                            except:
                                st.write("")
    # Precision-Recall Curve
                            try:
                                st.subheader("Precision-Recall Curve")
                                precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                                fig_pr, ax_pr = plt.subplots()
                                ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                ax_pr.set_xlabel('Recall')
                                ax_pr.set_ylabel('Precision')
                                ax_pr.set_title('Precision-Recall Curve')
                                ax_pr.legend(loc='upper right')
                                st.pyplot(fig_pr)
                            except:
                                st.write("")
                
                
            
                try:
                    
                    
                    model = all_models[model_name]
                    st.write(model)
                    
                except:
                    st.write(" ")
               
                

                if st.sidebar.button("Run model"):
                    
                        
                        
                    
                

                    
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        

                        st.write("### Model Trained Successfully!")

                        if model_name == 'Logistic Regression':
                            try:
                            
                                probabilities = model.predict_proba(X_test)[:,1] 
                            
                                acc = accuracy_score(y_test, predictions)
                                history.append(acc)
                                conf_matrix = confusion_matrix(y_test, predictions)
                                report = classification_report(y_test, predictions, output_dict=True)
                                
                                
                                precision = precision_score(y_test, predictions)  # or pos_label="no" if "no" is the positive class
                                
                                
                                recall = recall_score(y_test, predictions)
                                f1 = f1_score(y_test, predictions)
                                roc_auc = roc_auc_score(y_test, probabilities)
                                mcc = matthews_corrcoef(y_test, predictions)
                                
                                info += f"\nLogistic Regression Metrics:\n"
                                info += f"Accuracy: {acc:.4f}\n"
                                info += f"Precision: {precision:.4f}\n"
                                info += f"Recall: {recall:.4f}\n"
                                info += f"F1-Score: {f1:.4f}\n"
                                info += f"ROC AUC: {roc_auc:.4f}\n"
                                info += f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n"

                                # Convert confusion matrix to a string format and append
                                conf_matrix_str = np.array2string(conf_matrix)
                                info += f"Confusion Matrix:\n{conf_matrix_str}\n"

                                # Append the classification report (already formatted as a string)
                                info += f"Classification Report:\n{report}\n"
                                
                                
                                
                            except:
                                st.write()
                            
    # Streamlit App

    # Display Accuracy
                            try:

                            # Display Precision, Recall, F1-Score
                                st.metric("Accuracy", f"{acc:.2f}")
                                # st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write()
                            

    # Display ROC-AUC and MCC
                            try:
                                
                                # st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                                st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                                st.markdown("---")
                            except:
                                st.write("")

    # Confusion Matrix Heatmap
                            try:
                                
                                # st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write("")

    # Classification Report as a DataFrame
                            try:
                                # st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write("")

    # ROC Curve
                            try:
                                
                                # st.subheader("ROC Curve")
                                fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                ax_roc.plot([0, 1], [0, 1], 'k--')
                                ax_roc.set_xlabel('False Positive Rate')
                                ax_roc.set_ylabel('True Positive Rate')
                                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax_roc.legend(loc='lower right')
                                st.pyplot(fig_roc)
                            except:
                                st.write("")

    # Precision-Recall Curve
                            try:
                                # st.subheader("Precision-Recall Curve")
                                precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                                fig_pr, ax_pr = plt.subplots()
                                ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                ax_pr.set_xlabel('Recall')
                                ax_pr.set_ylabel('Precision')
                                ax_pr.set_title('Precision-Recall Curve')
                                ax_pr.legend(loc='upper right')
                                st.pyplot(fig_pr)
                            except:
                                st.write("")
                            try:
                                accuracy_history=[]
                                lrate=[0.01, 0.05, 0.1, 0.2]
                                for lr in lrate:
                                    model = SGDClassifier(loss='log', learning_rate='constant', eta0=lr, max_iter=1000, random_state=42)
                                    model.fit(X_train, y_train)
                                    
                                    # Predict on the test set
                                    y_pred = model.predict(X_test)
                                    
                                    # Calculate accuracy and store the result
                                    accuracy = accuracy_score(y_test, y_pred)
                                    accuracy_history.append(accuracy)
                                result_df=pd.DataFrame({
                                        'Learning Rate':lrate,
                                        'Accuracy':accuracy_history
                                    })
                                plt.figure(figsize=(10,6))
                                sns.lineplot(data=result_df,x='Learning Rate',y='Accuracy',marker='o',color='skyblue')
                                plt.xlabel('Learning Rate')
                                plt.ylabel('Accuracy')
                                plt.grid()
                                plt.tight_layout()
                                st.pyplot()
                                st.write(result_df)
                                
                               
                            except:
                                st.write("")
                            
                            
                        elif model_name == 'decison tree classifier':
                            try:
                                
                                probabilities = model.predict_proba(X_test)[:,1] 
                                acc = accuracy_score(y_test, predictions)
                                conf_matrix = confusion_matrix(y_test, predictions)
                                report = classification_report(y_test, predictions, output_dict=True)
                                precision = precision_score(y_test, predictions)
                                recall = recall_score(y_test, predictions)
                                f1 = f1_score(y_test, predictions)
                                roc_auc = roc_auc_score(y_test, probabilities)
                                mcc = matthews_corrcoef(y_test, predictions)
                                info += f"\nLogistic Regression Metrics:\n"
                                info += f"Accuracy: {acc:.4f}\n"
                                info += f"Precision: {precision:.4f}\n"
                                info += f"Recall: {recall:.4f}\n"
                                info += f"F1-Score: {f1:.4f}\n"
                                info += f"ROC AUC: {roc_auc:.4f}\n"
                                info += f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n"

                                # Convert confusion matrix to a string format and append
                                conf_matrix_str = np.array2string(conf_matrix)
                                info += f"Confusion Matrix:\n{conf_matrix_str}\n"

                                # Append the classification report (already formatted as a string)
                                info += f"Classification Report:\n{report}\n"
                                
                                
                                
                            except:
                                st.write()
                            
    # Streamlit App
                            st.title("Classification Model Evaluation")

    # Display Accuracy
                            try:
                                
                                st.metric("Accuracy", f"{acc:.2f}")

                                # Display Precision, Recall, F1-Score
                                st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write()

    # Display ROC-AUC and MCC
                            try:
                                st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                                st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                                st.markdown("---")
                            except:
                                st.write("")

    # Confusion Matrix Heatmap
                            try:
                                st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write("")

    # Classification Report as a DataFrame
                            try:
                                
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write("")

    # ROC Curve
                            try:
                                st.subheader("ROC Curve")
                                fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                ax_roc.plot([0, 1], [0, 1], 'k--')
                                ax_roc.set_xlabel('False Positive Rate')
                                ax_roc.set_ylabel('True Positive Rate')
                                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax_roc.legend(loc='lower right')
                                st.pyplot(fig_roc)
                            except:
                                st.write("")

    # Precision-Recall Curve
                            try:
                                st.subheader("Precision-Recall Curve")
                                precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                                fig_pr, ax_pr = plt.subplots()
                                ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                ax_pr.set_xlabel('Recall')
                                ax_pr.set_ylabel('Precision')
                                ax_pr.set_title('Precision-Recall Curve')
                                ax_pr.legend(loc='upper right')
                                st.pyplot(fig_pr)
                            except:
                                st.write("")
                            
                            try:
                                learning_rates=np.logspace(-3,0,10)
                                accuracy_history=[]
                                for lr in learning_rate:
                                    model = DecisionTreeClassifier(learning_rate=lr, n_estimators=100, random_state=42)
                                    model.fit(X_train,y_train)
                                    y_pred = model.predict(X_test)
    
    # Calculate accuracy and store the result
                                    accuracy = accuracy_score(y_test, y_pred)
                                    accuracy_history.append(accuracy)
                                    results_df = pd.DataFrame({
                                        'Learning Rate': learning_rates,
                                        'Accuracy': accuracy_history
                                    })
                                plt.figure(figsize=(10, 6))
                                sns.lineplot(data=results_df, x='Learning Rate', y='Accuracy', marker='o', color='skyblue')
                                plt.title('Learning Rate vs. Accuracy')
                                plt.xlabel('Learning Rate')
                                plt.ylabel('Accuracy')
                                plt.xscale('log')  # Use log scale for learning rates
                                plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
                                plt.grid()
                                plt.tight_layout()

                                # Display the plot in Streamlit
                                st.pyplot(plt)

                                # Optionally, display the DataFrame with results
                                st.write(results_df)
                            except:
                                st.write("")
                           
                                    
                                    
                        
                            
                            
                        elif model_name == 'gradient boosting classifer':
                            try:
                                probabilities = model.predict_proba(X_test)[:,1] 
                                acc = accuracy_score(y_test, predictions)
                                conf_matrix = confusion_matrix(y_test, predictions)
                                report = classification_report(y_test, predictions, output_dict=True)
                                precision = precision_score(y_test, predictions)
                                recall = recall_score(y_test, predictions)
                                f1 = f1_score(y_test, predictions)
                                roc_auc = roc_auc_score(y_test, probabilities)
                                mcc = matthews_corrcoef(y_test, predictions)
                            except:
                                st.write()
                            
    # Streamlit App
                            st.title("Classification Model Evaluation")

    # Display Accuracy
                            try:
                                st.metric("Accuracy", f"{acc:.2f}")

                                # Display Precision, Recall, F1-Score
                                st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write()

    # Display ROC-AUC and MCC
                            try:
                                
                                st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                                st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                                st.markdown("---")
                            except:
                                st.write("")

    # Confusion Matrix Heatmap
                            try:
                                st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write("")

    # Classification Report as a DataFrame
                            try:
                                
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write("")

    # ROC Curve
                            try:
                                st.subheader("ROC Curve")
                                fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                ax_roc.plot([0, 1], [0, 1], 'k--')
                                ax_roc.set_xlabel('False Positive Rate')
                                ax_roc.set_ylabel('True Positive Rate')
                                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax_roc.legend(loc='lower right')
                                st.pyplot(fig_roc)
                            except:
                                st.write("")

    # Precision-Recall Curve
                            try:
                                st.subheader("Precision-Recall Curve")
                                precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                                fig_pr, ax_pr = plt.subplots()
                                ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                ax_pr.set_xlabel('Recall')
                                ax_pr.set_ylabel('Precision')
                                ax_pr.set_title('Precision-Recall Curve')
                                ax_pr.legend(loc='upper right')
                                st.pyplot(fig_pr)
                            except:
                                st.write("")
                            
                            try:
                                learning_rates=np.logspace(-3,0,10)
                                accuracy_history=[]
                                for lr in learning_rate:
                                    model = GradientBoostingClassifier(learning_rate=lr, n_estimators=100, random_state=42)
                                    model.fit(X_train,y_train)
                                    y_pred = model.predict(X_test)
    
    # Calculate accuracy and store the result
                                    accuracy = accuracy_score(y_test, y_pred)
                                    accuracy_history.append(accuracy)
                                    results_df = pd.DataFrame({
                                        'Learning Rate': learning_rates,
                                        'Accuracy': accuracy_history
                                    })
                                plt.figure(figsize=(10, 6))
                                sns.lineplot(data=results_df, x='Learning Rate', y='Accuracy', marker='o', color='skyblue')
                                plt.title('Learning Rate vs. Accuracy')
                                plt.xlabel('Learning Rate')
                                plt.ylabel('Accuracy')
                                plt.xscale('log')  # Use log scale for learning rates
                                plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
                                plt.grid()
                                plt.tight_layout()

                                # Display the plot in Streamlit
                                st.pyplot(plt)

                                # Optionally, display the DataFrame with results
                                st.write(results_df)
                            except:
                                st.write()
                                
                                
                            
                            
                            
                            
                            
                        elif model_name =='Adabbost classifer':
                            try:
                                
                                probabilities = model.predict_proba(X_test)[:,1] 
                                
                                acc = accuracy_score(y_test, predictions)
                                conf_matrix = confusion_matrix(y_test, predictions)
                                report = classification_report(y_test, predictions, output_dict=True)
                                precision = precision_score(y_test, predictions)
                                recall = recall_score(y_test, predictions)
                                f1 = f1_score(y_test, predictions)
                                roc_auc = roc_auc_score(y_test, probabilities)
                                mcc = matthews_corrcoef(y_test, predictions)
                            except:
                                st.write()
                            
    # Streamlit App
                            st.title("Classification Model Evaluation")

    # Display Accuracy
                            try:
                                st.metric("Accuracy", f"{acc:.2f}")

                                # Display Precision, Recall, F1-Score
                                st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write()

    # Display ROC-AUC and MCC
                            try:
                                st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                                st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                                st.markdown("---")
                            except:
                                st.write()

    # Confusion Matrix Heatmap
                            try:
                                
                                st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write()

    # Classification Report as a DataFrame
                            try:
                                
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write()

    # ROC Curve
                            try:
                                
                                st.subheader("ROC Curve")
                                fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                ax_roc.plot([0, 1], [0, 1], 'k--')
                                ax_roc.set_xlabel('False Positive Rate')
                                ax_roc.set_ylabel('True Positive Rate')
                                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax_roc.legend(loc='lower right')
                                st.pyplot(fig_roc)
                            except:
                                st.write()

    # Precision-Recall Curve
                            try:
                                st.subheader("Precision-Recall Curve")
                                precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                                fig_pr, ax_pr = plt.subplots()
                                ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                ax_pr.set_xlabel('Recall')
                                ax_pr.set_ylabel('Precision')
                                ax_pr.set_title('Precision-Recall Curve')
                                ax_pr.legend(loc='upper right')
                                st.pyplot(fig_pr)
                            except:
                                st.write()
                            
                            try:
                                learning_rates=np.logspace(-3,0,10)
                                accuracy_history=[]
                                for lr in learning_rate:
                                    model = AdaBoostClassifier(learning_rate=lr, n_estimators=100, random_state=42)
                                    model.fit(X_train,y_train)
                                    y_pred = model.predict(X_test)
    
    # Calculate accuracy and store the result
                                    accuracy = accuracy_score(y_test, y_pred)
                                    accuracy_history.append(accuracy)
                                    results_df = pd.DataFrame({
                                        'Learning Rate': learning_rates,
                                        'Accuracy': accuracy_history
                                    })
                                plt.figure(figsize=(10, 6))
                                sns.lineplot(data=results_df, x='Learning Rate', y='Accuracy', marker='o', color='skyblue')
                                plt.title('Learning Rate vs. Accuracy')
                                plt.xlabel('Learning Rate')
                                plt.ylabel('Accuracy')
                                plt.xscale('log')  # Use log scale for learning rates
                                plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
                                plt.grid()
                                plt.tight_layout()

                                # Display the plot in Streamlit
                                st.pyplot(plt)

                                # Optionally, display the DataFrame with results
                                st.write(results_df)
                            except:
                                st.write()
                                
                            
                            
                            
                        elif model_name == 'SVC':
                            # probabilities = model.predict_proba(X_test)[:,1] 
                            try:
                                
                                acc = accuracy_score(y_test, predictions)
                                conf_matrix = confusion_matrix(y_test, predictions)
                                report = classification_report(y_test, predictions, output_dict=True)
                                precision = precision_score(y_test, predictions)
                                recall = recall_score(y_test, predictions)
                                f1 = f1_score(y_test, predictions)
                                # roc_auc = roc_auc_score(y_test, probabilities)
                                mcc = matthews_corrcoef(y_test, predictions)
                            except:
                                st.write()
                            
    # Streamlit App
                            st.title("Classification Model Evaluation")

                            try:
                                st.metric("Accuracy", f"{acc:.2f}")

                                # Display Precision, Recall, F1-Score
                                st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write()

    # Display ROC-AUC and MCC
                            # st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                            # st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                            # st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                            st.markdown("---")

                            try:
                                st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write()

                            try:
                                
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write()

    # ROC Curve
                            # st.subheader("ROC Curve")
                            # fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                            # fig_roc, ax_roc = plt.subplots()
                            # ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                            # ax_roc.plot([0, 1], [0, 1], 'k--')
                            # ax_roc.set_xlabel('False Positive Rate')
                            # ax_roc.set_ylabel('True Positive Rate')
                            # ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                            # ax_roc.legend(loc='lower right')
                            # st.pyplot(fig_roc)

    # Precision-Recall Curve
                            # st.subheader("Precision-Recall Curve")
                            # precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                            # fig_pr, ax_pr = plt.subplots()
                            # ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                            # ax_pr.set_xlabel('Recall')
                            # ax_pr.set_ylabel('Precision')
                            # ax_pr.set_title('Precision-Recall Curve')
                            # ax_pr.legend(loc='upper right')
                            # st.pyplot(fig_pr)
                            
                        
                            
                        
                        elif model_name == 'random forest classifer':
                            try:
                                probabilities = model.predict_proba(X_test)[:,1] 
                                
                                acc = accuracy_score(y_test, predictions)
                                conf_matrix = confusion_matrix(y_test, predictions)
                                report = classification_report(y_test, predictions, output_dict=True)
                                precision = precision_score(y_test, predictions)
                                recall = recall_score(y_test, predictions)
                                f1 = f1_score(y_test, predictions)
                                roc_auc = roc_auc_score(y_test, probabilities)
                                mcc = matthews_corrcoef(y_test, predictions)
                            except:
                                st.write()
                            
    # Streamlit App
                            st.title("Classification Model Evaluation")

                            try:
                                
                                st.metric("Accuracy", f"{acc:.2f}")

                                # Display Precision, Recall, F1-Score
                                st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write()

    # Display ROC-AUC and MCC
                            try:
                                
                                st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                                st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                                st.markdown("---")
                            except:
                                st.write()

                            try:
                                
                                st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write()

                            try:
                                    
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write()

                            try:
                                
                                st.subheader("ROC Curve")
                                fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                ax_roc.plot([0, 1], [0, 1], 'k--')
                                ax_roc.set_xlabel('False Positive Rate')
                                ax_roc.set_ylabel('True Positive Rate')
                                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax_roc.legend(loc='lower right')
                                st.pyplot(fig_roc)
                            except:
                                st.write()

                            try:
                                
                                st.subheader("Precision-Recall Curve")
                                precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                                fig_pr, ax_pr = plt.subplots()
                                ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                ax_pr.set_xlabel('Recall')
                                ax_pr.set_ylabel('Precision')
                                ax_pr.set_title('Precision-Recall Curve')
                                ax_pr.legend(loc='upper right')
                                st.pyplot(fig_pr)
                            except:
                                st.write()
                            
                            
                            
                                
                            
                            
                        elif model_name == 'Kneighbor Classifier':
                            try:
                                
                                probabilities = model.predict_proba(X_test)[:,1] 
                                acc = accuracy_score(y_test, predictions)
                                conf_matrix = confusion_matrix(y_test, predictions)
                                report = classification_report(y_test, predictions, output_dict=True)
                                precision = precision_score(y_test, predictions)
                                recall = recall_score(y_test, predictions)
                                f1 = f1_score(y_test, predictions)
                                roc_auc = roc_auc_score(y_test, probabilities)
                                mcc = matthews_corrcoef(y_test, predictions)
                            except:
                                st.write()
                            
    # Streamlit App
                            st.title("Classification Model Evaluation")

                            try:
                                
                                st.metric("Accuracy", f"{acc:.2f}")

                                # Display Precision, Recall, F1-Score
                                st.subheader("Precision, Recall, and F1-Score")
                                precision_val = report['weighted avg']['precision']
                                recall_val = report['weighted avg']['recall']
                                f1_val = report['weighted avg']['f1-score']
                                st.write(f"**Precision:** {precision_val:.2f}")
                                st.write(f"**Recall:** {recall_val:.2f}")
                                st.write(f"**F1-Score:** {f1_val:.2f}")
                            except:
                                st.write()

                            try:
                                
                                st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                st.write(f"**ROC-AUC:** {roc_auc:.2f}")
                                st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc:.2f}")

                                st.markdown("---")
                            except:
                                st.write()

                            try:
                                
                                st.subheader("Confusion Matrix")
                                fig_cm, ax_cm = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                ax_cm.set_xlabel('Predicted')
                                ax_cm.set_ylabel('Actual')
                                ax_cm.set_title('Confusion Matrix')
                                st.pyplot(fig_cm)
                            except:
                                st.write()

                            try:
                                    
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.highlight_max(axis=0))
                            except:
                                st.write()

                            try:
                                
                                st.subheader("ROC Curve")
                                fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                                ax_roc.plot([0, 1], [0, 1], 'k--')
                                ax_roc.set_xlabel('False Positive Rate')
                                ax_roc.set_ylabel('True Positive Rate')
                                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax_roc.legend(loc='lower right')
                                st.pyplot(fig_roc)
                            except:
                                st.write()

                            try:
                                
                                st.subheader("Precision-Recall Curve")
                                precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities)
                                fig_pr, ax_pr = plt.subplots()
                                ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                ax_pr.set_xlabel('Recall')
                                ax_pr.set_ylabel('Precision')
                                ax_pr.set_title('Precision-Recall Curve')
                                ax_pr.legend(loc='upper right')
                                st.pyplot(fig_pr)
                            except:
                                st.write()
                            
                            
                        
                        
                        elif model_name == 'Linear Regression':
                            st.write(df.head())
                            try:
                                mse = mean_squared_error(y_test, predictions)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, predictions)
                                mape = mean_absolute_percentage_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                evs = explained_variance_score(y_test, predictions)
                                n = len(y_test)
                                k = X_test.shape[1]
                                adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                                
                                st.header("📝 Regression Metrics")

        # Create a dictionary of metrics
                                metrics = {
                                        "Mean Absolute Error (MAE)": mae,
                                        "Mean Squared Error (MSE)": mse,
                                        "Root Mean Squared Error (RMSE)": rmse,
                                        "Mean Absolute Percentage Error (MAPE)": mape,
                                        "R² Score": r2,
                                        "Adjusted R² Score": adjusted_r2_score,
                                        "Explained Variance Score": evs
                                        }
                                
                                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                                metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

        # Display metrics using Streamlit
                                st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                                st.markdown("---")

        # Visualization: Predicted vs Actual Values
                                st.header("📊 Predicted vs. Actual Values")

                                fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                                ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                                ax_pv.set_xlabel("Actual Values")
                                ax_pv.set_ylabel("Predicted Values")
                                ax_pv.set_title("Predicted vs. Actual Values")
                                st.pyplot(fig_pv)

        # Visualization: Residuals Plot
                                st.header("📉 Residuals Plot")

                                residuals = y_test - predictions
                                fig_res, ax_res = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_xlabel("Predicted Values")
                                ax_res.set_ylabel("Residuals")
                                ax_res.set_title("Residuals vs. Predicted Values")
                                st.pyplot(fig_res)

        # Visualization: Residuals Distribution
                                st.header("📈 Residuals Distribution")

                                fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                                sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                                ax_res_dist.set_xlabel("Residuals")
                                ax_res_dist.set_ylabel("Frequency")
                                ax_res_dist.set_title("Distribution of Residuals")
                                st.pyplot(fig_res_dist)
                                
                            except:
                                st.write()  
                                
                            
                        
                            
                            
                        
                        
                        elif model_name == 'decison tree regressor':
                            mse = mean_squared_error(y_test, predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, predictions)
                            mape = mean_absolute_percentage_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            evs = explained_variance_score(y_test, predictions)
                            n = len(y_test)
                            k = X_test.shape[1]
                            adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                            st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                            metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                            st.markdown("---")

    # Visualization: Predicted vs Actual Values
                            st.header("📊 Predicted vs. Actual Values")

                            fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                            ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                            ax_pv.set_xlabel("Actual Values")
                            ax_pv.set_ylabel("Predicted Values")
                            ax_pv.set_title("Predicted vs. Actual Values")
                            st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                            st.header("📉 Residuals Plot")

                            residuals = y_test - predictions
                            fig_res, ax_res = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                            ax_res.axhline(0, color='r', linestyle='--')
                            ax_res.set_xlabel("Predicted Values")
                            ax_res.set_ylabel("Residuals")
                            ax_res.set_title("Residuals vs. Predicted Values")
                            st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                            st.header("📈 Residuals Distribution")

                            fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                            sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                            ax_res_dist.set_xlabel("Residuals")
                            ax_res_dist.set_ylabel("Frequency")
                            ax_res_dist.set_title("Distribution of Residuals")
                            st.pyplot(fig_res_dist)
                        
                            
                            
                            
                        elif model_name == 'gradient boost regressor':
                            
                                
                            mse = mean_squared_error(y_test, predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, predictions)
                            mape = mean_absolute_percentage_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            evs = explained_variance_score(y_test, predictions)
                            n = len(y_test)
                            k = X_test.shape[1]
                            adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                            st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                            metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                            st.markdown("---")

    # Visualization: Predicted vs Actual Values
                            st.header("📊 Predicted vs. Actual Values")

                            fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                            ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                            ax_pv.set_xlabel("Actual Values")
                            ax_pv.set_ylabel("Predicted Values")
                            ax_pv.set_title("Predicted vs. Actual Values")
                            st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                            st.header("📉 Residuals Plot")

                            residuals = y_test - predictions
                            fig_res, ax_res = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                            ax_res.axhline(0, color='r', linestyle='--')
                            ax_res.set_xlabel("Predicted Values")
                            ax_res.set_ylabel("Residuals")
                            ax_res.set_title("Residuals vs. Predicted Values")
                            st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                            st.header("📈 Residuals Distribution")

                            fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                            sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                            ax_res_dist.set_xlabel("Residuals")
                            ax_res_dist.set_ylabel("Frequency")
                            ax_res_dist.set_title("Distribution of Residuals")
                            st.pyplot(fig_res_dist)
                            
                            
                            
                        elif model_name == 'adaboost regressor':
                            mse = mean_squared_error(y_test, predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, predictions)
                            mape = mean_absolute_percentage_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            evs = explained_variance_score(y_test, predictions)
                            n = len(y_test)
                            k = X_test.shape[1]
                            adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                            st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                            metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                            st.markdown("---")

    # Visualization: Predicted vs Actual Values
                            st.header("📊 Predicted vs. Actual Values")

                            fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                            ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                            ax_pv.set_xlabel("Actual Values")
                            ax_pv.set_ylabel("Predicted Values")
                            ax_pv.set_title("Predicted vs. Actual Values")
                            st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                            st.header("📉 Residuals Plot")

                            residuals = y_test - predictions
                            fig_res, ax_res = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                            ax_res.axhline(0, color='r', linestyle='--')
                            ax_res.set_xlabel("Predicted Values")
                            ax_res.set_ylabel("Residuals")
                            ax_res.set_title("Residuals vs. Predicted Values")
                            st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                            st.header("📈 Residuals Distribution")

                            fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                            sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                            ax_res_dist.set_xlabel("Residuals")
                            ax_res_dist.set_ylabel("Frequency")
                            ax_res_dist.set_title("Distribution of Residuals")
                            st.pyplot(fig_res_dist)
                            
                            
                            
                        
                        
                        elif model_name == 'SVR':
                            mse = mean_squared_error(y_test, predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, predictions)
                            mape = mean_absolute_percentage_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            evs = explained_variance_score(y_test, predictions)
                            n = len(y_test)
                            k = X_test.shape[1]
                            adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                            st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                            metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                            st.markdown("---")

    # Visualization: Predicted vs Actual Values
                            st.header("📊 Predicted vs. Actual Values")

                            fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                            ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                            ax_pv.set_xlabel("Actual Values")
                            ax_pv.set_ylabel("Predicted Values")
                            ax_pv.set_title("Predicted vs. Actual Values")
                            st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                            st.header("📉 Residuals Plot")

                            residuals = y_test - predictions
                            fig_res, ax_res = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                            ax_res.axhline(0, color='r', linestyle='--')
                            ax_res.set_xlabel("Predicted Values")
                            ax_res.set_ylabel("Residuals")
                            ax_res.set_title("Residuals vs. Predicted Values")
                            st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                            st.header("📈 Residuals Distribution")

                            fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                            sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                            ax_res_dist.set_xlabel("Residuals")
                            ax_res_dist.set_ylabel("Frequency")
                            ax_res_dist.set_title("Distribution of Residuals")
                            st.pyplot(fig_res_dist)
                            
                            
                            
                            
                            
                        elif model_name == 'Random forest regressor':
                            mse = mean_squared_error(y_test, predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, predictions)
                            mape = mean_absolute_percentage_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            evs = explained_variance_score(y_test, predictions)
                            n = len(y_test)
                            k = X_test.shape[1]
                            adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                            st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                            metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                            st.markdown("---")

    # Visualization: Predicted vs Actual Values
                            st.header("📊 Predicted vs. Actual Values")

                            fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                            ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                            ax_pv.set_xlabel("Actual Values")
                            ax_pv.set_ylabel("Predicted Values")
                            ax_pv.set_title("Predicted vs. Actual Values")
                            st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                            st.header("📉 Residuals Plot")

                            residuals = y_test - predictions
                            fig_res, ax_res = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                            ax_res.axhline(0, color='r', linestyle='--')
                            ax_res.set_xlabel("Predicted Values")
                            ax_res.set_ylabel("Residuals")
                            ax_res.set_title("Residuals vs. Predicted Values")
                            st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                            st.header("📈 Residuals Distribution")

                            fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                            sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                            ax_res_dist.set_xlabel("Residuals")
                            ax_res_dist.set_ylabel("Frequency")
                            ax_res_dist.set_title("Distribution of Residuals")
                            st.pyplot(fig_res_dist)
                        
                            
                        
                        
                        elif model_name == 'k neighbour regressor':
                            mse = mean_squared_error(y_test, predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, predictions)
                            mape = mean_absolute_percentage_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            evs = explained_variance_score(y_test, predictions)
                            n = len(y_test)
                            k = X_test.shape[1]
                            adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                            st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                            metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                            metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                            st.markdown("---")

    # Visualization: Predicted vs Actual Values
                            st.header("📊 Predicted vs. Actual Values")

                            fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                            ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                            ax_pv.set_xlabel("Actual Values")
                            ax_pv.set_ylabel("Predicted Values")
                            ax_pv.set_title("Predicted vs. Actual Values")
                            st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                            st.header("📉 Residuals Plot")

                            residuals = y_test - predictions
                            fig_res, ax_res = plt.subplots(figsize=(7, 5))
                            sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                            ax_res.axhline(0, color='r', linestyle='--')
                            ax_res.set_xlabel("Predicted Values")
                            ax_res.set_ylabel("Residuals")
                            ax_res.set_title("Residuals vs. Predicted Values")
                            st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                            st.header("📈 Residuals Distribution")

                            fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                            sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                            ax_res_dist.set_xlabel("Residuals")
                            ax_res_dist.set_ylabel("Frequency")
                            ax_res_dist.set_title("Distribution of Residuals")
                            st.pyplot(fig_res_dist)
                        
                            
                            
                
                        # Button to save the model
                if model_name=='Logistic Regression':
                            
                                all_penalty=['l1','l2','elasticnet']
                                options = [1,10,20,30]
                                col1, col2 = st.columns([2, 1]) 
                                with col1:
                                        penalty_name = tuple(st.sidebar.multiselect('Select regularization', all_penalty))

    # In the second column, place the button and message
                                with col2:
                                        if st.sidebar.button('👁️ '):
                                            st.sidebar.info("tells which regularisation to prefer l1 = overfiting.l2=feature ectraction. eastic net =l1+l1")
                                

    # Alternatively, use an image as a button (optional, for a more customized approach)
                                

    # Streamlit multiselect to select options from the list
                                selected_options = st.sidebar.multiselect("C:", options) 

    # Display penalty_name as tuple
                                st.sidebar.write("Penalty Names :", penalty_name)

    # Display selected_options as a list (without index)
                                st.sidebar.write("regularisation parameter", selected_options)
                                
                                
                                
                            
                                parameter={'penalty':penalty_name,'C':selected_options}
                                tmodel=GridSearchCV(LogisticRegression(),param_grid=parameter,cv=5)
                                try:
                                    
                                    model.fit(X_train,y_train)
                                    predictions=model.predict(X_test)
                                except:
                                    st.sidebar.write(" ")
                                
                                
                                
                                if st.sidebar.button('hyperparameter tuning'):
                                    probabilities_with_param = model.predict_proba(X_test)[:,1] 
                                    acc_parameter = accuracy_score(y_test, predictions)
                                    conf_matrix_parameter = confusion_matrix(y_test, predictions)
                                    report_parameter = classification_report(y_test, predictions, output_dict=True)
                                    precision_parameter = precision_score(y_test, predictions)
                                    recall_parameter = recall_score(y_test, predictions)
                                    f1_parameter = f1_score(y_test, predictions)
                                    roc_auc_parameter = roc_auc_score(y_test, probabilities_with_param)
                                    mcc_parameter = matthews_corrcoef(y_test, predictions)
                                    info += f"\nModel Evaluation Metrics:\n"
                                    info += f"Accuracy: {acc_parameter:.4f}\n"
                                    info += f"Precision: {precision_parameter:.4f}\n"
                                    info += f"Recall: {recall_parameter:.4f}\n"
                                    info += f"F1-Score: {f1_parameter:.4f}\n"
                                    info += f"ROC AUC: {roc_auc_parameter:.4f}\n"
                                    info += f"Matthews Correlation Coefficient (MCC): {mcc_parameter:.4f}\n"

                                    # Convert confusion matrix to a string format and append
                                    conf_matrix_str = np.array2string(conf_matrix_parameter)
                                    info += f"Confusion Matrix:\n{conf_matrix_str}\n"

                                    # Append the classification report (already formatted as a string)
                                    info += f"Classification Report:\n{report_parameter}\n"
                                    

    # Streamlit App
                                    st.title("Classification Model Evaluation")

    # Display Accuracy
                                    st.metric("Accuracy", f"{acc_parameter:.2f}")

    # Display Precision, Recall, F1-Score
                                    st.subheader("Precision, Recall, and F1-Score")
                                    precision_val = report_parameter['weighted avg']['precision']
                                    recall_val = report_parameter['weighted avg']['recall']
                                    f1_val = report_parameter['weighted avg']['f1-score']
                                    st.write(f"**Precision:** {precision_val:.2f}")
                                    st.write(f"**Recall:** {recall_val:.2f}")
                                    st.write(f"**F1-Score:** {f1_val:.2f}")

    # Display ROC-AUC and MCC
                                    st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                    st.write(f"**ROC-AUC:** {roc_auc_parameter:.2f}")
                                    st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc_parameter:.2f}")

                                    st.markdown("---")

    # Confusion Matrix Heatmap
                                    st.subheader("Confusion Matrix")
                                    fig_cm, ax_cm = plt.subplots()
                                    sns.heatmap(conf_matrix_parameter, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                    ax_cm.set_xlabel('Predicted')
                                    ax_cm.set_ylabel('Actual')
                                    ax_cm.set_title('Confusion Matrix')
                                    st.pyplot(fig_cm)

    # Classification Report as a DataFrame
                                    st.subheader("Classification Report")
                                    report_df = pd.DataFrame(report_parameter).transpose()
                                    st.dataframe(report_df.style.highlight_max(axis=0))

    # ROC Curve
                                    st.subheader("ROC Curve")
                                    fpr, tpr, thresholds = roc_curve(y_test, probabilities_with_param)
                                    fig_roc, ax_roc = plt.subplots()
                                    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc_parameter:.2f}')
                                    ax_roc.plot([0, 1], [0, 1], 'k--')
                                    ax_roc.set_xlabel('False Positive Rate')
                                    ax_roc.set_ylabel('True Positive Rate')
                                    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                    ax_roc.legend(loc='lower right')
                                    st.pyplot(fig_roc)

    # Precision-Recall Curve
                                    st.subheader("Precision-Recall Curve")
                                    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities_with_param)
                                    fig_pr, ax_pr = plt.subplots()
                                    ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                    ax_pr.set_xlabel('Recall')
                                    ax_pr.set_ylabel('Precision')
                                    ax_pr.set_title('Precision-Recall Curve')
                                    ax_pr.legend(loc='upper right')
                                    st.pyplot(fig_pr)
                                    
                                    
                                    
                                    
                                
                if model_name=='decison tree classifier':
                                
                            crit=['gini', 'entropy', 'log_loss']
                            depth=[1, 2, 3, 4, 6, 8]
                            splitt=['best', 'random']
                            max_featur=['sqrt', 'log2']
                            
                            crite=st.sidebar.multiselect("selct the criterion",crit)
                            depthness=st.sidebar.multiselect("select the max_depth",depth)
                            splitness=st.sidebar.multiselect("select the splitter",splitt)
                            feature=st.sidebar.multiselect("selct the max_features",max_featur)
                            st.write(crite)
                            st.write(depthness)
                            st.write(splitness)
                            st.write(feature)
                                                
                            parameter = {
                                        'criterion': crite,
                                        'max_depth':depthness,
                                        'splitter':splitness,
                                        'max_features': feature
                                    }
                            
                            
                            model=GridSearchCV(DecisionTreeClassifier(),param_grid=parameter,cv=5)
                            try:
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write(" ")
                                
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                    probabilities_with_param = model.predict_proba(X_test)[:,1] 
                                    acc_parameter = accuracy_score(y_test, predictions)
                                    conf_matrix_parameter = confusion_matrix(y_test, predictions)
                                    report_parameter = classification_report(y_test, predictions, output_dict=True)
                                    precision_parameter = precision_score(y_test, predictions)
                                    recall_parameter = recall_score(y_test, predictions)
                                    f1_parameter = f1_score(y_test, predictions)
                                    roc_auc_parameter = roc_auc_score(y_test, probabilities_with_param)
                                    mcc_parameter = matthews_corrcoef(y_test, predictions)

    # Streamlit App
                                    st.title("Classification Model Evaluation")

    # Display Accuracy
                                    st.metric("Accuracy", f"{acc_parameter:.2f}")

    # Display Precision, Recall, F1-Score
                                    st.subheader("Precision, Recall, and F1-Score")
                                    precision_val = report_parameter['weighted avg']['precision']
                                    recall_val = report_parameter['weighted avg']['recall']
                                    f1_val = report_parameter['weighted avg']['f1-score']
                                    st.write(f"**Precision:** {precision_val:.2f}")
                                    st.write(f"**Recall:** {recall_val:.2f}")
                                    st.write(f"**F1-Score:** {f1_val:.2f}")

    # Display ROC-AUC and MCC
                                    st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                    st.write(f"**ROC-AUC:** {roc_auc_parameter:.2f}")
                                    st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc_parameter:.2f}")

                                    st.markdown("---")

    # Confusion Matrix Heatmap
                                    st.subheader("Confusion Matrix")
                                    fig_cm, ax_cm = plt.subplots()
                                    sns.heatmap(conf_matrix_parameter, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                    ax_cm.set_xlabel('Predicted')
                                    ax_cm.set_ylabel('Actual')
                                    ax_cm.set_title('Confusion Matrix')
                                    st.pyplot(fig_cm)

    # Classification Report as a DataFrame
                                    st.subheader("Classification Report")
                                    report_df = pd.DataFrame(report_parameter).transpose()
                                    st.dataframe(report_df.style.highlight_max(axis=0))

    # ROC Curve
                                    st.subheader("ROC Curve")
                                    fpr, tpr, thresholds = roc_curve(y_test, probabilities_with_param)
                                    fig_roc, ax_roc = plt.subplots()
                                    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc_parameter:.2f}')
                                    ax_roc.plot([0, 1], [0, 1], 'k--')
                                    ax_roc.set_xlabel('False Positive Rate')
                                    ax_roc.set_ylabel('True Positive Rate')
                                    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                    ax_roc.legend(loc='lower right')
                                    st.pyplot(fig_roc)

    # Precision-Recall Curve
                                    st.subheader("Precision-Recall Curve")
                                    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities_with_param)
                                    fig_pr, ax_pr = plt.subplots()
                                    ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                    ax_pr.set_xlabel('Recall')
                                    ax_pr.set_ylabel('Precision')
                                    ax_pr.set_title('Precision-Recall Curve')
                                    ax_pr.legend(loc='upper right')
                                    st.pyplot(fig_pr)
                if model_name=='Kneighbor Classifier':
                                
                            neigh=[3, 5, 6, 7, 10, 12, 15]
                            algori=['ball_tree', 'brute', 'kd_tree']
                            leave=[20, 30, 40 , 50]
                        
                            
                            crite=st.sidebar.multiselect("selct the criterion",neigh)
                            depthness=st.sidebar.multiselect("select the max_depth",algori)
                            splitness=st.sidebar.multiselect("select the splitter",leave)
                            # verbo=st.number_input("selct the verbos")
                            st.write(crite)
                            st.write(depthness)
                            st.write(splitness)
                            
                                                
                            parameter = {
                                        'n_neighbors': crite,
                                        'algorithm':depthness,
                                        'leaf_size':splitness,
                                    
                                    }
                            
                            
                                
                            model=GridSearchCV(KNeighborsClassifier(),param_grid=parameter,cv=5,verbose=3)
                            try:
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write(" ")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                    probabilities_with_param = model.predict_proba(X_test)[:,1] 
                                    acc_parameter = accuracy_score(y_test, predictions)
                                    conf_matrix_parameter = confusion_matrix(y_test, predictions)
                                    report_parameter = classification_report(y_test, predictions, output_dict=True)
                                    precision_parameter = precision_score(y_test, predictions)
                                    recall_parameter = recall_score(y_test, predictions)
                                    f1_parameter = f1_score(y_test, predictions)
                                    roc_auc_parameter = roc_auc_score(y_test, probabilities_with_param)
                                    mcc_parameter = matthews_corrcoef(y_test, predictions)

    # Streamlit App
                                    st.title("Classification Model Evaluation")

    # Display Accuracy
                                    st.metric("Accuracy", f"{acc_parameter:.2f}")

    # Display Precision, Recall, F1-Score
                                    st.subheader("Precision, Recall, and F1-Score")
                                    precision_val = report_parameter['weighted avg']['precision']
                                    recall_val = report_parameter['weighted avg']['recall']
                                    f1_val = report_parameter['weighted avg']['f1-score']
                                    st.write(f"**Precision:** {precision_val:.2f}")
                                    st.write(f"**Recall:** {recall_val:.2f}")
                                    st.write(f"**F1-Score:** {f1_val:.2f}")

    # Display ROC-AUC and MCC
                                    st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                    st.write(f"**ROC-AUC:** {roc_auc_parameter:.2f}")
                                    st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc_parameter:.2f}")

                                    st.markdown("---")

    # Confusion Matrix Heatmap
                                    st.subheader("Confusion Matrix")
                                    fig_cm, ax_cm = plt.subplots()
                                    sns.heatmap(conf_matrix_parameter, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                    ax_cm.set_xlabel('Predicted')
                                    ax_cm.set_ylabel('Actual')
                                    ax_cm.set_title('Confusion Matrix')
                                    st.pyplot(fig_cm)

    # Classification Report as a DataFrame
                                    st.subheader("Classification Report")
                                    report_df = pd.DataFrame(report_parameter).transpose()
                                    st.dataframe(report_df.style.highlight_max(axis=0))

    # ROC Curve
                                    st.subheader("ROC Curve")
                                    fpr, tpr, thresholds = roc_curve(y_test, probabilities_with_param)
                                    fig_roc, ax_roc = plt.subplots()
                                    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc_parameter:.2f}')
                                    ax_roc.plot([0, 1], [0, 1], 'k--')
                                    ax_roc.set_xlabel('False Positive Rate')
                                    ax_roc.set_ylabel('True Positive Rate')
                                    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                    ax_roc.legend(loc='lower right')
                                    st.pyplot(fig_roc)

    # Precision-Recall Curve
                                    st.subheader("Precision-Recall Curve")
                                    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities_with_param)
                                    fig_pr, ax_pr = plt.subplots()
                                    ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                    ax_pr.set_xlabel('Recall')
                                    ax_pr.set_ylabel('Precision')
                                    ax_pr.set_title('Precision-Recall Curve')
                                    ax_pr.legend(loc='upper right')
                                    st.pyplot(fig_pr)
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                
                
                if model_name=='gradient boosting classifer':
                                
                            estima=[50, 100, 200]
                            rate=[0.001, 0.1, 1, 1.5, 2, 2.5]
                            ccp=[1,2]
                        
                            
                            crite=st.sidebar.multiselect("selct the criterion",estima)
                            depthness=st.sidebar.multiselect("select the max_depth",rate)
                            splitness=st.sidebar.multiselect("select the ccp",ccp)
                            # verbo=st.sidebar.number_input("selct the verbos")
                            st.write(crite)
                            st.write(depthness)
                            st.write(splitness)
                            
                                                
                            parameter = {
                                        'n_estimators': crite,
                                        'learning_rate':depthness,
                                        'ccp_alpha':splitness,
                                    
                                    }
                            
                            
                            model=GridSearchCV(GradientBoostingClassifier(),param_grid=parameter,cv=5,verbose=3)
                            try:
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write("  ")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                    probabilities_with_param = model.predict_proba(X_test)[:,1] 
                                    acc_parameter = accuracy_score(y_test, predictions)
                                    conf_matrix_parameter = confusion_matrix(y_test, predictions)
                                    report_parameter = classification_report(y_test, predictions, output_dict=True)
                                    precision_parameter = precision_score(y_test, predictions)
                                    recall_parameter = recall_score(y_test, predictions)
                                    f1_parameter = f1_score(y_test, predictions)
                                    roc_auc_parameter = roc_auc_score(y_test, probabilities_with_param)
                                    mcc_parameter = matthews_corrcoef(y_test, predictions)

    # Streamlit App
                                    st.title("Classification Model Evaluation")

    # Display Accuracy
                                    st.metric("Accuracy", f"{acc_parameter:.2f}")

    # Display Precision, Recall, F1-Score
                                    st.subheader("Precision, Recall, and F1-Score")
                                    precision_val = report_parameter['weighted avg']['precision']
                                    recall_val = report_parameter['weighted avg']['recall']
                                    f1_val = report_parameter['weighted avg']['f1-score']
                                    st.write(f"**Precision:** {precision_val:.2f}")
                                    st.write(f"**Recall:** {recall_val:.2f}")
                                    st.write(f"**F1-Score:** {f1_val:.2f}")

    # Display ROC-AUC and MCC
                                    st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                    st.write(f"**ROC-AUC:** {roc_auc_parameter:.2f}")
                                    st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc_parameter:.2f}")

                                    st.markdown("---")

    # Confusion Matrix Heatmap
                                    st.subheader("Confusion Matrix")
                                    fig_cm, ax_cm = plt.subplots()
                                    sns.heatmap(conf_matrix_parameter, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                    ax_cm.set_xlabel('Predicted')
                                    ax_cm.set_ylabel('Actual')
                                    ax_cm.set_title('Confusion Matrix')
                                    st.pyplot(fig_cm)

    # Classification Report as a DataFrame
                                    st.subheader("Classification Report")
                                    report_df = pd.DataFrame(report_parameter).transpose()
                                    st.dataframe(report_df.style.highlight_max(axis=0))

    # ROC Curve
                                    st.subheader("ROC Curve")
                                    fpr, tpr, thresholds = roc_curve(y_test, probabilities_with_param)
                                    fig_roc, ax_roc = plt.subplots()
                                    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc_parameter:.2f}')
                                    ax_roc.plot([0, 1], [0, 1], 'k--')
                                    ax_roc.set_xlabel('False Positive Rate')
                                    ax_roc.set_ylabel('True Positive Rate')
                                    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                    ax_roc.legend(loc='lower right')
                                    st.pyplot(fig_roc)

    # Precision-Recall Curve
                                    st.subheader("Precision-Recall Curve")
                                    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities_with_param)
                                    fig_pr, ax_pr = plt.subplots()
                                    ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                    ax_pr.set_xlabel('Recall')
                                    ax_pr.set_ylabel('Precision')
                                    ax_pr.set_title('Precision-Recall Curve')
                                    ax_pr.legend(loc='upper right')
                                    st.pyplot(fig_pr)
                                
                                
        #                             learning_rates=np.logspace(-3,0,10)
        #                             accuracy_history=[]
        #                             for lr in learning_rate:
        #                                 model = GradientBoostingClassifier(learning_rate=lr, n_estimators=100, random_state=42)
        #                                 model.fit(X_train,y_train)
        #                                 y_pred = model.predict(X_test)
        
        # # Calculate accuracy and store the result
        #                                 accuracy = accuracy_score(y_test, y_pred)
        #                                 accuracy_history.append(accuracy)
        #                                 results_df = pd.DataFrame({
        #                                     'Learning Rate': learning_rates,
        #                                     'Accuracy': accuracy_history
        #                                 })
        #                             plt.figure(figsize=(10, 6))
        #                             sns.lineplot(data=results_df, x='Learning Rate', y='Accuracy', marker='o', color='skyblue')
        #                             plt.title('Learning Rate vs. Accuracy')
        #                             plt.xlabel('Learning Rate')
        #                             plt.ylabel('Accuracy')
        #                             plt.xscale('log')  # Use log scale for learning rates
        #                             plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
        #                             plt.grid()
        #                             plt.tight_layout()

                                    # # Display the plot in Streamlit
                                    # st.pyplot(plt)

                                    # # Optionally, display the DataFrame with results
                                    # st.write(results_df)
                                
                
                
                if model_name=='Adabbost classifer':
                                
                            estima=[50, 100, 200]
                            rate=[0.001, 0.1, 1, 1.5, 2, 2.5]
                            algora=["SAMME.R", "SAMME"]
                        
                            
                            crite=st.sidebar.multiselect("selct the criterion",estima)
                            depthness=st.sidebar.multiselect("select the max_depth",rate)
                            splitness=st.sidebar.multiselect("select the splitter",algora)
                            # verbo=st.sidebar.number_input("selct the verbos")
                            st.write(crite)
                            st.write(depthness)
                            st.write(splitness)
                            
                                                
                            parameter = {
                                        'n_estimators': crite,
                                        'learning_rate':depthness,
                                        'algorithm':splitness,
                                    
                                    }
                            
                            
                            model=GridSearchCV(AdaBoostClassifier(),param_grid=parameter,cv=5,verbose=3)
                            try:
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write(" ")
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                    probabilities_with_param = model.predict_proba(X_test)[:,1] 
                                    acc_parameter = accuracy_score(y_test, predictions)
                                    conf_matrix_parameter = confusion_matrix(y_test, predictions)
                                    report_parameter = classification_report(y_test, predictions, output_dict=True)
                                    precision_parameter = precision_score(y_test, predictions)
                                    recall_parameter = recall_score(y_test, predictions)
                                    f1_parameter = f1_score(y_test, predictions)
                                    roc_auc_parameter = roc_auc_score(y_test, probabilities_with_param)
                                    mcc_parameter = matthews_corrcoef(y_test, predictions)

    # Streamlit App
                                    st.title("Classification Model Evaluation")

    # Display Accuracy
                                    st.metric("Accuracy", f"{acc_parameter:.2f}")

    # Display Precision, Recall, F1-Score
                                    st.subheader("Precision, Recall, and F1-Score")
                                    precision_val = report_parameter['weighted avg']['precision']
                                    recall_val = report_parameter['weighted avg']['recall']
                                    f1_val = report_parameter['weighted avg']['f1-score']
                                    st.write(f"**Precision:** {precision_val:.2f}")
                                    st.write(f"**Recall:** {recall_val:.2f}")
                                    st.write(f"**F1-Score:** {f1_val:.2f}")

    # Display ROC-AUC and MCC
                                    st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                    st.write(f"**ROC-AUC:** {roc_auc_parameter:.2f}")
                                    st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc_parameter:.2f}")

                                    st.markdown("---")

    # Confusion Matrix Heatmap
                                    st.subheader("Confusion Matrix")
                                    fig_cm, ax_cm = plt.subplots()
                                    sns.heatmap(conf_matrix_parameter, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                    ax_cm.set_xlabel('Predicted')
                                    ax_cm.set_ylabel('Actual')
                                    ax_cm.set_title('Confusion Matrix')
                                    st.pyplot(fig_cm)

    # Classification Report as a DataFrame
                                    st.subheader("Classification Report")
                                    report_df = pd.DataFrame(report_parameter).transpose()
                                    st.dataframe(report_df.style.highlight_max(axis=0))

    # ROC Curve
                                    st.subheader("ROC Curve")
                                    fpr, tpr, thresholds = roc_curve(y_test, probabilities_with_param)
                                    fig_roc, ax_roc = plt.subplots()
                                    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc_parameter:.2f}')
                                    ax_roc.plot([0, 1], [0, 1], 'k--')
                                    ax_roc.set_xlabel('False Positive Rate')
                                    ax_roc.set_ylabel('True Positive Rate')
                                    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                    ax_roc.legend(loc='lower right')
                                    st.pyplot(fig_roc)

    # Precision-Recall Curve
                                    st.subheader("Precision-Recall Curve")
                                    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities_with_param)
                                    fig_pr, ax_pr = plt.subplots()
                                    ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                    ax_pr.set_xlabel('Recall')
                                    ax_pr.set_ylabel('Precision')
                                    ax_pr.set_title('Precision-Recall Curve')
                                    ax_pr.legend(loc='upper right')
                                    st.pyplot(fig_pr)
                                    
                                    
                                    
        #                             learning_rates=np.logspace(-3,0,10)
        #                             accuracy_history=[]
        #                             for lr in learning_rate:
        #                                 model = AdaBoostClassifier(learning_rate=lr, n_estimators=100, random_state=42)
        #                                 model.fit(X_train,y_train)
        #                                 y_pred = model.predict(X_test)
        
        # # Calculate accuracy and store the result
        #                                 accuracy = accuracy_score(y_test, y_pred)
        #                                 accuracy_history.append(accuracy)
        #                                 results_df = pd.DataFrame({
        #                                     'Learning Rate': learning_rates,
        #                                     'Accuracy': accuracy_history
        #                                 })
        #                             plt.figure(figsize=(10, 6))
        #                             sns.lineplot(data=results_df, x='Learning Rate', y='Accuracy', marker='o', color='skyblue')
        #                             plt.title('Learning Rate vs. Accuracy')
        #                             plt.xlabel('Learning Rate')
        #                             plt.ylabel('Accuracy')
        #                             plt.xscale('log')  # Use log scale for learning rates
        #                             plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
        #                             plt.grid()
        #                             plt.tight_layout()

                                # # Display the plot in Streamlit
                                #     st.pyplot(plt)

                                # # Optionally, display the DataFrame with results
                                #     st.write(results_df)
                                      
                                    
                                    
                if model_name=='random forest classifer':
                    
                            n_estimators= [100, 200, 500]
                            max_depth=[None, 10, 20, 30]
                            min_samples_split=[2, 5, 10]
                            min_samples_leaf= [1, 2, 5]
                            max_features= ['auto', 'sqrt', 'log2']
                                
                        
                            
                            crite=st.sidebar.multiselect("selct the n_estimators",n_estimators)
                            depthness=st.sidebar.multiselect("select the max_depth",max_depth)
                            splitness=st.sidebar.multiselect("select the min sample leaf",min_samples_leaf)
                            samples=st.sidebar.multiselect("select the min sample split",min_samples_split)
                            feature=st.sidebar.multiselect("select the sample split",max_features)
                            
                            # verbo=st.number_input("selct the verbos")
                            st.write(crite)
                            st.write(depthness)
                            st.write(splitness)
                            parameter = {
                                            'n_estimators':crite,
                                            'max_depth': depthness,
                                            'min_samples_split': samples,
                                                'min_samples_leaf': splitness,
                                            'max_features': feature
    }
                            
                                                
                    
                            
                            model=GridSearchCV(RandomForestClassifier(),param_grid=parameter,cv=5,verbose=3)
                            try:
                                
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write("bjb")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                    probabilities_with_param = model.predict_proba(X_test)[:,1] 
                                    acc_parameter = accuracy_score(y_test, predictions)
                                    conf_matrix_parameter = confusion_matrix(y_test, predictions)
                                    report_parameter = classification_report(y_test, predictions, output_dict=True)
                                    precision_parameter = precision_score(y_test, predictions)
                                    recall_parameter = recall_score(y_test, predictions)
                                    f1_parameter = f1_score(y_test, predictions)
                                    roc_auc_parameter = roc_auc_score(y_test, probabilities_with_param)
                                    mcc_parameter = matthews_corrcoef(y_test, predictions)

    # Streamlit App
                                    st.title("Classification Model Evaluation")

    # Display Accuracy
                                    st.metric("Accuracy", f"{acc_parameter:.2f}")

    # Display Precision, Recall, F1-Score
                                    st.subheader("Precision, Recall, and F1-Score")
                                    precision_val = report_parameter['weighted avg']['precision']
                                    recall_val = report_parameter['weighted avg']['recall']
                                    f1_val = report_parameter['weighted avg']['f1-score']
                                    st.write(f"**Precision:** {precision_val:.2f}")
                                    st.write(f"**Recall:** {recall_val:.2f}")
                                    st.write(f"**F1-Score:** {f1_val:.2f}")

    # Display ROC-AUC and MCC
                                    st.subheader("ROC-AUC and Matthews Correlation Coefficient")
                                    st.write(f"**ROC-AUC:** {roc_auc_parameter:.2f}")
                                    st.write(f"**Matthews Correlation Coefficient (MCC):** {mcc_parameter:.2f}")

                                    st.markdown("---")

    # Confusion Matrix Heatmap
                                    st.subheader("Confusion Matrix")
                                    fig_cm, ax_cm = plt.subplots()
                                    sns.heatmap(conf_matrix_parameter, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                    ax_cm.set_xlabel('Predicted')
                                    ax_cm.set_ylabel('Actual')
                                    ax_cm.set_title('Confusion Matrix')
                                    st.pyplot(fig_cm)

    # Classification Report as a DataFrame
                                    st.subheader("Classification Report")
                                    report_df = pd.DataFrame(report_parameter).transpose()
                                    st.dataframe(report_df.style.highlight_max(axis=0))

    # ROC Curve
                                    st.subheader("ROC Curve")
                                    fpr, tpr, thresholds = roc_curve(y_test, probabilities_with_param)
                                    fig_roc, ax_roc = plt.subplots()
                                    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc_parameter:.2f}')
                                    ax_roc.plot([0, 1], [0, 1], 'k--')
                                    ax_roc.set_xlabel('False Positive Rate')
                                    ax_roc.set_ylabel('True Positive Rate')
                                    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                    ax_roc.legend(loc='lower right')
                                    st.pyplot(fig_roc)

    # Precision-Recall Curve
                                    st.subheader("Precision-Recall Curve")
                                    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probabilities_with_param)
                                    fig_pr, ax_pr = plt.subplots()
                                    ax_pr.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
                                    ax_pr.set_xlabel('Recall')
                                    ax_pr.set_ylabel('Precision')
                                    ax_pr.set_title('Precision-Recall Curve')
                                    ax_pr.legend(loc='upper right')
                                    st.pyplot(fig_pr)   
                
                
                
                
            
                
                if model_name=="decison tree regressor":
                    
                            
                            max_depth=[None, 10, 20, 30]
                            min_samples_split=[2, 5, 10]
                            min_samples_leaf= [1, 2, 5]
                        
                                
                        
                            
                            
                            depthness=st.sidebar.multiselect("select the max_depth",max_depth)
                            splitness=st.sidebar.multiselect("select the min sample split",min_samples_split)
                            leaf=st.sidebar.multiselect("select the min sample leaf",min_samples_leaf)
                            
                            
                            
                            st.write(leaf)
                            st.write(depthness)
                            st.write(splitness)
                            parameter = {
                                            
                                            'max_depth': depthness,
                                            'min_samples_split': splitness,
                                                'min_samples_leaf': leaf
                                            
    }
                            
                                                
                    
                            
                            model=GridSearchCV(DecisionTreeRegressor(),param_grid=parameter,cv=5,n_jobs=-1)
                            try:
                                
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write("bjb")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                mse = mean_squared_error(y_test, predictions)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, predictions)
                                mape = mean_absolute_percentage_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                evs = explained_variance_score(y_test, predictions)
                                n = len(y_test)
                                k = X_test.shape[1]
                                adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                                st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                                metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                                metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                                st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                                st.markdown("---")

    # Visualization: Predicted vs Actual Values
                                st.header("📊 Predicted vs. Actual Values")

                                fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                                ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                                ax_pv.set_xlabel("Actual Values")
                                ax_pv.set_ylabel("Predicted Values")
                                ax_pv.set_title("Predicted vs. Actual Values")
                                st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                                st.header("📉 Residuals Plot")

                                residuals = y_test - predictions
                                fig_res, ax_res = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_xlabel("Predicted Values")
                                ax_res.set_ylabel("Residuals")
                                ax_res.set_title("Residuals vs. Predicted Values")
                                st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                                st.header("📈 Residuals Distribution")

                                fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                                sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                                ax_res_dist.set_xlabel("Residuals")
                                ax_res_dist.set_ylabel("Frequency")
                                ax_res_dist.set_title("Distribution of Residuals")
                                st.pyplot(fig_res_dist)                    
                
                
                
                if model_name=='Random forest regressor':
                    
                            n_estimators= [100, 200, 500]
                            max_depth=[None, 10, 20, 30]
                            min_samples_split=[2, 5, 10]
                        
                                
                        
                            
                            estimator=st.sidebar.multiselect("selct the n_estimators",n_estimators)
                            depth=st.sidebar.multiselect("select the max_depth",max_depth)
                            
                            sample_split=st.sidebar.multiselect("select the min sample split",min_samples_split)
                            
                            
                            
                            st.write(estimator)
                            st.write(depth)
                            st.write(sample_split)
                            parameter = {
                                            'n_estimators':estimator,
                                            'max_depth': depth,
                                            'min_samples_split': sample_split,
                            }
                            
                                                
                    
                            
                            model=GridSearchCV(RandomForestRegressor(),param_grid=parameter,cv=5,n_jobs=-1)
                            try:
                                
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write("bjb")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                mse = mean_squared_error(y_test, predictions)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, predictions)
                                mape = mean_absolute_percentage_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                evs = explained_variance_score(y_test, predictions)
                                n = len(y_test)
                                k = X_test.shape[1]
                                adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                                st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                                metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                                metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                                st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                                st.markdown("---")

    # Visualization: Predicted vs Actual Values
                                st.header("📊 Predicted vs. Actual Values")

                                fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                                ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                                ax_pv.set_xlabel("Actual Values")
                                ax_pv.set_ylabel("Predicted Values")
                                ax_pv.set_title("Predicted vs. Actual Values")
                                st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                                st.header("📉 Residuals Plot")

                                residuals = y_test - predictions
                                fig_res, ax_res = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_xlabel("Predicted Values")
                                ax_res.set_ylabel("Residuals")
                                ax_res.set_title("Residuals vs. Predicted Values")
                                st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                                st.header("📈 Residuals Distribution")

                                fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                                sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                                ax_res_dist.set_xlabel("Residuals")
                                ax_res_dist.set_ylabel("Frequency")
                                ax_res_dist.set_title("Distribution of Residuals")
                                st.pyplot(fig_res_dist)
                
                
                if model_name=='SVR':
                    
                        
                            
                            
                            c=[0.1, 1.0, 10.0]
                            epsilon=[0.01, 0.1, 0.5]
                            kernel=['linear', 'rbf']
                                
                        
                            
                            cr=st.sidebar.multiselect("selct the C",c)
                            epsi=st.sidebar.multiselect("select the epsilon",epsilon)
                            ker=st.sidebar.multiselect("select the kernel",kernel)

                            
                            
                            st.write(cr)
                            st.write(epsi)
                            st.write(ker)
                            parameter = {
                                "C": cr,
                                    "epsilon": epsi,
                                    "kernel":ker
                                            
                                                        }
                            
                                                
                    
                            
                            model=GridSearchCV(SVR(),param_grid=parameter,cv=5,n_jobs=-1)
                            try:
                                
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write("")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                mse = mean_squared_error(y_test, predictions)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, predictions)
                                mape = mean_absolute_percentage_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                evs = explained_variance_score(y_test, predictions)
                                n = len(y_test)
                                k = X_test.shape[1]
                                adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                                st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                                metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                                metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                                st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                                st.markdown("---")

    # Visualization: Predicted vs Actual Values
                                st.header("📊 Predicted vs. Actual Values")

                                fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                                ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                                ax_pv.set_xlabel("Actual Values")
                                ax_pv.set_ylabel("Predicted Values")
                                ax_pv.set_title("Predicted vs. Actual Values")
                                st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                                st.header("📉 Residuals Plot")

                                residuals = y_test - predictions
                                fig_res, ax_res = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_xlabel("Predicted Values")
                                ax_res.set_ylabel("Residuals")
                                ax_res.set_title("Residuals vs. Predicted Values")
                                st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                                st.header("📈 Residuals Distribution")

                                fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                                sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                                ax_res_dist.set_xlabel("Residuals")
                                ax_res_dist.set_ylabel("Frequency")
                                ax_res_dist.set_title("Distribution of Residuals")
                                st.pyplot(fig_res_dist)
                
                
                if model_name=='gradient boost regressor':
                    
                            n_estimators= [50, 100, 200]
                            learning_rate= [0.01, 0.1, 0.2]
                            max_depth= [3, 5, 7]
                            subsample= [0.8, 1.0]
                                
                        
                            
                            estima=st.sidebar.multiselect("selct the n_estimators",n_estimators)
                            rate=st.sidebar.multiselect("select the learning rate",learning_rate)
                            depth=st.sidebar.multiselect("select the ax_depth",max_depth)
                            samples=st.sidebar.multiselect("select the sub sample",subsample)
                            
                            
                            
                            st.write(estima)
                            st.write(rate)
                            st.write(depth)
                            st.write(samples)
                            parameter = {
                                "n_estimators": estima,
                                "learning_rate": rate,
                                "max_depth": depth,
                                "subsample": samples
    }
                            
                                                
                    
                            
                            model=GridSearchCV(GradientBoostingRegressor(),param_grid=parameter,cv=5,n_jobs=-1)
                            try:
                                
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write("")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                mse = mean_squared_error(y_test, predictions)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, predictions)
                                mape = mean_absolute_percentage_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                evs = explained_variance_score(y_test, predictions)
                                n = len(y_test)
                                k = X_test.shape[1]
                                adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                                st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                                metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                                metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                                st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                                st.markdown("---")

    # Visualization: Predicted vs Actual Values
                                st.header("📊 Predicted vs. Actual Values")

                                fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                                ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                                ax_pv.set_xlabel("Actual Values")
                                ax_pv.set_ylabel("Predicted Values")
                                ax_pv.set_title("Predicted vs. Actual Values")
                                st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                                st.header("📉 Residuals Plot")

                                residuals = y_test - predictions
                                fig_res, ax_res = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_xlabel("Predicted Values")
                                ax_res.set_ylabel("Residuals")
                                ax_res.set_title("Residuals vs. Predicted Values")
                                st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                                st.header("📈 Residuals Distribution")

                                fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                                sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                                ax_res_dist.set_xlabel("Residuals")
                                ax_res_dist.set_ylabel("Frequency")
                                ax_res_dist.set_title("Distribution of Residuals")
                                st.pyplot(fig_res_dist)
                
                
                if model_name=='k neighbour regressor':
                            n_neighbors=[3, 5, 10, 15]
                            weights=['uniform', 'distance']
                            algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
                            metric=['euclidean', 'manhattan']
                    
                        
                        
                            
                            neighbor=st.sidebar.multiselect("selct the n_eighbors",n_neighbors)
                            weight=st.sidebar.multiselect("select the weights",weights)
                            algorith=st.sidebar.multiselect("select the algorithm",algorithm)
                            met=st.sidebar.multiselect("select the metrics",metric)
                            
                            
                            
                            st.write(neighbor)
                            st.write(weight)
                            st.write(algorith)
                            st.write(met)
                            parameter = {
                                            "n_neighbors": neighbor,
                                            "weights": weight,
                                            "algorithm": algorith,
                                            "metric": met
                                            
    }
                            
                                                
                    
                            
                            model=GridSearchCV(KNeighborsRegressor(),param_grid=parameter,cv=5,n_jobs=-1)
                            try:
                                
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write("")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                mse = mean_squared_error(y_test, predictions)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, predictions)
                                mape = mean_absolute_percentage_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                evs = explained_variance_score(y_test, predictions)
                                n = len(y_test)
                                k = X_test.shape[1]
                                adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                                st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                                metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                                metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                                st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                                st.markdown("---")

    # Visualization: Predicted vs Actual Values
                                st.header("📊 Predicted vs. Actual Values")

                                fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                                ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                                ax_pv.set_xlabel("Actual Values")
                                ax_pv.set_ylabel("Predicted Values")
                                ax_pv.set_title("Predicted vs. Actual Values")
                                st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                                st.header("📉 Residuals Plot")

                                residuals = y_test - predictions
                                fig_res, ax_res = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_xlabel("Predicted Values")
                                ax_res.set_ylabel("Residuals")
                                ax_res.set_title("Residuals vs. Predicted Values")
                                st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                                st.header("📈 Residuals Distribution")

                                fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                                sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                                ax_res_dist.set_xlabel("Residuals")
                                ax_res_dist.set_ylabel("Frequency")
                                ax_res_dist.set_title("Distribution of Residuals")
                                st.pyplot(fig_res_dist)
                
                
                
                
                if model_name=='adaboost regressor':
                    
                            n_estimators= [100, 200, 500]
                            learning_rate=[0.01, 0.1, 1.0]
                            loss=['linear', 'square', 'exponential']
                            
                                
                        
                            
                            estima=st.sidebar.multiselect("selct the n_estimators",n_estimators)
                            rate=st.sidebar.multiselect("select the learning rate",learning_rate)
                            lo=st.sidebar.multiselect("select the loss",loss)
                            
                            
                            
                            st.write(estima)
                            st.write(rate)
                            st.write(lo)
                            parameter = {
                                "n_estimators": estima,
                                "learning_rate": rate,
                                "loss": lo
                                            
                                    }
                            
                                                
                    
                            
                            model=GridSearchCV(AdaBoostRegressor(),param_grid=parameter,cv=5,n_jobs=-1)
                            try:
                                
                                
                                model.fit(X_train,y_train)
                                predictions=model.predict(X_test)
                                st.write(model.best_params_)
                            except:
                                st.write("")
                            
                            
                            if st.sidebar.button('hyperparameter tuning'):
                                mse = mean_squared_error(y_test, predictions)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, predictions)
                                mape = mean_absolute_percentage_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                evs = explained_variance_score(y_test, predictions)
                                n = len(y_test)
                                k = X_test.shape[1]
                                adjusted_r2_score=1 - (1 - r2) * (n - 1) / (n - k - 1)
                            
                                st.header("📝 Regression Metrics")

    # Create a dictionary of metrics
                                metrics = {
                                    "Mean Absolute Error (MAE)": mae,
                                    "Mean Squared Error (MSE)": mse,
                                    "Root Mean Squared Error (RMSE)": rmse,
                                    "Mean Absolute Percentage Error (MAPE)": mape,
                                    "R² Score": r2,
                                    "Adjusted R² Score": adjusted_r2_score,
                                    "Explained Variance Score": evs
                                    }
                            
                                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                                metrics_df = metrics_df.reset_index().rename(columns={'index': 'Metric'})

    # Display metrics using Streamlit
                                st.table(metrics_df.style.format({"Value": "{:.4f}"}))

                                st.markdown("---")

    # Visualization: Predicted vs Actual Values
                                st.header("📊 Predicted vs. Actual Values")

                                fig_pv, ax_pv = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=y_test, y=predictions, alpha=0.6, ax=ax_pv)
                                ax_pv.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
                                ax_pv.set_xlabel("Actual Values")
                                ax_pv.set_ylabel("Predicted Values")
                                ax_pv.set_title("Predicted vs. Actual Values")
                                st.pyplot(fig_pv)

    # Visualization: Residuals Plot
                                st.header("📉 Residuals Plot")

                                residuals = y_test - predictions
                                fig_res, ax_res = plt.subplots(figsize=(7, 5))
                                sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax_res)
                                ax_res.axhline(0, color='r', linestyle='--')
                                ax_res.set_xlabel("Predicted Values")
                                ax_res.set_ylabel("Residuals")
                                ax_res.set_title("Residuals vs. Predicted Values")
                                st.pyplot(fig_res)

    # Visualization: Residuals Distribution
                                st.header("📈 Residuals Distribution")

                                fig_res_dist, ax_res_dist = plt.subplots(figsize=(7, 5))
                                sns.histplot(residuals, kde=True, bins=30, ax=ax_res_dist)
                                ax_res_dist.set_xlabel("Residuals")
                                ax_res_dist.set_ylabel("Frequency")
                                ax_res_dist.set_title("Distribution of Residuals")
                                st.pyplot(fig_res_dist)
                
           
                
                
                if st.sidebar.button('Save Model'):
                        
                            
                    model_file = io.BytesIO()
                    pickle.dump(model, model_file)
                    model_file.seek(0)

                    st.success('Model saved successfully as model.pkl!')

                        
                        
                    st.download_button(
                            label="Download Model",
                            data=model_file,
                            file_name="model.hd5",
                            mime="application/octet-stream"
                            )   
                        
if __name__ == "__main__":
    main()                                  
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                