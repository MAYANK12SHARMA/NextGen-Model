import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from scipy.stats import mode
from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    r2_score,
    roc_curve,
    precision_recall_curve,
)
from PIL import Image


class ClassificationMetrics:
    def __init__(self, y_test, predictions, probabilities):
        """
        Initialize the class with true labels, predicted labels, and predicted probabilities.
        :param y_test: Actual target values.
        :param predictions: Predicted target values (binary classification).
        :param probabilities: Predicted probabilities for the positive class.
        """
        self.y_test = y_test
        self.predictions = predictions
        self.probabilities = probabilities
        self.metrics = self.calculate_metrics()
        self.conf_matrix = confusion_matrix(y_test, predictions)
        self.report = classification_report(y_test, predictions, output_dict=True)

    def calculate_metrics(self):
        """Calculate classification metrics and return them as a dictionary."""
        acc = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = recall_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)
        roc_auc = roc_auc_score(self.y_test, self.probabilities)
        mcc = matthews_corrcoef(self.y_test, self.predictions)

        metrics = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Matthews Correlation Coefficient": mcc,
        }
        return metrics

    def display_metrics(self):
        """Display metrics as Streamlit components."""
        st.title("Classification Model Evaluation")
        st.metric("Accuracy", f"{self.metrics['Accuracy']:.2f}")

        st.subheader("Precision, Recall, and F1-Score")
        st.write(f"**Precision:** {self.metrics['Precision']:.2f}")
        st.write(f"**Recall:** {self.metrics['Recall']:.2f}")
        st.write(f"**F1-Score:** {self.metrics['F1-Score']:.2f}")

        st.subheader("ROC-AUC and Matthews Correlation Coefficient")
        st.write(f"**ROC-AUC:** {self.metrics['ROC-AUC']:.2f}")
        st.write(
            f"**Matthews Correlation Coefficient (MCC):** {self.metrics['Matthews Correlation Coefficient']:.2f}"
        )
        st.markdown("---")

    def plot_confusion_matrix(self):
        """Plot the confusion matrix as a heatmap."""
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(self.conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    def display_classification_report(self):
        """Display the classification report as a DataFrame."""
        st.subheader("Classification Report")
        report_df = pd.DataFrame(self.report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))

    def plot_roc_curve(self):
        """Plot the Receiver Operating Characteristic (ROC) curve."""
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {self.metrics["ROC-AUC"]:.2f}')
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    def plot_precision_recall_curve(self):
        """Plot the Precision-Recall curve."""
        st.subheader("Precision-Recall Curve")
        precision_vals, recall_vals, _ = precision_recall_curve(
            self.y_test, self.probabilities
        )
        fig, ax = plt.subplots()
        ax.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        st.pyplot(fig)

    def run_all_visualizations(self):
        """Run all visualizations."""
        self.plot_confusion_matrix()
        self.display_classification_report()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()


class RegressionMetrics:
    def __init__(self, y_test, predictions, X_test=None):
        """
        Initialize the class with true values, predicted values, and optionally the feature set.
        :param y_test: Actual target values.
        :param predictions: Predicted target values.
        :param X_test: Feature set used for testing, required for adjusted R².
        """
        self.y_test = y_test
        self.predictions = predictions
        self.X_test = X_test
        self.metrics = self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate regression metrics and return them as a dictionary."""
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.predictions)
        mape = mean_absolute_percentage_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)
        evs = explained_variance_score(self.y_test, self.predictions)

        n = len(self.y_test)
        k = self.X_test.shape[1] if self.X_test is not None else 0
        adjusted_r2_score = 1 - (1 - r2) * (n - 1) / (n - k - 1) if k > 0 else None

        metrics = {
            "Mean Absolute Error (MAE)": mae,
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "Mean Absolute Percentage Error (MAPE)": mape,
            "R² Score": r2,
            "Adjusted R² Score": adjusted_r2_score,
            "Explained Variance Score": evs,
        }
        return metrics

    def display_metrics(self):
        """Display metrics as a Streamlit table."""
        metrics_df = pd.DataFrame.from_dict(
            self.metrics, orient="index", columns=["Value"]
        ).reset_index()
        metrics_df = metrics_df.rename(columns={"index": "Metric"})
        st.header("📝 Regression Metrics")
        st.table(metrics_df.style.format({"Value": "{:.4f}"}))
        st.markdown("---")

    def plot_predicted_vs_actual(self):
        """Plot Predicted vs Actual values."""
        st.header("📊 Predicted vs. Actual Values")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=self.y_test, y=self.predictions, alpha=0.6, ax=ax)
        ax.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--",
        )  # Ideal line
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predicted vs. Actual Values")
        st.pyplot(fig)

    def plot_residuals(self):
        """Plot residuals vs predicted values."""
        st.header("📉 Residuals Plot")
        residuals = self.y_test - self.predictions
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(x=self.predictions, y=residuals, alpha=0.6, ax=ax)
        ax.axhline(0, color="r", linestyle="--")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Predicted Values")
        st.pyplot(fig)

    def plot_residuals_distribution(self):
        """Plot the distribution of residuals."""
        st.header("📈 Residuals Distribution")
        residuals = self.y_test - self.predictions
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(residuals, kde=True, bins=30, ax=ax)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Residuals")
        st.pyplot(fig)

    def run_all_visualizations(self):
        """Run all visualizations."""
        self.plot_predicted_vs_actual()
        self.plot_residuals()
        self.plot_residuals_distribution()


def train_bagging_models(selct_models, X_train, y_train, X_test):
    trained_models = []
    predictions = []

    for model in selct_models:
        # Bootstrap sampling (sampling with replacement)
        X_resampled, y_resampled = resample(X_train, y_train, random_state=42)

        # Fit the model on the bootstrapped data
        model.fit(X_resampled, y_resampled)

        # Add the trained model to the list
        predictions.append(model.predict(X_test))
        trained_models.append(model)

    predictions = np.array(predictions)
    majority_vote_predictions = []
    for i in range(predictions.shape[1]):
        m = mode(predictions[:, i])
        # Check if mode is scalar and handle accordingly
        majority_vote_predictions.append(
            m.mode[0] if isinstance(m.mode, np.ndarray) else m.mode
        )

    return majority_vote_predictions


def create_correlation_plot(df):
    for col in df.columns:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

    fig, ax = plt.subplots(figsize=((len(df.columns)), (len(df.columns))))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
    return fig


def create_pie_plot(df, column_to_plot):
    pie_data = df[column_to_plot].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)


def Modelling():
    
    all_type = ["select your type", "regression", "classification"]

    a = st.sidebar.selectbox("select your type", all_type)
    if a == "classification":
        data = st.file_uploader("Upload a Dataset", type=["csv", "xlsx"])
        if data is not None:
            if data.name.endswith("xlsx"):
                df = pd.read_excel(data)
            else:
                df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.sidebar.checkbox("EDA"):
                if st.checkbox("shape"):
                    st.write(f"Shape of the dataset: {df.shape}")

                if st.checkbox("Show Columns"):
                    all_columns = df.columns.to_list()

                    st.write(all_columns)

                if st.checkbox("Summary"):

                    st.write(df.describe())

                if st.checkbox("Show Null Values"):
                    null_info = df.isnull().sum().to_string() + "\n"

                    st.write(df.isnull().sum())

                #
                if st.checkbox("Show Selected Columns"):
                    selected_columns = st.multiselect(
                        "Select Columns", df.columns.to_list()
                    )
                    new_df = df[selected_columns]
                    st.dataframe(new_df)

                if st.checkbox("Show Value Counts"):
                    value_counts_info = df.iloc[:, -1].value_counts().to_string() + "\n"

                    st.write(df.iloc[:, -1].value_counts())

                if st.checkbox("Show unique value"):

                    st.write(df.nunique())

                if st.checkbox("Correlation Plot (Seaborn)"):
                    correlation_fig = create_correlation_plot(df)
                    st.write(df.head())

                if st.checkbox("Pie Plot"):
                    unique_value_counts = df.nunique()
                    filtered_columns = unique_value_counts[
                        unique_value_counts < 12
                    ].index.tolist()
                    all_columns = df.columns.to_list()
                    column_to_plot = st.selectbox(
                        "Select 1 Column for Pie Plot", filtered_columns
                    )
                    pie_fig = create_pie_plot(df, column_to_plot)

            if st.sidebar.checkbox("Generate Plot"):
                plots = [
                    "histogram",
                    "scatterplot",
                    "cumulative distribution plots",
                    "density plot",
                ]
                a = st.selectbox("choose any plot", plots)
                st.write(a)
                numerical_cols = df.select_dtypes(include=np.number).columns.to_list()
                if a == "histogram":
                    if st.checkbox("custom plots"):
                        a = st.selectbox("selct numerical columns", numerical_cols)
                        fig, ax = plt.subplots()
                        sns.histplot(df[a], kde=True, ax=ax)
                        ax.set_title(f"distribution of{a}")
                        st.pyplot(fig)

                    if st.checkbox("automatic plot"):

                        for col in numerical_cols:
                            fig, ax = plt.subplots()
                            sns.histplot(df[col], kde=True, ax=ax)
                            ax.set_title(f"Distribution of {col}")
                            st.pyplot(fig)

                if a == "scatterplot":
                    if st.checkbox("custom plots"):

                        xaxis = st.selectbox(
                            "selct numerical columns", numerical_cols, key="Xaxisselct"
                        )
                        yaxis = st.selectbox("selct numerical columns", numerical_cols)
                        fig, ax = plt.subplots()
                        sns.scatterplot(
                            x=df[xaxis],
                            y=df[yaxis],
                            ax=ax,
                            hue=df[xaxis],
                            palette="viridis",
                        )
                        ax.set_title(f"scatterplot betwwen the{xaxis} and {yaxis}")
                        st.pyplot(fig)

                    if st.checkbox("automated plot"):
                        for i, col1 in enumerate(numerical_cols):
                            for col2 in numerical_cols[i + 1 :]:
                                fig, ax = plt.subplots()
                                sns.scatterplot(
                                    x=df[col1],
                                    y=df[col2],
                                    ax=ax,
                                    hue=df[col1],
                                    palette="viridis",
                                )
                                ax.set_title(f"Scatterplot between {col1} and {col2}")
                                st.pyplot(fig)
                if a == "density plot":
                    for col in numerical_cols:
                        fig, ax = plt.subplots()
                        sns.kdeplot(df[col], shade=True, ax=ax, color="orange")
                        ax.set_title(f"Density Plot of {col}")
                        st.pyplot(fig)

                if a == "cumulative distribution plots":
                    for col in numerical_cols:
                        fig, ax = plt.subplots()
                        sns.ecdfplot(df[col], ax=ax, color="purple")
                        ax.set_title(f"CDF of {col}")
                        st.pyplot(fig)

            if st.sidebar.checkbox("feature scaling"):
                try:
                    all_columns = df.columns.to_list()
                    target = st.sidebar.selectbox(
                        "Select Target Column", all_columns, key="gopi"
                    )
                    feature_columns = [col for col in all_columns if col != target]
                    label_encoder = LabelEncoder()
                    for col in feature_columns:
                        df[col] = label_encoder.fit_transform(df[col])
                    X = df[feature_columns]
                    df[target] = label_encoder.fit_transform(df[target])
                    y = df[target]
                    imp_feature = {"random forest", "correlation"}
                    a = st.selectbox("features selction", imp_feature)
                    if a == "random forest":
                        # X = pd.get_dummies(X, drop_first=True)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.20, random_state=42
                        )
                        randomforest = RandomForestClassifier(n_estimators=100)
                        randomforest.fit(X_train, y_train)
                        selected_features = (
                            pd.Series(
                                randomforest.feature_importances_, index=X_train.columns
                            )
                            .sort_values(ascending=False)
                            .index
                        )
                        abcd = selected_features = (
                            pd.Series(
                                randomforest.feature_importances_, index=X_train.columns
                            )
                            .sort_values(ascending=False)
                            .index
                        )

                        st.write(selected_features)

                    if a == "correlation":

                        def create_correlation_plot(df):
                            for col in df.columns:
                                label_encoder = LabelEncoder()
                                df[col] = label_encoder.fit_transform(df[col])

                            fig, ax = plt.subplots(
                                figsize=((len(df.columns)), (len(df.columns)))
                            )
                            sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
                            st.pyplot(fig)
                            return fig

                        correlation_fig = create_correlation_plot(df)

                except:
                    st.write("")

            all_columns = df.columns.to_list()
            target = st.sidebar.selectbox("Select Target Column", all_columns)
            default_columns = [col for col in all_columns if col != target]

            feature_columns = st.sidebar.multiselect(
                "Select Feature Columns",
                all_columns,  # All columns as the options
                default=default_columns,  # Default selection (all except target)
            )
            if len(feature_columns) == 0:
                st.error("Please select at least one feature column.")
            else:

                label_encoder = LabelEncoder()
                for col in feature_columns:
                    df[col] = label_encoder.fit_transform(df[col])
                X = df[feature_columns]
                df[target] = label_encoder.fit_transform(df[target])
                y = df[target]
                # X = pd.get_dummies(X, drop_first=True)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=42
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            all_models = [
                # 'Stacking Classifer':'StackingClassi',
                LogisticRegression(),
                DecisionTreeClassifier(),
                GradientBoostingClassifier(),
                AdaBoostClassifier(),
                SVC(),
                RandomForestClassifier(),
                KNeighborsClassifier(),
            ]
            if st.sidebar.checkbox("ensemble"):
                st.subheader("Boothstrap Aggreation")
                # models=['custom bagging','custom boosting']

                selct_models = st.multiselect(
                    "Select models for ",
                    options=all_models,
                    format_func=lambda x: type(x).__name__,
                )
                st.write(selct_models)
                if selct_models:
                    if st.checkbox("run"):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.20, random_state=42
                        )
                        # Train the models
                        predictions = train_bagging_models(
                            selct_models, X_train, y_train, X_test
                        )

                        accuracy = accuracy_score(y_test, predictions)

                        st.write(f"Accuracy: {accuracy:.4f}")

                        acc = accuracy_score(y_test, predictions)
                        conf_matrix = confusion_matrix(y_test, predictions)

                        report = classification_report(
                            y_test, predictions, output_dict=True
                        )

                        precision = precision_score(y_test, predictions)

                        recall = recall_score(y_test, predictions)

                        f1 = f1_score(y_test, predictions)

            all_models = {
                "Logistic Regression": LogisticRegression(),
                "Decison Tree Classifer": DecisionTreeClassifier(),
                "Gradient Boosting Classifer": GradientBoostingClassifier(),
                "Adaboost Classifer": AdaBoostClassifier(),
                "Support Vector Classifer": SVC(),
                "Random Forest Classifier": RandomForestClassifier(),
                "KNeighborClassifier": KNeighborsClassifier(),
            }

            if st.sidebar.checkbox("Stacking "):
                selected_base_models = st.multiselect(
                    "Select base models to be stacked", list(all_models.keys())
                )

                # User selection for final model to use after stacking
                final_model = st.selectbox(
                    "Select the final model to be used", list(all_models.keys())
                )

                try:
                    base_models = [
                        (model_name, all_models[model_name])
                        for model_name in selected_base_models
                    ]

                    st.write("Selected base models for stacking:")
                    for model in base_models:
                        st.write(model[0])

                    # Select final model
                    final_model_instance = all_models[final_model]

                    st.write(f"Final model: {final_model}")
                    model = StackingClassifier(
                        estimators=base_models, final_estimator=final_model_instance
                    )

                    if st.button("run"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        probabilities = model.predict_proba(X_test)[:, 1]
                        stacks = ClassificationMetrics(y_test, predictions, probabilities)
                        stacks.display_metrics()
                        stacks.run_all_visualizations()
                except:
                    st.write("")

            try:

                model_name = st.sidebar.selectbox(
                    "Select Model", options=["Select a model"] + list(all_models.keys())
                )

                model = all_models[model_name]
                st.write(model)
            except:

                st.write("")

            if st.sidebar.button("Run model"):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                if model_name == "Logistic Regression":
                    try:

                        probabilities = model.predict_proba(X_test)[:, 1]
                        logistic = ClassificationMetrics(y_test, predictions, probabilities)
                        logistic.display_metrics()
                        logistic.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "Decison Tree Classifer":
                    try:

                        probabilities = model.predict_proba(X_test)[:, 1]
                        decisonc = ClassificationMetrics(y_test, predictions, probabilities)
                        decisonc.display_metrics()
                        decisonc.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "Gradient Boosting Classifer":
                    try:

                        probabilities = model.predict_proba(X_test)[:, 1]
                        Gradientc = ClassificationMetrics(
                            y_test, predictions, probabilities
                        )
                        Gradientc.display_metrics()
                        Gradientc.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "Adaboost Classifer":
                    try:

                        probabilities = model.predict_proba(X_test)[:, 1]
                        Adac = ClassificationMetrics(y_test, predictions, probabilities)
                        Adac.display_metrics()
                        Adac.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "Support Vector Classifer":
                    try:

                        probabilities = model.predict_proba(X_test)[:, 1]
                        supportc = ClassificationMetrics(y_test, predictions, probabilities)
                        supportc.display_metrics()
                        supportc.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "Random Forest Classifier":
                    try:

                        probabilities = model.predict_proba(X_test)[:, 1]
                        randomc = ClassificationMetrics(y_test, predictions, probabilities)
                        randomc.display_metrics()
                        randomc.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "KNeighborClassifier":
                    try:

                        probabilities = model.predict_proba(X_test)[:, 1]
                        Kneighc = ClassificationMetrics(y_test, predictions, probabilities)
                        Kneighc.display_metrics()
                        Kneighc.run_all_visualizations()
                    except:
                        st.write("")
            if st.sidebar.checkbox("optimisation"):
                if model_name == "Logistic Regression":
                    # try:

                    all_penalty = ["l1", "l2", "elasticnet"]
                    options = [1, 10, 20, 30]
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        penalty_name = tuple(
                            st.sidebar.multiselect("Select regularization", all_penalty)
                        )

                    with col2:
                        if st.sidebar.button("👁️ "):
                            st.sidebar.info(
                                "tells which regularisation to prefer l1 = overfiting.l2=feature ectraction. eastic net =l1+l1"
                            )

                    selected_options = st.sidebar.multiselect("C:", options)

                    st.sidebar.write("Penalty Names :", penalty_name)

                    st.sidebar.write("regularisation parameter", selected_options)

                    parameter = {"penalty": penalty_name, "C": selected_options}
                    model = RandomizedSearchCV(
                        LogisticRegression(), param_distributions=parameter, cv=5
                    )

                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        probabilities = model.predict_proba(X_test)[:, 1]
                        logitichc = ClassificationMetrics(
                            y_test, predictions, probabilities
                        )
                        logitichc.display_metrics()
                        logitichc.run_all_visualizations()
                # except:
                #             st.sidebar.write(" ")

                if model_name == "Decison Tree Classifer":
                    # try:
                    crit = ["gini", "entropy", "log_loss"]
                    depth = [1, 2, 3, 4, 6, 8]
                    splitt = ["best", "random"]
                    max_featur = ["sqrt", "log2"]

                    crite = st.sidebar.multiselect("selct the criterion", crit)
                    depthness = st.sidebar.multiselect("select the max_depth", depth)
                    splitness = st.sidebar.multiselect("select the splitter", splitt)
                    feature = st.sidebar.multiselect("selct the max_features", max_featur)
                    st.write(crite)
                    st.write(depthness)
                    st.write(splitness)
                    st.write(feature)

                    parameter = {
                        "criterion": crite,
                        "max_depth": depthness,
                        "splitter": splitness,
                        "max_features": feature,
                    }

                    model = RandomizedSearchCV(
                        DecisionTreeClassifier(), param_distributions=parameter, cv=5
                    )

                    # except:
                    #         st.write(" ")
                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)

                        predictions = model.predict(X_test)

                        st.write(model.best_params_)

                        probabilities = model.predict_proba(X_test)[:, 1]

                        decisonhc = ClassificationMetrics(
                            y_test, predictions, probabilities
                        )
                        decisonhc.display_metrics()
                        decisonhc.run_all_visualizations()

                if model_name == "KNeighborClassifier":
                    neigh = [3, 5, 6, 7, 10, 12, 15]
                    algori = ["ball_tree", "brute", "kd_tree"]
                    leave = [20, 30, 40, 50]

                    crite = st.sidebar.multiselect("selct the criterion", neigh)
                    depthness = st.sidebar.multiselect("select the max_depth", algori)
                    splitness = st.sidebar.multiselect("select the splitter", leave)
                    # verbo=st.number_input("selct the verbos")
                    st.write(crite)
                    st.write(depthness)
                    st.write(splitness)

                    parameter = {
                        "n_neighbors": crite,
                        "algorithm": depthness,
                        "leaf_size": splitness,
                    }

                    model = RandomizedSearchCV(
                        KNeighborsClassifier(),
                        param_distributions=parameter,
                        cv=5,
                        verbose=3,
                    )

                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        st.write(model.best_params_)
                        probabilities = model.predict_proba(X_test)[:, 1]
                        Kneighch = ClassificationMetrics(y_test, predictions, probabilities)
                        Kneighch.display_metrics()
                        Kneighch.run_all_visualizations()

                if model_name == "Gradient Boosting Classifer":

                    estima = [50, 100, 200]
                    rate = [0.001, 0.1, 1, 1.5, 2, 2.5]
                    ccp = [1, 2]

                    crite = st.sidebar.multiselect("selct the criterion", estima)
                    depthness = st.sidebar.multiselect("select the max_depth", rate)
                    splitness = st.sidebar.multiselect("select the ccp", ccp)
                    # verbo=st.sidebar.number_input("selct the verbos")
                    st.write(crite)
                    st.write(depthness)
                    st.write(splitness)

                    parameter = {
                        "n_estimators": crite,
                        "learning_rate": depthness,
                        "ccp_alpha": splitness,
                    }

                    model = RandomizedSearchCV(
                        GradientBoostingClassifier(),
                        param_distributions=parameter,
                        cv=5,
                        verbose=3,
                    )

                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        # st.write(model.best_params_)
                        probabilities = model.predict_proba(X_test)[:, 1]
                        gradientc = ClassificationMetrics(
                            y_test, predictions, probabilities
                        )
                        gradientc.display_metrics()
                        gradientc.run_all_visualizations()

                if model_name == "Random Forest Classifier":

                    n_estimators = [100, 200, 500]
                    max_depth = [None, 10, 20, 30]
                    min_samples_split = [2, 5, 10]
                    min_samples_leaf = [1, 2, 5]
                    max_features = ["auto", "sqrt", "log2"]

                    crite = st.sidebar.multiselect("selct the n_estimators", n_estimators)
                    depthness = st.sidebar.multiselect("select the max_depth", max_depth)
                    splitness = st.sidebar.multiselect(
                        "select the min sample leaf", min_samples_leaf
                    )
                    samples = st.sidebar.multiselect(
                        "select the min sample split", min_samples_split
                    )
                    feature = st.sidebar.multiselect(
                        "select the sample split", max_features
                    )

                    # verbo=st.number_input("selct the verbos")
                    st.write(crite)
                    st.write(depthness)
                    st.write(splitness)
                    parameter = {
                        "n_estimators": crite,
                        "max_depth": depthness,
                        "min_samples_split": samples,
                        "min_samples_leaf": splitness,
                        "max_features": feature,
                    }

                    model = RandomizedSearchCV(
                        RandomForestClassifier(),
                        param_distributions=parameter,
                        cv=5,
                        verbose=3,
                    )

                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        st.write(model.best_params_)
                        probabilities = model.predict_proba(X_test)[:, 1]

                        Randomch = ClassificationMetrics(y_test, predictions, probabilities)
                        Randomch.display_metrics()
                        Randomch.run_all_visualizations()

                if model_name == "Adaboost Classifer":

                    estima = [50, 100, 200]
                    rate = [0.001, 0.1, 1, 1.5, 2, 2.5]
                    algora = ["SAMME.R", "SAMME"]

                    crite = st.sidebar.multiselect("selct the criterion", estima)
                    depthness = st.sidebar.multiselect("select the max_depth", rate)
                    splitness = st.sidebar.multiselect("select the splitter", algora)
                    # verbo=st.sidebar.number_input("selct the verbos")
                    st.write(crite)
                    st.write(depthness)
                    st.write(splitness)
                    parameter = {
                        "n_estimators": crite,
                        "learning_rate": depthness,
                        "algorithm": splitness,
                    }
                    tmodel = RandomizedSearchCV(
                        AdaBoostClassifier(), param_distributions=parameter, cv=5, verbose=3
                    )
                    # st.write(model.best_params_)

                    if st.button("run optimise"):
                        tmodel.fit(X_train, y_train)
                        predictions = tmodel.predict(X_test)

                        probabilities = tmodel.predict_proba(X_test)[:, 1]

                        Adachc = ClassificationMetrics(y_test, predictions, probabilities)
                        Adachc.display_metrics()
                        Adachc.run_all_visualizations()


    if a == "regression":
        data = st.file_uploader("Upload a Dataset", type=["csv", "xlsx"])
        if data is not None:
            if data.name.endswith("xlsx"):
                df = pd.read_excel(data)
            else:
                df = pd.read_csv(data)
            st.dataframe(df.head())
            if st.sidebar.checkbox("EDA"):
                if st.checkbox("shape"):
                    st.write(f"Shape of the dataset: {df.shape}")

                if st.checkbox("Show Columns"):
                    all_columns = df.columns.to_list()

                    st.write(all_columns)

                if st.checkbox("Summary"):

                    st.write(df.describe())

                if st.checkbox("Show Null Values"):
                    null_info = df.isnull().sum().to_string() + "\n"

                    st.write(df.isnull().sum())

                #
                if st.checkbox("Show Selected Columns"):
                    selected_columns = st.multiselect(
                        "Select Columns", df.columns.to_list()
                    )
                    new_df = df[selected_columns]
                    st.dataframe(new_df)

                if st.checkbox("Show Value Counts"):
                    value_counts_info = df.iloc[:, -1].value_counts().to_string() + "\n"

                    st.write(df.iloc[:, -1].value_counts())

                if st.checkbox("Show unique value"):

                    st.write(df.nunique())

                if st.checkbox("Correlation Plot (Seaborn)"):
                    correlation_fig = create_correlation_plot(df)
                    st.write(df.head())

                if st.checkbox("Pie Plot"):
                    unique_value_counts = df.nunique()
                    filtered_columns = unique_value_counts[
                        unique_value_counts < 12
                    ].index.tolist()
                    all_columns = df.columns.to_list()
                    column_to_plot = st.selectbox(
                        "Select 1 Column for Pie Plot", filtered_columns
                    )
                    pie_fig = create_pie_plot(df, column_to_plot)

            if st.sidebar.checkbox("Generate Plot"):
                plots = [
                    "histogram",
                    "scatterplot",
                    "cumulative distribution plots",
                    "density plot",
                ]
                a = st.selectbox("choose any plot", plots)
                st.write(a)
                numerical_cols = df.select_dtypes(include=np.number).columns.to_list()
                if a == "histogram":
                    if st.checkbox("custom plots"):
                        a = st.selectbox("selct numerical columns", numerical_cols)
                        fig, ax = plt.subplots()
                        sns.histplot(df[a], kde=True, ax=ax)
                        ax.set_title(f"distribution of{a}")
                        st.pyplot(fig)

                    if st.checkbox("automatic plot"):

                        for col in numerical_cols:
                            fig, ax = plt.subplots()
                            sns.histplot(df[col], kde=True, ax=ax)
                            ax.set_title(f"Distribution of {col}")
                            st.pyplot(fig)

                if a == "scatterplot":
                    if st.checkbox("custom plots"):

                        xaxis = st.selectbox(
                            "selct numerical columns", numerical_cols, key="Xaxisselct"
                        )
                        yaxis = st.selectbox("selct numerical columns", numerical_cols)
                        fig, ax = plt.subplots()
                        sns.scatterplot(
                            x=df[xaxis],
                            y=df[yaxis],
                            ax=ax,
                            hue=df[xaxis],
                            palette="viridis",
                        )
                        ax.set_title(f"scatterplot betwwen the{xaxis} and {yaxis}")
                        st.pyplot(fig)

                    if st.checkbox("automated plot"):
                        for i, col1 in enumerate(numerical_cols):
                            for col2 in numerical_cols[i + 1 :]:
                                fig, ax = plt.subplots()
                                sns.scatterplot(
                                    x=df[col1],
                                    y=df[col2],
                                    ax=ax,
                                    hue=df[col1],
                                    palette="viridis",
                                )
                                ax.set_title(f"Scatterplot between {col1} and {col2}")
                                st.pyplot(fig)
                if a == "density plot":
                    for col in numerical_cols:
                        fig, ax = plt.subplots()
                        sns.kdeplot(df[col], shade=True, ax=ax, color="orange")
                        ax.set_title(f"Density Plot of {col}")
                        st.pyplot(fig)

                if a == "cumulative distribution plots":
                    for col in numerical_cols:
                        fig, ax = plt.subplots()
                        sns.ecdfplot(df[col], ax=ax, color="purple")
                        ax.set_title(f"CDF of {col}")
                        st.pyplot(fig)

            if st.sidebar.checkbox("Feature Scaling"):
                try:
                    all_columns = df.columns.to_list()
                    target = st.sidebar.selectbox("Select Target Column", all_columns)
                    feature_columns = [col for col in all_columns if col != target]
                    label_encoder = LabelEncoder()
                    for col in feature_columns:
                        df[col] = label_encoder.fit_transform(df[col])
                    X = df[feature_columns]
                    df[target] = label_encoder.fit_transform(df[target])
                    y = df[target]
                    imp_feature = {"random forest", "correlation"}
                    a = st.selectbox("features selction", imp_feature)
                    if a == "random forest":
                        # X = pd.get_dummies(X, drop_first=True)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.20, random_state=42
                        )
                        randomforest = RandomForestRegressor(n_estimators=100)
                        randomforest.fit(X_train, y_train)
                        selected_features = (
                            pd.Series(
                                randomforest.feature_importances_, index=X_train.columns
                            )
                            .sort_values(ascending=False)
                            .index
                        )
                        abcd = selected_features = (
                            pd.Series(
                                randomforest.feature_importances_, index=X_train.columns
                            )
                            .sort_values(ascending=False)
                            .index
                        )

                        st.write(selected_features)

                    if a == "correlation":

                        def create_correlation_plot(df):
                            for col in df.columns:
                                label_encoder = LabelEncoder()
                                df[col] = label_encoder.fit_transform(df[col])

                            fig, ax = plt.subplots(
                                figsize=((len(df.columns)), (len(df.columns)))
                            )
                            sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
                            st.pyplot(fig)
                            return fig

                        correlation_fig = create_correlation_plot(df)

                except:
                    st.write("")

            all_columns = df.columns.to_list()
            target = st.sidebar.selectbox("Select Target Column", all_columns)
            default_columns = [col for col in all_columns if col != target]

            feature_columns = st.sidebar.multiselect(
                "Select Feature Columns",
                all_columns,  # All columns as the options
                default=default_columns,  # Default selection (all except target)
            )
            if len(feature_columns) == 0:
                st.error("Please select at least one feature column.")
            else:

                label_encoder = LabelEncoder()
                for col in feature_columns:
                    df[col] = label_encoder.fit_transform(df[col])
                X = df[feature_columns]
                df[target] = label_encoder.fit_transform(df[target])
                y = df[target]
                # X = pd.get_dummies(X, drop_first=True)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=42
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            all_models = [
                # 'Stacking Classifer':'StackingClassi',
                LinearRegression(),
                DecisionTreeRegressor(),
                GradientBoostingRegressor(),
                AdaBoostRegressor(),
                SVR(),
                RandomForestRegressor(),
                KNeighborsRegressor(),
            ]
            if st.sidebar.checkbox("Ensemble"):
                st.subheader("Boothstrap Aggreation")
                # models=['custom bagging','custom boosting']

                selct_models = st.multiselect(
                    "Select models for ",
                    options=all_models,
                    format_func=lambda x: type(x).__name__,
                )
                st.write(selct_models)
                if selct_models:
                    if st.checkbox("run"):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.20, random_state=42
                        )
                        # Train the models
                        predictions = train_bagging_models(
                            selct_models, X_train, y_train, X_test
                        )

                        accuracy = accuracy_score(y_test, predictions)

                        st.write(f"Accuracy: {accuracy:.4f}")

                        acc = accuracy_score(y_test, predictions)
                        conf_matrix = confusion_matrix(y_test, predictions)

                        report = classification_report(
                            y_test, predictions, output_dict=True
                        )

                        precision = precision_score(y_test, predictions)

                        recall = recall_score(y_test, predictions)

                        f1 = f1_score(y_test, predictions)

            all_models = {
                # 'Stacking Classifer':'StackingClassi',
                "Linear Regression": LinearRegression(),
                "Decison Tree Regressor": DecisionTreeRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Adaboost Regressor": AdaBoostRegressor(),
                "Support Vector Regressor": SVR(),
                "Random Forest Regressor": RandomForestRegressor(),
                "KNeighborRegressor": KNeighborsRegressor(),
            }
            if st.sidebar.checkbox("Stacking "):
                selected_base_models = st.multiselect(
                    "Select base models to be stacked", list(all_models.keys())
                )

                # User selection for final model to use after stacking
                final_model = st.selectbox(
                    "Select the final model to be used", list(all_models.keys())
                )

                base_models = [
                    (model_name, all_models[model_name])
                    for model_name in selected_base_models
                ]

                st.write("Selected base models for stacking:")
                for model in base_models:
                    st.write(model[0])

                # Select final model
                final_model_instance = all_models[final_model]

                st.write(f"Final model: {final_model}")
                model = StackingRegressor(
                    estimators=base_models, final_estimator=final_model_instance
                )

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                if st.button("run"):
                    stack_r = RegressionMetrics(y_test, predictions, X_test)
                    stack_r.display_metrics()
                    stack_r.run_all_visualizations()

            try:

                model_name = st.sidebar.selectbox(
                    "Select Model", options=["Select a model"] + list(all_models.keys())
                )

                model = all_models[model_name]
                st.write(model)
            except:
                st.write("")
            if st.sidebar.button("Run model"):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                if model_name == "Linear Regression":
                    try:

                        predictions = model.predict(X_test)
                        linear = RegressionMetrics(y_test, predictions, X_test)
                        linear.display_metrics()
                        linear.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "Decison Tree Regressor":
                    try:

                        predictions = model.predict(X_test)
                        decison = RegressionMetrics(y_test, predictions, X_test)
                        decison.display_metrics()
                        decison.run_all_visualizations()
                    except:
                        st.write("")
                if model_name == "Gradient Boosting Regressor":
                    try:
                        predictions = model.predict(X_test)
                        gradient = RegressionMetrics(y_test, predictions, X_test)
                        gradient.display_metrics()
                        gradient.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "Adaboost Regressor":
                    try:

                        predictions = model.predict(X_test)
                        adaboost = RegressionMetrics(y_test, predictions, X_test)
                        adaboost.display_metrics()
                        adaboost.run_all_visualizations()
                    except:
                        st.write("")
                if model_name == "Support Vector Regressor":
                    try:
                        predictions = model.predict(X_test)
                        support = RegressionMetrics(y_test, predictions, X_test)
                        support.display_metrics()
                        support.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "Random Forest Regressor":
                    try:
                        predictions = model.predict(X_test)
                        random = RegressionMetrics(y_test, predictions, X_test)
                        random.display_metrics()
                        random.run_all_visualizations()

                    except:
                        st.write("")
                if model_name == "KNeighborRegressor":
                    try:
                        predictions = model.predict(X_test)
                        kneigh = RegressionMetrics(y_test, predictions, X_test)
                        kneigh.display_metrics()
                        kneigh.run_all_visualizations()
                    except:
                        st.write("")
            if st.sidebar.checkbox("Optimisation"):

                if model_name == "Decison Tree Regressor":

                    try:
                        max_depth = [None, 10, 20, 30]
                        min_samples_split = [2, 5, 10]
                        min_samples_leaf = [1, 2, 5]

                        depthness = st.sidebar.multiselect(
                            "select the max_depth", max_depth
                        )
                        splitness = st.sidebar.multiselect(
                            "select the min sample split", min_samples_split
                        )
                        leaf = st.sidebar.multiselect(
                            "select the min sample leaf", min_samples_leaf
                        )

                        st.write(leaf)
                        st.write(depthness)
                        st.write(splitness)
                        parameter = {
                            "max_depth": depthness,
                            "min_samples_split": splitness,
                            "min_samples_leaf": leaf,
                        }

                        model = RandomizedSearchCV(
                            DecisionTreeRegressor(),
                            param_distributions=parameter,
                            cv=5,
                            n_jobs=-1,
                        )

                    except:
                        st.write(" ")
                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        st.write(model.best_params_)
                        predictions = model.predict(X_test)
                        Decison = RegressionMetrics(y_test, predictions, X_test)
                        Decison.display_metrics()
                        Decison.run_all_visualizations()

                if model_name == "KNeighborRegressor":
                    try:
                        n_neighbors = [3, 5, 10, 15]
                        weights = ["uniform", "distance"]
                        algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
                        metric = ["euclidean", "manhattan"]

                        neighbor = st.sidebar.multiselect(
                            "selct the n_eighbors", n_neighbors
                        )
                        weight = st.sidebar.multiselect("select the weights", weights)
                        algorith = st.sidebar.multiselect("select the algorithm", algorithm)
                        met = st.sidebar.multiselect("select the metrics", metric)

                        st.write(neighbor)
                        st.write(weight)
                        st.write(algorith)
                        st.write(met)
                        parameter = {
                            "n_neighbors": neighbor,
                            "weights": weight,
                            "algorithm": algorith,
                            "metric": met,
                        }

                        model = RandomizedSearchCV(
                            KNeighborsRegressor(),
                            param_distributions=parameter,
                            cv=5,
                            n_jobs=-1,
                        )

                    except:
                        st.write(" ")
                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        predictions = model.predict(X_test)
                        Kneighou = RegressionMetrics(y_test, predictions, X_test)
                        Kneighou.display_metrics()
                        Kneighou.run_all_visualizations()

                if model_name == "Gradient Boosting Regressor":
                    try:
                        n_estimators = [50, 100, 200]
                        learning_rate = [0.01, 0.1, 0.2]
                        max_depth = [3, 5, 7]
                        subsample = [0.8, 1.0]

                        estima = st.sidebar.multiselect(
                            "selct the n_estimators", n_estimators
                        )
                        rate = st.sidebar.multiselect(
                            "select the learning rate", learning_rate
                        )
                        depth = st.sidebar.multiselect("select the ax_depth", max_depth)
                        samples = st.sidebar.multiselect("select the sub sample", subsample)

                        st.write(estima)
                        st.write(rate)
                        st.write(depth)
                        st.write(samples)
                        parameter = {
                            "n_estimators": estima,
                            "learning_rate": rate,
                            "max_depth": depth,
                            "subsample": samples,
                        }

                        model = RandomizedSearchCV(
                            GradientBoostingRegressor(),
                            param_distributions=parameter,
                            cv=5,
                            n_jobs=-1,
                        )

                    except:
                        st.write("")
                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        st.write(model.best_params_)

                        Gradient = RegressionMetrics(y_test, predictions, X_test)
                        Gradient.display_metrics()
                        Gradient.run_all_visualizations()

                if model_name == "Random Forest Regressor":
                    try:
                        n_estimators = [100, 200, 500]
                        max_depth = [None, 10, 20, 30]
                        min_samples_split = [2, 5, 10]

                        estimator = st.sidebar.multiselect(
                            "selct the n_estimators", n_estimators
                        )
                        depth = st.sidebar.multiselect("select the max_depth", max_depth)

                        sample_split = st.sidebar.multiselect(
                            "select the min sample split", min_samples_split
                        )

                        st.write(estimator)
                        st.write(depth)
                        st.write(sample_split)
                        parameter = {
                            "n_estimators": estimator,
                            "max_depth": depth,
                            "min_samples_split": sample_split,
                        }

                        model = RandomizedSearchCV(
                            RandomForestRegressor(),
                            param_distributions=parameter,
                            cv=5,
                            n_jobs=-1,
                        )

                    except:
                        st.write()
                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        predictions = model.predict(X_test)
                        Randomf = RegressionMetrics(y_test, predictions, X_test)
                        Randomf.display_metrics()
                        Randomf.run_all_visualizations()

                if model_name == "Adaboost Regressor":
                    try:
                        n_estimators = [100, 200, 500]
                        learning_rate = [0.01, 0.1, 1.0]
                        loss = ["linear", "square", "exponential"]

                        estima = st.sidebar.multiselect(
                            "selct the n_estimators", n_estimators
                        )
                        rate = st.sidebar.multiselect(
                            "select the learning rate", learning_rate
                        )
                        lo = st.sidebar.multiselect("select the loss", loss)

                        st.write(estima)
                        st.write(rate)
                        st.write(lo)
                        parameter = {
                            "n_estimators": estima,
                            "learning_rate": rate,
                            "loss": lo,
                        }

                        model = RandomizedSearchCV(
                            AdaBoostRegressor(),
                            param_distributions=parameter,
                            cv=5,
                            n_jobs=-1,
                        )

                    except:
                        st.write("")
                    if st.sidebar.button("run optimised"):
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        st.write(model.best_params_)
                        predictions = model.predict(X_test)
                        Adabost = RegressionMetrics(y_test, predictions, X_test)
                        Adabost.display_metrics()
                        Adabost.run_all_visualizations()

                if model_name == "Support Vector Regressor":
                    try:
                        c = [0.1, 1.0, 10.0]
                        epsilon = [0.01, 0.1, 0.5]
                        kernel = ["linear", "rbf"]

                        cr = st.sidebar.multiselect("selct the C", c)
                        epsi = st.sidebar.multiselect("select the epsilon", epsilon)
                        ker = st.sidebar.multiselect("select the kernel", kernel)

                        st.write(cr)
                        st.write(epsi)
                        st.write(ker)
                        parameter = {"C": cr, "epsilon": epsi, "kernel": ker}

                        model = RandomizedSearchCV(
                            SVR(), param_distributions=parameter, cv=5, n_jobs=-1
                        )
                    except:
                        st.write()
                    if st.sidebar.button("run"):
                        model.fit(X_train, y_train)
                        model.predict(X_test)
                        st.write(model.best_params_)
                        predictions = model.predict(X_test)
                        support = RegressionMetrics(y_test, predictions, X_test)
                        support.display_metrics()
                        support.run_all_visualizations()
    