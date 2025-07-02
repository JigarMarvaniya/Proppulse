import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules

from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

import io

st.set_page_config(page_title="Dubai Real Estate ML Analytics", layout="wide")
st.title("🏙️ Dubai Real Estate Consumer Analytics Dashboard")

####### Utility Functions ########

@st.cache_data
def load_data(url=None, uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    elif url:
        try:
            return pd.read_csv(url)
        except Exception:
            return None
    return None

def get_data():
    csv_url = st.secrets["data_url"] if "data_url" in st.secrets else "https://raw.githubusercontent.com/jpmarvaniya/dubai-re-data/main/data/dubai_realestate_survey_synthetic.csv"
    uploaded = st.sidebar.file_uploader("Or upload your CSV", type=['csv'])
    df = load_data(url=csv_url, uploaded=uploaded)
    if df is None:
        st.warning("Could not load data from GitHub. Please upload manually.")
    return df

def label_encode_df(df, exclude=[]):
    df_enc = df.copy()
    le_dict = {}
    for c in df_enc.columns:
        if df_enc[c].dtype == 'object' and c not in exclude:
            le = LabelEncoder()
            df_enc[c] = le.fit_transform(df_enc[c].astype(str))
            le_dict[c] = le
    return df_enc, le_dict

def multilabel_encode_col(df, col):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df[col].str.split(','))
    out = pd.DataFrame(X, columns=[f"{col}__{c}" for c in mlb.classes_])
    return out

def download_button(data, filename, label="Download"):
    csv = data.to_csv(index=False)
    st.download_button(label=label, data=csv, file_name=filename, mime='text/csv')

##################### SIDEBAR #######################

tabs = [
    "Data Visualisation",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression"
]
tab = st.sidebar.radio("Select Dashboard Tab:", tabs)

st.sidebar.markdown("---")
st.sidebar.write("Default data will be loaded from GitHub. You can also upload your own CSV.")

##################### LOAD DATA #####################

df = get_data()
if df is None:
    st.stop()
else:
    st.success(f"Loaded data shape: {df.shape}")

##################### DATA VISUALISATION TAB #####################

if tab == "Data Visualisation":
    st.header("🔍 Data Visualisation & Insights")
    df_viz = df.copy()
    st.write("#### Preview of the Data")
    st.dataframe(df_viz.head())

    st.markdown("### 1. Distribution of Willingness to Invest")
    fig, ax = plt.subplots()
    df_viz["Willingness_to_Invest"].value_counts().plot(kind='bar', ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Willingness to Invest")
    st.pyplot(fig)
    st.info("Shows which segments are more/less likely to invest.")

    st.markdown("### 2. Age Distribution & Outliers")
    fig, ax = plt.subplots()
    sns.boxplot(x=df_viz["Age"], ax=ax)
    st.pyplot(fig)
    st.info("Boxplot of age group distribution, highlighting skew and outliers.")

    st.markdown("### 3. Income vs Budget Scatterplot")
    fig, ax = plt.subplots()
    sns.scatterplot(x="Income_Numeric", y="Budget_Numeric", hue="Willingness_to_Invest", data=df_viz, alpha=0.7, ax=ax)
    st.pyplot(fig)
    st.info("Higher earners typically have higher real estate budgets, but not always.")

    st.markdown("### 4. Correlation Heatmap (Numerics Only)")
    numerics = df_viz.select_dtypes(include=np.number)
    fig, ax = plt.subplots()
    sns.heatmap(numerics.corr(), annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)
    st.info("Shows which numeric variables are strongly related.")

    st.markdown("### 5. Top Preferred Areas")
    all_areas = pd.Series([a.strip() for x in df_viz['Preferred_Areas'] for a in str(x).split(",")])
    fig, ax = plt.subplots()
    all_areas.value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)
    st.info("Popular areas in Dubai according to consumers.")

    st.markdown("### 6. Persona vs Budget Bin")
    cross = pd.crosstab(df_viz['Persona'], df_viz['Budget_Bin'])
    st.dataframe(cross.style.background_gradient(cmap='YlOrRd'))
    st.info("See how different personas segment by their intended budget.")

    st.markdown("### 7. Distribution of Challenges")
    all_challenges = pd.Series([a.strip() for x in df_viz['Challenges'] for a in str(x).split(",")])
    fig, ax = plt.subplots()
    all_challenges.value_counts().plot(kind='bar', color='coral', ax=ax)
    st.pyplot(fig)
    st.info("Pain points faced most commonly.")

    st.markdown("### 8. Importance of Analytics by Persona")
    fig, ax = plt.subplots()
    sns.countplot(x="Persona", hue="Analytics_Importance", data=df_viz, ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)
    st.info("Which personas value data/analytics in their decisions.")

    st.markdown("### 9. Monthly Income Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_viz['Income_Numeric'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    st.info("Some outliers and a right-skew, typical for income.")

    st.markdown("### 10. Download Current Data")
    download_button(df_viz, "realestate_dashboard_data.csv", "Download CSV")

##################### CLASSIFICATION TAB #####################

elif tab == "Classification":
    st.header("🤖 ML Classification")
    df_clf = df.copy()
    target = st.selectbox("Select classification label", ["Willingness_to_Invest", "Waitlist_Interest"])
    features = st.multiselect("Select features for classification", [c for c in df_clf.columns if c not in [target, "Income_Numeric", "Budget_Numeric"]], default=[
        "Age", "Nationality", "Employment_Status", "Persona", "Income_Bin", "Property_Type", "Purpose",
        "Analytics_Importance", "Research_Duration", "Online_Platform_Usage", "Influencer"
    ])

    # Handle encoding
    df_X = df_clf[features].copy()
    df_y = df_clf[target].copy()
    # For multi-select columns: take the first value only for simplicity
    for col in df_X.columns:
        if df_X[col].dtype == 'object' and df_X[col].str.contains(",").any():
            df_X[col] = df_X[col].astype(str).str.split(",").str[0]
    df_X, le_dict = label_encode_df(df_X)
    y_enc = LabelEncoder().fit_transform(df_y)

    X_train, X_test, y_train, y_test = train_test_split(df_X, y_enc, test_size=0.2, random_state=42)
    classifiers = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GradientBoosted": GradientBoostingClassifier(random_state=42)
    }
    metrics = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score, "F1-score": f1_score}

    results = {}
    roc_info = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pred_tr = clf.predict(X_train)
        pred_te = clf.predict(X_test)
        results[name] = {
            "Train Acc": accuracy_score(y_train, pred_tr),
            "Test Acc": accuracy_score(y_test, pred_te),
            "Test Prec": precision_score(y_test, pred_te, average="weighted"),
            "Test Recall": recall_score(y_test, pred_te, average="weighted"),
            "Test F1": f1_score(y_test, pred_te, average="weighted")
        }
        # ROC Curve info
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob[:,1], pos_label=1) if y_prob.shape[1]>1 else (None,None,None)
            roc_auc = auc(fpr, tpr) if fpr is not None else None
            roc_info[name] = (fpr, tpr, roc_auc)
    st.markdown("### Model Performance Table")
    st.dataframe(pd.DataFrame(results).T.style.background_gradient(cmap='PuBu'))

    st.markdown("### Confusion Matrix (Select Model)")
    model_sel = st.selectbox("Select model for confusion matrix", list(classifiers.keys()))
    clf = classifiers[model_sel]
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)
    st.info("Confusion matrix with axes labelled by encoded class values.")

    st.markdown("### ROC Curve: All Models")
    fig, ax = plt.subplots()
    for name, (fpr, tpr, roc_auc) in roc_info.items():
        if fpr is not None and tpr is not None and roc_auc is not None:
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0,1], [0,1], '--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Upload New Data to Predict Label")
    new_data = st.file_uploader("Upload data for prediction (must match feature columns, no label)", type=['csv'], key="pred_upload")
    if new_data:
        df_new = pd.read_csv(new_data)
        for col in df_X.columns:
            if col in df_new.columns:
                if df_new[col].dtype == 'object' and df_new[col].str.contains(",").any():
                    df_new[col] = df_new[col].astype(str).str.split(",").str[0]
        df_new_enc = df_new.copy()
        for c in df_X.columns:
            if c in le_dict:
                le = le_dict[c]
                df_new_enc[c] = le.transform(df_new_enc[c].astype(str).fillna(le.classes_[0]).values)
        preds = classifiers[model_sel].predict(df_new_enc[df_X.columns])
        st.write("Predicted labels (encoded):")
        st.write(preds)
        # Download
        df_pred_out = df_new.copy()
        df_pred_out['Predicted_Label'] = preds
        download_button(df_pred_out, "classified_results.csv", "Download Predicted Results")

##################### CLUSTERING TAB #####################

elif tab == "Clustering":
    st.header("📈 Clustering (KMeans)")
    st.write("Apply KMeans clustering to segment consumers and define personas.")

    # Use relevant features
    features = st.multiselect("Features to cluster on:", [
        "Income_Numeric", "Budget_Numeric", "Age", "Persona", "Employment_Status", "Analytics_Comfort"
    ], default=["Income_Numeric", "Budget_Numeric", "Analytics_Comfort"])
    df_kmeans = df.copy()
    # Encode categorical
    for col in features:
        if df_kmeans[col].dtype == 'object':
            df_kmeans[col] = LabelEncoder().fit_transform(df_kmeans[col].astype(str))
    scaler = StandardScaler()
    X = scaler.fit_transform(df_kmeans[features])
    # Elbow chart
    st.markdown("### Elbow Chart (Optimal k)")
    max_k = 10
    sse = []
    K_range = list(range(2, max_k+1))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        sse.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K_range, sse, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("SSE (Inertia)")
    st.pyplot(fig)
    st.info("Elbow point helps choose the right k for segmentation.")

    # Clustering
    num_clusters = st.slider("Select number of clusters", 2, max_k, 4)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_kmeans['Cluster'] = kmeans.fit_predict(X)
    # Persona summary
    st.markdown("### Cluster Persona Summary")
    summary = df_kmeans.groupby('Cluster')[['Income_Numeric', 'Budget_Numeric', 'Analytics_Comfort']].mean().round(1)
    st.dataframe(summary)
    # Download
    download_button(df_kmeans, "data_with_clusters.csv", "Download Data with Cluster Labels")

##################### ASSOCIATION RULES TAB #####################

elif tab == "Association Rule Mining":
    st.header("🧠 Association Rule Mining (Apriori)")
    st.write("Explore associations between preferences, challenges, and more.")

    # Select columns (only those suitable: multi-selects)
    ar_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].str.contains(',').any()]
    col1 = st.selectbox("Select first column (multi-select)", ar_cols, index=0)
    col2 = st.selectbox("Select second column (multi-select)", ar_cols, index=1 if len(ar_cols)>1 else 0)
    # MultiLabelBinarizer
    mlb1 = multilabel_encode_col(df, col1)
    mlb2 = multilabel_encode_col(df, col2)
    ar_df = pd.concat([mlb1, mlb2], axis=1)
    min_sup = st.slider("Minimum Support", 0.01, 0.3, 0.07)
    min_conf = st.slider("Minimum Confidence", 0.1, 0.9, 0.4)

    freq_items = apriori(ar_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
    st.info("Shows top-10 rules by confidence for selected columns.")

##################### REGRESSION TAB #####################

elif tab == "Regression":
    st.header("📊 Regression Analysis")
    st.write("Apply Ridge, Lasso, and Decision Tree regression for deeper quantitative insights.")

    # Select regression target and features
    reg_target = st.selectbox("Target variable", ["Income_Numeric", "Budget_Numeric"])
    reg_features = st.multiselect("Features", [
        "Analytics_Comfort", "Cluster" if "Cluster" in df.columns else None, "Age", "Persona", "Employment_Status", "Analytics_Importance"
    ], default=["Analytics_Comfort", "Age"])
    df_reg = df.copy()
    # Encode as needed
    for col in reg_features:
        if col is not None and df_reg[col].dtype == 'object':
            df_reg[col] = LabelEncoder().fit_transform(df_reg[col].astype(str))
    X_reg = df_reg[reg_features]
    y_reg = df_reg[reg_target]

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    reg_results = {}
    st.markdown("### Regression Performance Table")
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        pred = model.predict(X_test)
        mse = np.mean((y_test - pred) ** 2)
        reg_results[name] = {"Test R2": score, "Test MSE": mse}
    st.dataframe(pd.DataFrame(reg_results).T.style.background_gradient(cmap='BuGn'))

    # Feature importance for DT
    st.markdown("### Decision Tree Feature Importances")
    dt = models["Decision Tree"]
    importances = dt.feature_importances_
    feat_imp = pd.Series(importances, index=reg_features)
    fig, ax = plt.subplots()
    feat_imp.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # Actual vs Predicted
    st.markdown("### Actual vs Predicted Plot (All Models)")
    fig, ax = plt.subplots()
    for name, model in models.items():
        y_pred = model.predict(X_test)
        ax.scatter(y_test, y_pred, label=name, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='black')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    st.pyplot(fig)

    st.info("These charts/tables help understand which features most influence predictions, and the accuracy of different regression models.")

st.markdown("---")
st.caption("Developed by your AI data architect — Streamlit ML dashboard for Dubai Real Estate. Ready for deployment!")

