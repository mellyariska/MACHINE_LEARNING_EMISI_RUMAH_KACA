import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="EDGAR GHG Forecasting Dashboard",
    layout="wide"
)

st.title("🌍 Dashboard Forecasting Emisi Gas Rumah Kaca Global")
st.write("Berbasis Machine Learning menggunakan Data EDGAR Periode 1970–2024")

uploaded_file = st.sidebar.file_uploader(
    "Upload file EDGAR Excel",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Silakan upload file EDGAR Excel terlebih dahulu.")
    st.stop()

@st.cache_data
def load_data(file):
    totals = pd.read_excel(file, sheet_name="GHG_totals_by_country")
    percapita = pd.read_excel(file, sheet_name="GHG_per_capita_by_country")
    pergdp = pd.read_excel(file, sheet_name="GHG_per_GDP_by_country")
    sector = pd.read_excel(file, sheet_name="GHG_by_sector_and_country")
    return totals, percapita, pergdp, sector

totals, percapita, pergdp, sector = load_data(uploaded_file)

years = sorted([c for c in totals.columns if isinstance(c, int)])

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Home",
        "Data Explorer",
        "Tren Emisi",
        "Forecasting",
        "PCA & Clustering",
        "Anomaly Detection",
        "Feature Importance",
        "Sektor Emisi"
    ]
)

# ================= HOME =================
if menu == "Home":
    st.subheader("Tentang Dashboard")
    st.write("""
    Dashboard ini digunakan untuk menganalisis dan memprediksi emisi gas rumah kaca global
    berdasarkan dataset EDGAR periode 1970–2024.
    
    Fitur utama:
    - Visualisasi tren emisi
    - Forecasting emisi masa depan
    - PCA dan clustering negara
    - Deteksi anomali emisi
    - Feature importance
    - Analisis sektor emisi
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Negara", totals["Country"].nunique())
    col2.metric("Periode Data", f"{min(years)}–{max(years)}")
    col3.metric("Jumlah Tahun", len(years))

# ================= DATA EXPLORER =================
elif menu == "Data Explorer":
    st.subheader("Data EDGAR")
    st.dataframe(totals)

    st.download_button(
        "Download Data Total Emisi",
        totals.to_csv(index=False),
        file_name="data_total_emisi_edgar.csv"
    )

# ================= TREN EMISI =================
elif menu == "Tren Emisi":
    st.subheader("Tren Emisi Gas Rumah Kaca")

    countries = st.multiselect(
        "Pilih negara",
        totals["Country"].dropna().unique(),
        default=["Indonesia"]
    )

    plot_data = []

    for country in countries:
        row = totals[totals["Country"] == country]
        if not row.empty:
            values = row[years].iloc[0].values
            temp = pd.DataFrame({
                "Year": years,
                "Emission": values,
                "Country": country
            })
            plot_data.append(temp)

    if plot_data:
        plot_df = pd.concat(plot_data)

        fig = px.line(
            plot_df,
            x="Year",
            y="Emission",
            color="Country",
            markers=True,
            title="Tren Emisi GHG 1970–2024"
        )
        st.plotly_chart(fig, use_container_width=True)

# ================= FORECASTING =================
elif menu == "Forecasting":
    st.subheader("Forecasting Emisi Menggunakan Random Forest")

    country = st.selectbox(
        "Pilih negara untuk prediksi",
        totals["Country"].dropna().unique(),
        index=list(totals["Country"]).index("Indonesia") if "Indonesia" in list(totals["Country"]) else 0
    )

    forecast_until = st.slider(
        "Prediksi sampai tahun",
        min_value=2025,
        max_value=2050,
        value=2035
    )

    row = totals[totals["Country"] == country].iloc[0]

    X = np.array(years).reshape(-1, 1)
    y = row[years].astype(float).values

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X, y)

    future_years = np.arange(2025, forecast_until + 1).reshape(-1, 1)
    future_pred = model.predict(future_years)

    historical_df = pd.DataFrame({
        "Year": years,
        "Emission": y,
        "Type": "Historical"
    })

    forecast_df = pd.DataFrame({
        "Year": future_years.flatten(),
        "Emission": future_pred,
        "Type": "Forecast"
    })

    combined = pd.concat([historical_df, forecast_df])

    fig = px.line(
        combined,
        x="Year",
        y="Emission",
        color="Type",
        markers=True,
        title=f"Forecasting Emisi GHG {country}"
    )

    st.plotly_chart(fig, use_container_width=True)

    pred_train = model.predict(X)

    mae = mean_absolute_error(y, pred_train)
    rmse = np.sqrt(mean_squared_error(y, pred_train))
    r2 = r2_score(y, pred_train)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", round(mae, 4))
    col2.metric("RMSE", round(rmse, 4))
    col3.metric("R²", round(r2, 4))

    st.subheader("Tabel Hasil Forecasting")
    st.dataframe(forecast_df)

    st.download_button(
        "Download Hasil Forecasting",
        forecast_df.to_csv(index=False),
        file_name=f"forecasting_{country}.csv"
    )

# ================= PCA & CLUSTERING =================
elif menu == "PCA & Clustering":
    st.subheader("PCA dan Clustering Negara Berdasarkan Pola Emisi")

    n_cluster = st.slider("Jumlah Cluster", 2, 8, 4)

    data_ml = totals[["Country"] + years].dropna()
    X = data_ml[years].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    cluster = kmeans.fit_predict(X_scaled)

    pca_df = pd.DataFrame({
        "Country": data_ml["Country"],
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Cluster": cluster.astype(str)
    })

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_name="Country",
        title="PCA dan K-Means Clustering Negara"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pca_df)

# ================= ANOMALY DETECTION =================
elif menu == "Anomaly Detection":
    st.subheader("Deteksi Anomali Emisi Negara")

    contamination = st.slider(
        "Tingkat anomali",
        min_value=0.01,
        max_value=0.20,
        value=0.05
    )

    data_ml = totals[["Country"] + years].dropna()
    X = data_ml[years].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    iso = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    anomaly = iso.fit_predict(X_scaled)

    anomaly_df = pd.DataFrame({
        "Country": data_ml["Country"],
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Anomaly": anomaly
    })

    anomaly_df["Status"] = anomaly_df["Anomaly"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    fig = px.scatter(
        anomaly_df,
        x="PC1",
        y="PC2",
        color="Status",
        hover_name="Country",
        title="Anomaly Detection Menggunakan Isolation Forest"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Negara Anomali")
    st.dataframe(anomaly_df[anomaly_df["Status"] == "Anomaly"])

# ================= FEATURE IMPORTANCE =================
elif menu == "Feature Importance":
    st.subheader("Feature Importance untuk Prediksi Emisi 2024")

    feature_years = years[:-1]
    target_year = years[-1]

    X = totals[feature_years].fillna(0)
    y = totals[target_year].fillna(0)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X, y)

    importance_df = pd.DataFrame({
        "Year": feature_years,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        importance_df.head(15),
        x="Year",
        y="Importance",
        title="15 Tahun Paling Berpengaruh dalam Prediksi Emisi 2024"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(importance_df)

# ================= SEKTOR EMISI =================
elif menu == "Sektor Emisi":
    st.subheader("Analisis Emisi Berdasarkan Sektor")

    country = st.selectbox(
        "Pilih negara",
        sector["Country"].dropna().unique(),
        index=list(sector["Country"]).index("Indonesia") if "Indonesia" in list(sector["Country"]) else 0
    )

    latest_year = max(years)

    sector_country = sector[sector["Country"] == country]

    sector_sum = (
        sector_country
        .groupby("Sector")[latest_year]
        .sum()
        .reset_index()
        .sort_values(latest_year, ascending=False)
    )

    fig = px.bar(
        sector_sum,
        x="Sector",
        y=latest_year,
        title=f"Emisi Berdasarkan Sektor di {country} Tahun {latest_year}"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig_pie = px.pie(
        sector_sum,
        names="Sector",
        values=latest_year,
        title=f"Komposisi Emisi Sektor di {country}"
    )

    st.plotly_chart(fig_pie, use_container_width=True)