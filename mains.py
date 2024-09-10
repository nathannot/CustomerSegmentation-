import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.sparse import issparse
import plotly.graph_objs as go
import io

st.title("Customer Segmentation App")
st.subheader("Find Optimal Groups for your Marketing Needs")
st.markdown("Might take a while if the dataset is large - to speed up the process, take out unnecessary columns before uploading.")

# Upload the CSV
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    # Read uploaded CSV
    df = pd.read_csv(uploaded_file)

    # Select columns to exclude
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to exclude", columns)

    if selected_columns:
        df = df.drop(columns=selected_columns, axis=1)

    # Preprocess the data
    num_features = df.select_dtypes(include=[float, int])
    cat_features = df.select_dtypes(include=object)

    num = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    cat = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())

    preprocess = ColumnTransformer([
        ('num', num, make_column_selector(dtype_include=[float, int])),
        ('cat', cat, make_column_selector(dtype_include=object))
    ])

    X = preprocess.fit_transform(df)

    # Check if X is sparse and convert to dense if necessary
    X_dense = X.toarray() if issparse(X) else X

    # Find optimal number of clusters using silhouette score
    sil = []
    for k in range(3, 20):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_dense)
        sil.append(silhouette_score(X_dense, kmeans.labels_))

    best_k = 3 + np.argmax(sil)

    # Fit KMeans with the best number of clusters
    best_kmeans = KMeans(n_clusters=best_k, n_init='auto')
    clusters = best_kmeans.fit_predict(X_dense)

    model = df.copy()
    model['Customer Segment'] = clusters

    # Summary statistics for numerical and categorical features
    if not num_features.empty:
        a = model.groupby('Customer Segment')[num_features.columns].mean().round(2)
    else:
        a = pd.DataFrame()

    if not cat_features.empty:
        b = model.groupby('Customer Segment')[cat_features.columns].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.Series(dtype='object'))
    else:
        b = pd.DataFrame()

    if not a.empty and not b.empty:
        final = pd.concat([a, b], axis=1).reset_index()
    elif not a.empty:
        final = a.reset_index()
    elif not b.empty:
        final = b.reset_index()
    else:
        final = pd.DataFrame()

    st.subheader("Cluster Summary")
    st.dataframe(final)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    Xt = pca.fit_transform(X_dense)

    kmeans_pca = KMeans(n_clusters=best_k, n_init='auto')
    kmeans_pca.fit(Xt)
    clusters_pca = kmeans_pca.labels_
    cc = kmeans_pca.cluster_centers_

    # Create Plotly scatter plot
    scatter_data = []
    unique_clusters = np.unique(clusters_pca)
    for cluster in unique_clusters:
        scatter_data.append(go.Scatter(
            x=Xt[clusters_pca == cluster, 0],
            y=Xt[clusters_pca == cluster, 1],
            mode='markers',
            marker=dict(size=10),
            name=f'Cluster {cluster}'
        ))

    scatter_data.append(go.Scatter(
        x=cc[:, 0],
        y=cc[:, 1],
        mode='markers',
        marker=dict(color='red', size=25, symbol='x'),
        name='Cluster Centres'
    ))

    layout = go.Layout(
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2'),
        title='Customer Segments Visualized',
        showlegend=True
    )

    fig = go.Figure(data=scatter_data, layout=layout)
    st.plotly_chart(fig)

    # Download the final data as a CSV file
    csv = final.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="final_data.csv">Download final data as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
