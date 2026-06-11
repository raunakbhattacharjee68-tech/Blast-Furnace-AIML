import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Blast Furnace Dashboard",
    page_icon="🔥",
    layout="wide"
)

df = pd.read_csv("channeling_detection_results.csv")

st.sidebar.title("🔥 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Overview",
        "Feature Analysis",
        "Channeling Events"
    ]
)

total = len(df)
channeling = (df['Channeling_Flag'] == 1).sum()
normal = total - channeling
percentage = 100 * channeling / total

features = [
    'Temp_Imbalance',
    'Pressure_Imbalance',
    'Gas_Utilization',
    'TOTAL_K',
    'K_LM',
    'K_MU'
]

means = df.groupby('Channeling_Flag')[features].mean()

if page == "Overview":

    st.title(" Blast Furnace Channeling Detection Dashboard")

    st.success("""
    Key Findings

    • Temp Imbalance increased 4×
    • Pressure Imbalance increased 8×
    • Gas Utilization decreased
    • TOTAL_K increased
    • K_MU increased significantly
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total)
    col2.metric("Normal Records", normal)
    col3.metric("Channeling Events", channeling)
    col4.metric("Channeling %", f"{percentage:.2f}%")

    st.divider()

    # CLUSTER DISTRIBUTION
    st.subheader("Cluster Distribution")

    cluster_counts = df['Channeling_Flag'].value_counts()

    fig = px.bar(
        x=["Normal", "Channeling"],
        y=[normal, channeling],
        labels={"x": "Class", "y": "Count"},
        title="Normal vs Channeling Records"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # FEATURE COMPARISON
    st.subheader("Mean Feature Comparison")

    fig = px.bar(
        means.T,
        barmode='group',
        title="Feature Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Cluster Statistics")

    st.dataframe(
        means.round(3),
        use_container_width=True
    )

elif page == "Feature Analysis":

    st.title("Feature Analysis")

    feature = st.selectbox(
        "Select Feature",
        features
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Distribution")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.histplot(
            df[feature],
            kde=True,
            ax=ax
        )

        plt.title(feature)

        st.pyplot(fig)

    with col2:

        st.subheader("Normal vs Channeling")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.boxplot(
            x='Channeling_Flag',
            y=feature,
            data=df,
            ax=ax
        )

        ax.set_xticklabels(
            ["Normal", "Channeling"]
        )

        st.pyplot(fig)

elif page == "Channeling Events":

    st.title("Potential Channeling Events")

    anomalies = df[
        df['Channeling_Flag'] == 1
    ]

    st.metric(
        "Detected Channeling Events",
        len(anomalies)
    )

    st.subheader("Channeling Records")

    st.dataframe(
        anomalies,
        use_container_width=True
    )

    csv = anomalies.to_csv(index=False)

    st.download_button(
        label="Download Channeling Events",
        data=csv,
        file_name="channeling_events.csv",
        mime="text/csv"
    )