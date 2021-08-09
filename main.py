from utils.preprocessing import load_data, balance_labels, get_data_target
from utils.processing import _clf_pipeline
from utils.postprocessing import shap_decision_plot, get_confusion_mtx, plot_3D, get_corr, plot_testing_set, _pickle_SHAP, classifiers
from utils.metric_style import metric_row, metric_row_custom, metric
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

df = load_data()

#_clf_pipeline('RandomForest', df, balance=True, n_repeat=1)
#_clf_pipeline('KNN', df, balance=True, n_repeat=1)

_features = ('Avg_Open_To_Buy',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Total_Relationship_Count',
        'Total_Revolving_Bal',
        'Credit_Limit')

with st.sidebar:
    st.set_page_config(layout="wide")
    st.sidebar.title('Data Analysis Strategy')

    selection_strategy = st.sidebar.selectbox('Please select an algorithm:',
                                            ('RandomForest',
                                             'KNN'))
    st.markdown("""---""")

    selection_x = st.sidebar.selectbox('Please select a feature for the X axis:',
                                              _features)

    selection_y = st.sidebar.selectbox('Please select a feature for the Y axis:',
                                       _features[1:]+_features[0:1])

    selection_z = st.sidebar.selectbox('Please select a feature for the Z axis:',
                                       _features[2:]+_features[0:2])

    st.markdown("""---""")

    selection_seed = st.slider("Seed", 0, 100, 42)


mark = {True:"CORRECT", False:"WRONG"}
features = selection_x, selection_y, selection_z

decision_plot, match, index, true_y, predict_y = shap_decision_plot(selection_strategy, selection_seed)
confusion_matrix = get_confusion_mtx(selection_strategy)
scatter_3D_balanced = plot_3D(balance_labels(df), features)
scatter_3D_dataset = plot_3D(df, features)
scatter_3D_testing_set = plot_testing_set(selection_strategy, index, features)

corr_matrix = get_corr(df)

#_pickle_SHAP('RandomForest')
#_pickle_SHAP('KNN')


metric('Corentin Chanet - becode.org 2021', "Explainable AI for Churn Prediction")

expander_seleccion_data = st.expander("Exploratory Data Analysis", expanded=False)
with expander_seleccion_data:

    data, target = get_data_target(balance_labels(df.copy()))

    margin_0_left, original_table, margin_0_right = st.columns((0.1, 1, 0.1))

    with original_table:
        st.write(f"Original Dataset")
        st.dataframe(df)

    margin_KPI_left, KPI_1, KPI_2, KPI_3, KPI_4, margin_KPI_right = st.columns((1, 1, 1, 1, 1, 1))

    with KPI_1:
        metric("NaN Values", 0)

    with KPI_2:
        metric("Features", df.shape[1])

    with KPI_3:
        metric("Attrited Customers", df['Attrition_Flag'].value_counts()[1])

    with KPI_4:
        metric("Remaining Customers", df['Attrition_Flag'].value_counts()[0])

    margin_1_left, clean_table, margin_1_right = st.columns((0.1, 1, 0.1))

    with clean_table:
        st.write(f"Dataset after cleaning, One Hot Encoding and rebalancing through random downsampling")
        st.dataframe(data)

    margin_KPIa_left, KPI_a, KPI_b, KPI_c, KPI_d, margin_KPId_right = st.columns((1, 1, 1, 1, 1, 1))

    with KPI_a:
        metric("NaN Values", 0)

    with KPI_b:
        metric("Features", data.shape[1])

    with KPI_c:
        metric("Attrited Customers", target.value_counts()[0])

    with KPI_d:
        metric("Remaining Customers", len(target) - target.value_counts()[0])

    row_0_left, row_0_margin, row_0_right = st.columns((1, 0.1, 1))

    with row_0_left:
        st.subheader(f'Correlation Matrix')
        fig, ax = plt.subplots()
        ax = sns.heatmap(corr_matrix)
        st.pyplot(fig)

    with row_0_right:
        st.subheader(f'3D Plot of entire unbalanced dataset')
        st.plotly_chart(scatter_3D_dataset, use_container_width=True)

expander_seleccion_data = st.expander("Classification", expanded=False)
with expander_seleccion_data:

    row_1_left, row_1_margin = st.columns((1, 0.1))

    with row_1_left:
        row_1_left.title(f'{selection_strategy}')
        row_1_left.write('\n')

    if selection_strategy == "RandomForest":
        metric_row(
            {
                "Precision": 0.93,
                "Recall": 0.94,
                "f1-score": 0.94
            }
        )

    elif selection_strategy == "KNN":
        metric_row(
            {
                "Precision": 0.77,
                "Recall": 0.77,
                "f1-score": 0.77
            }
        )

    metric_row_custom(
        {
            "True Positive": (confusion_matrix[0][0], 'green', 'green', '30px'),
            "True Negative": (confusion_matrix[1][1], 'green', 'green', '30px'),
            "False Positive": (confusion_matrix[1][0], 'red', 'red', '30px'),
            "False Negative": (confusion_matrix[0][1], 'red', 'red', '30px')
        }
    )

    row_2_left, row_2_margin, row_2_right = st.columns((1, 0.1, 1))

    with row_2_left:
        st.subheader(f'SHAP random single-sample decision plot \n {mark[match]} prediction (TRUE : {true_y})')
        st.write(decision_plot)

    with row_2_right:
        st.subheader(f'3D Plot of X_test (balanced)')
        st.plotly_chart(scatter_3D_testing_set, use_container_width=True)

    with row_2_left:
        st.subheader(f"SHAP summary plot \n Training Set's Features Importances")
        st.image(f'./assets/{selection_strategy}_shap_summary.png')

    with row_2_right:
        st.subheader(f'3D Plot of entire balanced dataset')
        st.plotly_chart(scatter_3D_balanced, use_container_width=True)

    with row_2_left:
        st.subheader(f'Correlation Matrix')
        fig, ax = plt.subplots()
        ax = sns.heatmap(corr_matrix)
        st.pyplot(fig)

    with row_2_right:
        st.subheader(f'3D Plot of entire unbalanced dataset')
        st.plotly_chart(scatter_3D_dataset, use_container_width=True)

# plt_corr = plot_corr(df)

# plt_3D = plot_3D(df)







#ari_kmeans, ari_dbscan = cluster(df)

