from utils.preprocessing import load_data, balance_labels, get_data_target
from utils.processing import _clf_pipeline
from utils.postprocessing import shap_decision_plot, get_confusion_mtx, plot_3D, get_corr, plot_testing_set, _pickle_SHAP, classifiers
from utils.metric_style import metric_row, metric_row_custom, metric_row_report, metric
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

data, target = get_data_target(load_data())
df = pd.concat([data, target], axis=1)

#_clf_pipeline('RandomForest', df, balance=True, n_repeat=1)
#_clf_pipeline('KNN', df, balance=True, n_repeat=1)
#_clf_pipeline('BalancedRandomForest', df, balance=False, n_repeat=1)
#_clf_pipeline('LightGBM', df, balance=False, n_repeat=1)

_features = sorted(df.columns.to_list())

with st.sidebar:
    st.set_page_config(layout="wide", initial_sidebar_state='collapsed')
    st.sidebar.header('Algorithms')
    selection_strategy = st.selectbox('Please select an algorithm:',
                                              ('RandomForest',
                                               'BalancedRandomForest',
                                               'KNN',
                                               'LightGBM'))
    st.markdown("""---""")
    if selection_strategy == 'KNN':
        st.markdown(f"""Neighbors-based classification is a type of instance-based learning 
        or non-generalizing learning: it does not attempt to construct a general internal model, 
        but simply stores instances of the training data. Classification is computed from a simple 
        majority vote of the nearest neighbors of each point: a query point is assigned the data 
        class which has the most representatives within the nearest neighbors of the point. <br></br>
        <a href=https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>From scikit-learn</a>""", unsafe_allow_html=True)

    elif selection_strategy == 'BalancedRandomForest':
        st.markdown(f"""A balanced random forest randomly under-samples each boostrap sample to 
        balance it. This class provides all functionality of the RandomForestClassifier. <br></br>
        <a href=https://imbalanced-learn.org/stable/ensemble.html#forest>From imbalance-learn</a>""", unsafe_allow_html=True)

    elif selection_strategy == 'RandomForest':
        st.markdown(f"""A random forest is a meta estimator that fits a number of decision tree 
        classifiers on various sub-samples of the dataset and uses averaging to improve the predictive 
        accuracy and control over-fitting. This means a diverse set of classifiers is created by introducing 
        randomness in the classifier construction. The prediction of the ensemble is given as the averaged 
        prediction of the individual classifiers. <br></br>
        <a href=https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>From scikit-learn</a>""", unsafe_allow_html=True)

    elif selection_strategy == 'LightGBM':
        st.markdown(f"""LightGBM is a gradient boosting framework that uses tree based learning algorithms. 
        Many boosting tools use pre-sort-based algorithms(e.g. default algorithm in xgboost) for decision tree learning. 
        It is a simple solution, but not easy to optimize. LightGBM uses histogram-based algorithms, which bucket continuous feature 
        (attribute) values into discrete bins. This speeds up training and reduces memory usage.<br></br>
        <a href=https://lightgbm.readthedocs.io/en/latest/Features.html>From LightGBM</a>""", unsafe_allow_html=True)

mark = {True:"CORRECT", False:"WRONG"}


corr_matrix = get_corr(df)

#_pickle_SHAP('RandomForest')
#_pickle_SHAP('KNN')
#_pickle_SHAP('BalancedRandomForest')
#_pickle_SHAP('LightGBM')

metric('Corentin Chanet - becode.org 2021', "Explainable AI for Churn Prediction")

st.markdown("Challenge statement and dataset available on <a href=https://www.kaggle.com/sakshigoyal7/credit-card-customers>Kaggle</a>", unsafe_allow_html=True)
st.markdown("Code available on <a href=https://github.com/CorentinChanet/churn-prediction>GitHub</a>", unsafe_allow_html=True)

expander_EDA = st.expander("Exploratory Data Analysis", expanded=False)

with expander_EDA:

    margin_0_left, original_table, margin_0_right = st.columns((0.1, 1, 0.1))

    with original_table:
        if st.checkbox("Show Dataframe", key='original_dataframe'):
            st.dataframe(df)
        st.subheader(f"Original Dataset")

    st.write("\n")
    metric_row_custom({
        "NaN Values": (0, '', 'black', 'black', 'black', '30px'),
        "Features": (data.shape[1], '', 'black', 'black', 'black', '30px'),
        "Attrited Customers": (target.value_counts()[1], '', 'black', 'red', 'black', '30px'),
        "Remaining Customers": (target.value_counts()[0], '', 'black', 'blue', 'black', '30px')
    })

    margin_1_left, clean_table, margin_1_right = st.columns((0.1, 1, 0.1))

    with clean_table:
        if st.checkbox("Show Dataframe", key='cleaned_dataframe'):
            st.dataframe(data)
        st.subheader(f"Features dataset after cleaning, One Hot Encoding and manual undersampling")

    data, target = get_data_target(balance_labels(df.copy()))

    st.write("\n")
    metric_row_custom({
        "NaN Values": (0, '', 'black', 'black', 'black', '30px'),
        "Features": (data.shape[1], '', 'black', 'black', 'black', '30px'),
        "Attrited Customers": (target.value_counts()[1], '', 'black', 'red', 'black', '30px'),
        "Remaining Customers": (target.value_counts()[0], '', 'black', 'blue', 'black', '30px')
    })

    row_eda_feature_x, row_eda_feature_y, row_eda_feature_z = st.columns((1, 1, 1))
    with row_eda_feature_x:
        selection_x = st.selectbox('Please select a feature for the X axis:',
                                   _features, index=len(_features) - 4, key='selection_x_eda')
    with row_eda_feature_y:
        selection_y = st.selectbox('Please select a feature for the Y axis:',
                                   _features, index=len(_features) - 2, key='selection_y_eda')
    with row_eda_feature_z:
        selection_z = st.selectbox('Please select a feature for the Z axis:',
                                   _features, index=len(_features) - 1, key='selection_z_eda')

    st.write('\n')
    features = selection_x, selection_y, selection_z
    confusion_matrix = get_confusion_mtx(selection_strategy)
    scatter_3D_balanced = plot_3D(balance_labels(df), features)
    scatter_3D_dataset = plot_3D(df, features)

    row_0_left, row_0_margin, row_0_right = st.columns((1, 0.1, 1))

    with row_0_left:
        st.subheader(f'3D Plot of manually balanced dataset')
        st.plotly_chart(scatter_3D_balanced, use_container_width=True)

    with row_0_right:
        st.subheader(f'3D Plot of entire unbalanced dataset')
        st.plotly_chart(scatter_3D_dataset, use_container_width=True)


expander_classification = st.expander("Classification", expanded=False)
with expander_classification:

    row_1_left, row_1_margin = st.columns((1, 0.1))

    with row_1_left:
        st.title(f'{selection_strategy}')
        st.write('\n')

    if selection_strategy == "RandomForest":
        st.markdown('Based on Manual Undersampling')

        metric_row_report(
            {
                "Precision": ('0.93', '0.94', 'black', 'blue', 'red', '40px'),
                "Recall": ('0.95', '0.94', 'black', 'blue', 'red', '40px'),
                "f1-score": ('0.94', '0.94', 'black', 'blue', 'red', '40px'),
            }
        )

        st.write("\n")
        st.write("\n")
        st.write("\n")


    elif selection_strategy == "KNN":
        st.markdown('Based on Manual Undersampling')
        metric_row_report(
            {
                "Precision": ('0.78', '0.77', 'black', 'blue', 'red', '40px'),
                "Recall": ('0.76', '0.78', 'black', 'blue', 'red', '40px'),
                "f1-score": ('0.77', '0.77', 'black', 'blue', 'red', '40px'),
            }
        )

        st.write("\n")
        st.write("\n")
        st.write("\n")

    elif selection_strategy == "BalancedRandomForest":
        st.markdown('Based on Bootstrap Sampling')
        metric_row_report(
            {
                "Precision": ('0.99', '0.72', 'black', 'blue', 'red', '40px'),
                "Recall": ('0.93', '0.96', 'black', 'blue', 'red', '40px'),
                "f1-score": ('0.96', '0.83', 'black', 'blue', 'red', '40px'),
            }
        )

        st.write("\n")
        st.write("\n")
        st.write("\n")

    elif selection_strategy == "LightGBM":
        st.write('Based on Relative Weights Scaling')
        metric_row_report(
            {
                "Precision": ('0.92', '0.99', 'black', 'red', 'blue', '40px'),
                "Recall": ('0.94', '0.98', 'black', 'red', 'blue', '40px'),
                "f1-score": ('0.93', '0.99', 'black', 'red', 'blue', '40px'),
            }
        )

        st.write("\n")
        st.write("\n")
        st.write("\n")

    metric_row_custom(
        {
            "True Positive": (confusion_matrix[0][0], '', 'black', 'green', 'black', '30px'),
            "True Negative": (confusion_matrix[1][1], '', 'black', 'green', 'black', '30px'),
            "False Positive": (confusion_matrix[1][0], '', 'black', 'black', 'black', '30px'),
            "False Negative": (confusion_matrix[0][1], '', 'black', 'black', 'black', '30px')
        }
    )
    if st.checkbox("Show model parameters"):
        advanced_KPI = {}
        for key, value in classifiers[selection_strategy]['gridsearch'].best_params_.items():
            advanced_KPI[key[7:]] = (value, '', 'black', 'black', 'black', '24px')
        metric_row_custom(
            advanced_KPI
        )

    if st.checkbox("Show SHAP diagnostics"):

        row_2_left, row_2_margin, row_2_right = st.columns((1, 0.1, 1))

        with row_2_left:
            st.subheader(f"SHAP summary plot \n Training Set's Features Importances")
            st.image(f'./assets/{selection_strategy}_shap_summary.png')

        with row_2_right:
            st.subheader(f"A brief explanation")
            explanation_text = """
            SHAP (SHapley Additive exPlanations) by Lundberg and Lee (2016) is a method to explain individual predictions. 
            SHAP is based on the game theoretically optimal Shapley Values.
            The SHAP authors proposed KernelSHAP, an alternative, kernel-based estimation approach for Shapley values 
            inspired by local surrogate models. They also offer TreeSHAP, an efficient estimation approach for tree-based models.
            (<a href="https://christophm.github.io/interpretable-ml-book/shap.html">Molnar 2021</a>)
            <br></br>
            This summary plot is designed to display an information-dense summary of how the top features in a dataset 
            impact the model’s output. Each instance the given explanation is represented by a single dot on each feature fow. 
            The x position of the dot is determined by the SHAP value of that feature, and dots “pile up” along each feature 
            row to show density. Color is used to display the original value of a feature.
            By default the features are ordered using the mean absolute value of the SHAP values for each feature. 
            This order however places more emphasis on broad average impact, and less on rare but high magnitude impacts.
            (<a href="https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html">SHAP Python Library</a>)
            <br></br>
            """
            st.markdown(f'<p style="font-family:IBM Plex Sans; color:black; font-size: 16px;">{explanation_text}</p>',
                        unsafe_allow_html=True)

        st.write('\n')
        row_class_feature_x, row_class_feature_y, row_class_feature_z = st.columns((1, 1, 1))
        with row_class_feature_x:
            selection_x = st.selectbox('Please select a feature for the X axis:',
                                       _features, index=len(_features) - 4, key='selection_x_clf')
        with row_class_feature_y:
            selection_y = st.selectbox('Please select a feature for the Y axis:',
                                       _features, index=len(_features) - 2, key='selection_y_clf')
        with row_class_feature_z:
            selection_z = st.selectbox('Please select a feature for the Z axis:',
                                       _features, index=len(_features) - 1, key='selection_z_clf')


        row_3_left, row_3_margin, row_3_right = st.columns((1, 0.1, 1))

        with row_3_left:
            st.subheader(f'3D Plot of the Testing Set')
            selection_seed = st.slider("Pick a sample", 0, 100, 42)
            decision_plot, match, index, true_y, predict_y = shap_decision_plot(selection_strategy, selection_seed)
            scatter_3D_testing_set = plot_testing_set(selection_strategy, index, features)
            st.plotly_chart(scatter_3D_testing_set, use_container_width=True)

        with row_3_right:
            st.subheader(f'SHAP random single-sample decision plot \n {mark[match]} prediction (TRUE : {true_y}) \n')
            st.write(decision_plot)
