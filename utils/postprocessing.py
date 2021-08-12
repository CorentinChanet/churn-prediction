import joblib
import shap
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st

classifiers = {
    "RandomForest": {},
    "BalancedRandomForest": {},
    "KNN": {},
    "LightGBM": {}
}

for name in classifiers.keys():
    classifiers[name]['gridsearch'] = joblib.load(f'./assets/{name}_gridsearch.pkl')
    classifiers[name]['score'] = joblib.load(f'./assets/{name}_score.pkl')
    classifiers[name]['confusion_matrix'] = joblib.load(f'./assets/{name}_confusion_matrix.pkl')
    classifiers[name]['classification_report'] = joblib.load(f'./assets/{name}_classification_report.pkl')
    classifiers[name]['model_features'] = joblib.load(f'./assets/{name}_model_features.pkl')
    classifiers[name]['train_test_predict_proba'] = joblib.load(f'./assets/{name}_train_test_predict_proba.pkl')


def _pickle_SHAP(name):

    X_train, X_test, y_train, y_test, y_predictions, y_proba = classifiers[name]['train_test_predict_proba']
    model = classifiers[name]['gridsearch'].best_estimator_.named_steps['model']

    if name in ["RandomForest", "BalancedRandomForest"]:
        explainer = shap.TreeExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_train, 3).data, l1_reg='num_features(10)')
        joblib.dump(explainer, './assets/KernelExplainerKNN.pkl')

    shap_values = explainer.shap_values(X_test)

    f = plt.figure()
    shap.summary_plot(shap_values[0], X_test, show=False)
    f.savefig(f"./assets/{name}_shap_summary.png", bbox_inches='tight', dpi=600)
    plt.close()


def shap_decision_plot(name, seed):
    X_train, X_test, y_train, y_test, y_predictions, y_proba = classifiers[name]['train_test_predict_proba']
    model = classifiers[name]['gridsearch'].best_estimator_.named_steps['model']

    if name in ["RandomForest", "BalancedRandomForest"]:
        explainer = shap.TreeExplainer(model, X_train, model_output="raw")
    else:
        explainer = joblib.load('./assets/KernelExplainerKNN.pkl')

    X_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    sample = X_test.sample(1, random_state=seed)
    index = sample.index[0]

    if np.all(model.predict(sample)[0] == y_test.loc[index]):
        match = True
    else:
        match = False

    shap_values = explainer.shap_values(sample)[0]
    expected_value = explainer.expected_value[0]

    shap.decision_plot(expected_value, shap_values, sample, new_base_value=0.5, show=False)
    fig_decision = plt.gcf()
    fig_decision.set_dpi(400)
    plt.close()

    return fig_decision, match, index, y_test.loc[index], model.predict(sample)[0]


@st.cache(max_entries=10, ttl=3600)
def get_confusion_mtx(name):

    confusion_matrix = classifiers[name]['confusion_matrix']
    return confusion_matrix

@st.cache(max_entries=10, ttl=3600)
def plot_testing_set(name, index, features):
    X_train, X_test, y_train, y_test, y_predictions, y_proba = classifiers[name]['train_test_predict_proba']

    X_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    fig = px.scatter_3d(X_test, x=features[0], y=features[1], z=features[2],
                        color=y_test, color_discrete_map={'Attrited Customer':'red',
                                                                    'Existing Customer':'blue'})

    fig.update_layout(scene={"aspectratio": {"x": 3, "y": 4, "z": 3},
                             "annotations": [dict(x=X_test[features[0]].loc[index],
                                                  y=X_test[features[1]].loc[index],
                                                  z=X_test[features[2]].loc[index],
                                                  text="Random Sample",
                                                  textangle=0,
                                                  ax=-75,
                                                  ay=-75,
                                                  font=dict(color="brown",
                                                            size=20),
                                                  arrowcolor="black",
                                                  arrowsize=3,
                                                  arrowwidth=2,
                                                  arrowhead=1)]
                             },
                      autosize=True,
                      margin=dict(l=20, r=20, b=50, t=20),
                      template='plotly_white',
                      showlegend=True
                      )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)')

    fig.update_traces(marker={'size': 3})

    return fig

@st.cache(max_entries=10, ttl=3600)
def plot_3D(df, features):


    fig = px.scatter_3d(df, x=features[0], y=features[1], z=features[2],
                        color='Attrition_Flag', color_discrete_map={'Attrited Customer':'red',
                                                                    'Existing Customer':'blue'})

    fig.update_layout(scene={"aspectratio": {"x": 3, "y": 4, "z": 3},
                             },
                      autosize=True,
                      margin=dict(l=20, r=20, b=50, t=20),
                      template='plotly_white',
                      showlegend=True
                      )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)')

    fig.update_traces(marker={'size': 3})

    return fig


@st.cache(max_entries=10, ttl=3600)
def plot_3D_all(df, features):

    fig = px.scatter_3d(df, x=features[0], y=features[1], z=features[2],
                        color='Attrition_Flag', color_discrete_map={'Attrited Customer':'red',
                                                                    'Existing Customer':'blue'})

    fig.update_layout(scene={"aspectratio": {"x": 3, "y": 4, "z": 3},
                             },
                      autosize=True,
                      margin=dict(l=20, r=20, b=20, t=20),
                      template='plotly_white',
                      showlegend=True
                      )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)')

    fig.update_traces(marker={'size': 3})

    return fig


@st.cache(max_entries=10, ttl=3600)
def get_corr(df):
    table = df.copy()
    table['Attrition_Flag'] = table['Attrition_Flag'].apply(lambda x: 1 if x == "Attrited Customer" else 0)
    small_df = table[['Total_Relationship_Count',
        'Total_Revolving_Bal',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Avg_Open_To_Buy',
        'Credit_Limit',
        'Attrition_Flag'
       ]]

    return small_df.corr()

