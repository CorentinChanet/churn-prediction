from sklearn.metrics import plot_confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr(df):
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
    fig, ax = plt.subplots()
    return fig

def plot_3D(df):
    table = df.copy()
    small_df = table[['Total_Relationship_Count',
        'Total_Revolving_Bal',
        'Total_Trans_Amt',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Attrition_Flag'
       ]]

    fig = px.scatter_3d(small_df, x='Total_Revolving_Bal', y='Total_Ct_Chng_Q4_Q1', z='Total_Trans_Amt',
                        color='Attrition_Flag', color_discrete_map={'Attrited Customer':'orange',
                                                                    'Existing Customer':'blue'})

    fig.update_layout(scene={"aspectratio": {"x": 3, "y": 4, "z": 3}},
                      autosize=True,
                      scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                      width=800, height=500,
                      margin=dict(l=20, r=20, b=20, t=20),
                      template='plotly_dark'
                      )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)')

    fig.update_traces(marker={'size': 3})

    return fig

def plot_confusion_mtx(name):

    confusion_matrix = classifiers[name]['confusion_matrix']

    z_text = [[f'{confusion_matrix[0][0]}\nTrue Positive',
               f'{confusion_matrix[0][1]}\nFalse Negative'],
              [f'{confusion_matrix[1][0]}\nFalse Positive',
               f'{confusion_matrix[1][1]}\nTrue Negative']]

    my_colorsc = [[0, 'rgb(244,113,116)'],  # red
                  [0.10, 'rgb(244,113,116)'],
                  [0.11, 'rgb(0,168,107)'],  # green
                  [1, 'rgb(0,168,107)']]

    fig = ff.create_annotated_heatmap(confusion_matrix, annotation_text=z_text, colorscale=my_colorsc)

    fig.update_layout({'autosize':True,
                       'width':400,
                       'height':400,
                       'margin' : dict(l=20, r=20, t=20, b=20),
                       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                       'font' : dict(family="Verdana",size=14),
    })

    return fig


