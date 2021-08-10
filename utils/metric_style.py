# ©sjd333
# https://discuss.streamlit.io/t/metrics-kpi-component/6991
# https://github.com/samdobson/streamlit-metrics

import streamlit as st
import streamlit.components.v1 as components
from jinja2 import Template

def _build_metric(label, value, value_2='', color_label='black', color_value='black', color_value_2='black', value_size='48px'):
    html_text = """
    <style>
    .metric {
       font-family: "IBM Plex Sans", sans-serif;
       text-align: center;
    }
    .metric .value {
       font-size: {{ value_size }};
       line-height: 1.8;
       color: {{ color_value }};
    }
    
    .metric .value_2 {
        letter-spacing: 2px;
       font-size: {{ value_size }};
       line-height: 1.7;
       color: {{ color_value_2 }};
    }
    
    .metric .label {
       letter-spacing: 2px;
       font-size: 16px;
       text-transform: uppercase;
       color: {{ color_label }};
    }

    </style>
    <div class="metric">
       <div class="value">
          {{ value }}
       </div>
       <div class="label">
          {{ label }}
       </div>
        <div class="value_2">
          {{ value_2 }} 
       </div>
    </div>
    """
    html = Template(html_text)
    return html.render(label=label, value=value, value_2=value_2, color_label=color_label, color_value=color_value, color_value_2=color_value_2, value_size=value_size)

def metric_row(data):
    columns = st.columns(len(data))
    for i, (label, value) in enumerate(data.items()):
        with columns[i]:
            components.html(_build_metric(label, value))

def metric_row_custom(data):
    columns = st.columns(len(data))
    for i, (label, (value, value_2, color_label, color_value, color_value_2, value_size)) in enumerate(data.items()):
        with columns[i]:
            components.html(_build_metric(label, value, value_2, color_label, color_value, color_value_2, value_size))

def metric_row_report(data):
    columns = st.columns([0.5, 1, 1, 1, 0.5])
    for i, (label, (value, value_2, color_label, color_value, color_value_2, value_size)) in enumerate(data.items()):
        with columns[i+1]:
            components.html(_build_metric(label, value, value_2, color_label, color_value, color_value_2, value_size))

def metric(label, value):
    components.html(_build_metric(label, value))
