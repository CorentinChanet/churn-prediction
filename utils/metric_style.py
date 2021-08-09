# ©sjd333
# https://discuss.streamlit.io/t/metrics-kpi-component/6991
# https://github.com/samdobson/streamlit-metrics

import streamlit as st
import streamlit.components.v1 as components
from jinja2 import Template

def _build_metric(label, value, color_label='black', color_value='black', value_size='48px'):
    html_text = """
    <style>
    .metric {
       font-family: "IBM Plex Sans", sans-serif;
       text-align: center;
    }
    .metric .value {
       font-size: {{ value_size }};
       line-height: 1.6;
       color: {{ color_value }};
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
    </div>
    """
    html = Template(html_text)
    return html.render(label=label, value=value, color_label=color_label, color_value=color_value, value_size=value_size)

def metric_row(data):
    columns = st.columns(len(data))
    for i, (label, value) in enumerate(data.items()):
        with columns[i]:
            components.html(_build_metric(label, value))

def metric_row_custom(data):
    columns = st.columns(len(data))
    for i, (label, (value, color_label, color_value, value_size)) in enumerate(data.items()):
        with columns[i]:
            components.html(_build_metric(label, value, color_label, color_value, value_size))

def metric(label, value):
    components.html(_build_metric(label, value))
