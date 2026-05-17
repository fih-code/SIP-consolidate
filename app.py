# Run locally with: streamlit run app.py

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from statsmodels.distributions.empirical_distribution import ECDF
import streamlit as st

st.set_page_config(layout="centered", page_title="SIP Portfolio Risk")


# ------------------------------------------------------------------
# ECDF helpers
# ------------------------------------------------------------------

def _prepare_curve(series, n_points=500):
    arr = np.asarray(series).ravel()
    ecdf = ECDF(arr)
    loss_grid = np.linspace(arr.min(), arr.max(), n_points)
    exceed_pct = 100 * (1 - ecdf(loss_grid))
    return loss_grid, exceed_pct


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_combined(series_dict, agg_series, title, ymax, xmin, xmax,
                  appetite_pts=None, currency="MSEK"):
    """Individual LEC curves + aggregate curve on the same axes."""
    palette = sns.color_palette('tab10', len(series_dict))
    fig, ax = plt.subplots(figsize=(9, 5))

    for (label, arr), color in zip(series_dict.items(), palette):
        loss_grid, exceed_pct = _prepare_curve(arr)
        ax.plot(loss_grid, exceed_pct, linewidth=1.5, color=color,
                label=label, alpha=0.8)

    loss_agg, exceed_agg = _prepare_curve(agg_series)
    ax.plot(loss_agg, exceed_agg, linewidth=2.5, color='black',
            linestyle='-', label='Aggregate')

    if appetite_pts and any(p[1] > 0 for p in appetite_pts):
        pts = sorted(appetite_pts, key=lambda p: p[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                linestyle='--', linewidth=1.5, color='red',
                marker='o', markersize=5, label='Risk appetite')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Loss amount ({currency})', fontsize=11)
    ax.set_ylabel('Exceedance Probability (%)', fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.set_ylim(0, ymax)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=9, loc='upper right')
    return fig


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

currency = st.sidebar.text_input("Currency", value="MSEK")

st.title('SIP Portfolio Risk')
st.write('Upload up to 10 SIP Excel files to compare individual risks '
         'and view the aggregate loss exceedance curve.')

uploaded = st.file_uploader(
    'SIP Excel files (up to 10)',
    type=['xlsx'],
    accept_multiple_files=True,
)

if len(uploaded) > 10:
    st.warning('More than 10 files uploaded — only the first 10 will be used.')
    uploaded = uploaded[:10]

if not uploaded:
    st.info('Upload at least one file to continue.')
    st.stop()

# Read all files
dfs = {}
for f in uploaded:
    label = f.name.removesuffix('.xlsx').removesuffix('.XLSX')
    dfs[label] = pd.read_excel(f)

# Column selection — common numeric columns, default risk_sip
all_numeric = [set(df.select_dtypes(include='number').columns) for df in dfs.values()]
common_cols = sorted(set.intersection(*all_numeric)) if all_numeric else []

if not common_cols:
    st.error('The uploaded files share no numeric columns.')
    st.stop()

default_col = 'risk_sip' if 'risk_sip' in common_cols else common_cols[0]
col = st.selectbox('Column to use', common_cols, index=common_cols.index(default_col))

series_dict = {label: np.asarray(df[col]).ravel() for label, df in dfs.items()}
row_counts = [len(arr) for arr in series_dict.values()]

if len(set(row_counts)) > 1:
    st.warning(
        f'Files have different row counts {sorted(set(row_counts))}. '
        f'Aggregate uses the shortest ({min(row_counts)} rows).'
    )

min_rows = min(row_counts)
agg = sum(arr[:min_rows] for arr in series_dict.values())

# Axis controls
with st.expander('Axis limits', expanded=False):
    ac1, ac2, ac3 = st.columns(3)
    xmin = ac1.number_input(f'X min ({currency})', value=0.0, min_value=0.0, format='%.1f')
    xmax = ac2.number_input(f'X max ({currency})',
                             value=float(max(arr.max() for arr in series_dict.values()) * 1.1),
                             min_value=0.001, format='%.1f')
    ymax = ac3.number_input('Y max (%)', value=100.0,
                             min_value=1.0, max_value=100.0, format='%.1f')

# Risk appetite
appetite_pts = None
with st.expander('Risk appetite', expanded=False):
    if st.checkbox('Show risk appetite line'):
        st.caption('Enter three coordinates — the line appears once at least one probability is above zero.')
        hc1, hc2 = st.columns(2)
        hc1.markdown(f'**Loss ({currency})**')
        hc2.markdown('**Probability (%)**')
        pts = []
        for i in range(3):
            rc1, rc2 = st.columns(2)
            loss = rc1.number_input(f'Loss {i+1}', min_value=0.0,
                                    key=f'ap_loss_{i}', label_visibility='collapsed', format='%.1f')
            prob = rc2.number_input(f'Prob {i+1}', min_value=0.0, max_value=100.0,
                                    key=f'ap_prob_{i}', label_visibility='collapsed', format='%.1f')
            pts.append((float(loss), float(prob)))
        appetite_pts = tuple(pts)

chart_title = st.text_input('Chart title', value='Portfolio Risk — Loss Exceedance Curves')

if st.button('Plot', type='primary'):
    fig = plot_combined(series_dict, agg, chart_title, ymax, xmin, xmax, appetite_pts, currency)
    png = _to_png(fig)
    st.image(png, use_container_width=True)
    st.download_button('Download chart (PNG)', data=png,
                       file_name='portfolio_lec.png', mime='image/png')
