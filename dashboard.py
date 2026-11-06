"""
SCM Dashboard — Demand Forecasting & Inventory Management
==========================================================
Data  : orders_and_shipments.csv  (auto-detect separator)
Models: ARIMA(1,0,0) · XGBoost
Pages : Overview | Demand Forecasting | Model Comparison | Inventory Management
Run   : streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import math
import os

# Absolute path — works regardless of which directory Streamlit is launched from
_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "orders_and_shipments.csv")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SCM Dashboard – Demand Forecasting & Inventory",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRODUCTS = [
    "Perfect Fitness Perfect Rip Deck",
    "Nike Men's CJ Elite 2 TD Football Cleat",
    "Nike Men's Dri-FIT Victory Golf Polo",
    "O'Brien Men's Neoprene Life Vest",
    "Field & Stream Sportsman 16 Gun Fire Safe",
    "Pelican Sunstream 100 Kayak",
    "Diamondback Women's Serene Classic Comfort Bi",
    "Nike Men's Free 5.0+ Running Shoe",
    "Under Armour Girls' Toddler Spine Surge Runni",
    "Team Golf Pittsburgh Steelers Putter Grip",
]

HORIZONS    = [3, 5, 7]
MIN_TRAIN   = 18
DATE_CUTOFF = "2017-09-01"

# ─────────────────────────────────────────────────────────────────────────────
# DATA + MODEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data ...")
def load_data():
    for sep in (";", ","):
        df = pd.read_csv(CSV_PATH, sep=sep)
        df.columns = df.columns.str.strip()
        if len(df.columns) > 5:
            return df
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df


def get_monthly_demand(df, product):
    sub = df[df["Product Name"] == product].copy()
    md  = (
        sub.groupby(["Order Year", "Order Month"])["Order Quantity"]
        .sum()
        .reset_index()
    )
    md["Date"] = pd.to_datetime(
        md["Order Year"].astype(str) + "-" + md["Order Month"].astype(str) + "-01"
    )
    md = md[["Date", "Order Quantity"]].rename(columns={"Order Quantity": "Demand"})
    md = md.sort_values("Date").reset_index(drop=True)
    md = md[md["Date"] <= DATE_CUTOFF].copy()
    return md


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def forecast_arima(train, h, order=(1, 0, 0)):
    model = ARIMA(train, order=order).fit()
    return np.asarray(model.forecast(h))


def create_lag_features(series, lags=3):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def forecast_xgboost(train, h, lags=3):
    if len(train) <= lags:
        return np.repeat(train[-1], h)
    X_tr, y_tr = create_lag_features(train, lags=lags)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    history = list(train.copy())
    preds   = []
    for _ in range(h):
        xi   = np.array(history[-lags:]).reshape(1, -1)
        pred = model.predict(xi)[0]
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


MODELS = {
    "ARIMA(1,0,0)": lambda tr, h: forecast_arima(tr, h),
    "XGBoost":      lambda tr, h: forecast_xgboost(tr, h, lags=3),
}


@st.cache_data(show_spinner="Loading evaluation results ...")
def evaluate_all():
    cv_path = os.path.join(_HERE, "cv_results.csv")
    if os.path.exists(cv_path):
        return pd.read_csv(cv_path)
    # Fallback: compute on-the-fly if CSV not found
    df   = load_data()
    rows = []
    for product in PRODUCTS:
        md = get_monthly_demand(df, product)
        y  = md["Demand"].astype(float).values
        for horizon in HORIZONS:
            last_origin = len(y) - horizon
            origins     = range(MIN_TRAIN, last_origin + 1, horizon)
            for model_name, model_fn in MODELS.items():
                fold_metrics = []
                for origin in origins:
                    train = y[:origin]
                    test  = y[origin:origin + horizon]
                    try:
                        pred = np.asarray(model_fn(train, horizon), dtype=float)
                        if len(pred) != horizon or not np.all(np.isfinite(pred)):
                            continue
                        fold_metrics.append({
                            "MAE":  _mae(test, pred),
                            "RMSE": _rmse(test, pred),
                            "MAPE": _mape(test, pred),
                        })
                    except Exception:
                        continue
                if not fold_metrics:
                    continue
                maes  = [m["MAE"]  for m in fold_metrics]
                rmses = [m["RMSE"] for m in fold_metrics]
                mapes = [m["MAPE"] for m in fold_metrics if math.isfinite(m["MAPE"])]
                rows.append({
                    "Product": product,
                    "Model":   model_name,
                    "Horizon": horizon,
                    "MAE":     float(np.mean(maes)),
                    "RMSE":    float(np.mean(rmses)),
                    "MAPE":    float(np.mean(mapes)) if mapes else float("nan"),
                    "Folds":   len(fold_metrics),
                })
    return pd.DataFrame(rows)


def get_best_model(results_df, product, horizon, metric="MAE"):
    sub = results_df[
        (results_df["Product"] == product) & (results_df["Horizon"] == horizon)
    ]
    if sub.empty:
        return None, None
    idx = sub[metric].idxmin()
    row = sub.loc[idx]
    return row["Model"], row[metric]


@st.cache_data(show_spinner="Generating forecast ...")
def generate_forecast(product, horizon, model_name):
    df = load_data()
    md = get_monthly_demand(df, product)
    y  = md["Demand"].astype(float).values
    fn = MODELS[model_name]
    try:
        pred = np.asarray(fn(y, horizon), dtype=float)
    except Exception:
        pred = np.full(horizon, float("nan"))
    last_date    = md["Date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    return md, pd.DataFrame({"Date": future_dates, "Forecast": pred})


def compute_inventory(demand_series, lead_time=2, service_z=1.645,
                      holding_frac=0.20, order_cost=50.0, unit_cost=10.0):
    avg_demand   = float(np.mean(demand_series))
    std_demand   = float(np.std(demand_series, ddof=1)) if len(demand_series) > 1 else 1.0
    safety_stock = service_z * std_demand * math.sqrt(lead_time)
    rop          = avg_demand * lead_time + safety_stock
    annual_d     = avg_demand * 12
    holding_cost = holding_frac * unit_cost
    eoq = math.sqrt((2 * annual_d * order_cost) / holding_cost) if holding_cost > 0 else float("nan")
    return {
        "Avg Monthly Demand":  round(avg_demand, 1),
        "Std Demand":          round(std_demand, 1),
        "Safety Stock":        round(safety_stock, 0),
        "Reorder Point (ROP)": round(rop, 0),
        "EOQ":                 round(eoq, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("SCM Dashboard")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Demand Forecasting", "Model Comparison", "Inventory Management"],
        index=0,
    )
    st.divider()
    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 10px;
            padding: 14px 16px;
            font-family: sans-serif;
            font-size: 0.82rem;
            line-height: 1.8;
            color: #c9d9f0;
        ">
            <div style="color:#6fa3ef; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:2px;">Data Source</div>
            orders_and_shipments.csv
            <div style="color:#6fa3ef; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.1em; margin-top:10px; margin-bottom:2px;">Models</div>
            ARIMA &nbsp;&middot;&nbsp; XGBoost
            <div style="color:#6fa3ef; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.1em; margin-top:10px; margin-bottom:2px;">Period</div>
            Jan 2015 &ndash; Sep 2017
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Overview — Supply Chain Summary")

    df = load_data()

    total_orders   = int(df["Order Quantity"].sum())
    total_products = int(df["Product Name"].nunique())
    total_revenue  = int(df["Gross Sales"].sum())
    total_profit   = int(df["Profit"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders",    f"{total_orders:,}")
    c2.metric("Unique Products", f"{total_products}")
    c3.metric("Gross Sales",     f"${total_revenue:,}")
    c4.metric("Total Profit",    f"${total_profit:,}")

    st.divider()

    st.subheader("Monthly Demand Trend — 10 Analysed Products")
    all_md = []
    for prod in PRODUCTS:
        md = get_monthly_demand(df, prod)
        md["Product"] = prod
        all_md.append(md)
    combined = pd.concat(all_md, ignore_index=True)
    fig_trend = px.line(
        combined, x="Date", y="Demand", color="Product",
        template="plotly_dark",
        labels={"Demand": "Monthly Demand (units)", "Date": "Period"},
    )
    fig_trend.update_layout(
        height=420,
        legend=dict(font=dict(size=10), orientation="v", x=1.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,35,0.85)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 10 Products by Total Demand")
        top10 = (
            df.groupby("Product Name")["Order Quantity"]
            .sum().nlargest(10).reset_index()
            .rename(columns={"Order Quantity": "Total Demand"})
        )
        fig_top = px.bar(
            top10.sort_values("Total Demand"),
            x="Total Demand", y="Product Name", orientation="h",
            template="plotly_dark", color="Total Demand",
            color_continuous_scale="Blues",
        )
        fig_top.update_layout(
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,20,35,0.85)", coloraxis_showscale=False,
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_b:
        st.subheader("Orders by Shipment Mode")
        ship_mode = df.groupby("Shipment Mode")["Order Quantity"].sum().reset_index()
        fig_ship  = px.pie(
            ship_mode, values="Order Quantity", names="Shipment Mode",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4,
        )
        fig_ship.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ship, use_container_width=True)

    st.divider()
    st.subheader("Orders by Customer Market")
    market_df = df.groupby("Customer Market")["Order Quantity"].sum().reset_index()
    fig_market = px.bar(
        market_df.sort_values("Order Quantity", ascending=False),
        x="Customer Market", y="Order Quantity",
        template="plotly_dark", color="Order Quantity",
        color_continuous_scale="Teal", labels={"Order Quantity": "Total Orders"},
    )
    fig_market.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,35,0.85)", coloraxis_showscale=False,
    )
    st.plotly_chart(fig_market, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DEMAND FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Demand Forecasting":
    st.title("Demand Forecasting")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        sel_product = st.selectbox("Select Product", PRODUCTS)
    with col_ctrl2:
        sel_horizon = st.selectbox("Forecast Horizon (months)", HORIZONS)
    with col_ctrl3:
        sel_model = st.selectbox("Model", list(MODELS.keys()))

    st.divider()

    hist_df, forecast_df = generate_forecast(sel_product, sel_horizon, sel_model)

    results = evaluate_all()
    sub = results[
        (results["Product"] == sel_product)
        & (results["Horizon"] == sel_horizon)
        & (results["Model"]   == sel_model)
    ]
    if not sub.empty:
        row = sub.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE",      f"{row['MAE']:.1f}")
        m2.metric("RMSE",     f"{row['RMSE']:.1f}")
        m3.metric("MAPE",     f"{row['MAPE']:.1f}%" if math.isfinite(row['MAPE']) else "N/A")
        m4.metric("CV Folds", f"{int(row['Folds'])}")

    st.divider()

    st.subheader("Historical Demand & Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["Date"], y=hist_df["Demand"],
        mode="lines+markers", name="Historical Demand",
        line=dict(color="#4285f4", width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Forecast"],
        mode="lines+markers", name=f"Forecast ({sel_model})",
        line=dict(color="#34a853", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
    ))
    fig.add_vline(
        x=hist_df["Date"].iloc[-1].timestamp() * 1000,
        line_dash="dot", line_color="#fbbc04",
        annotation_text="Forecast Start", annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_dark", height=450,
        xaxis_title="Period", yaxis_title="Demand (units)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,20,35,0.85)",
        legend=dict(orientation="h", y=1.06),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Values")
    fcd = forecast_df.copy()
    fcd["Date"]     = fcd["Date"].dt.strftime("%b %Y")
    fcd["Forecast"] = fcd["Forecast"].round(0).astype(int)
    st.dataframe(
        fcd.rename(columns={"Date": "Period", "Forecast": "Predicted Demand"}),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("ARIMA vs XGBoost Forecast (same horizon)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist_df["Date"], y=hist_df["Demand"],
        mode="lines+markers", name="Historical",
        line=dict(color="#4285f4", width=2), marker=dict(size=4),
    ))
    colors = {"ARIMA(1,0,0)": "#ea4335", "XGBoost": "#34a853"}
    for mname in MODELS.keys():
        _, fc = generate_forecast(sel_product, sel_horizon, mname)
        fig2.add_trace(go.Scatter(
            x=fc["Date"], y=fc["Forecast"],
            mode="lines+markers", name=mname,
            line=dict(color=colors[mname], width=2.5, dash="dash"),
            marker=dict(size=7),
        ))
    fig2.add_vline(
        x=hist_df["Date"].iloc[-1].timestamp() * 1000,
        line_dash="dot", line_color="#fbbc04",
    )
    fig2.update_layout(
        template="plotly_dark", height=400,
        xaxis_title="Period", yaxis_title="Demand (units)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,20,35,0.85)",
        legend=dict(orientation="h", y=1.06),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("Cross-validation results for all products, models, and forecast horizons.")

    results = evaluate_all()
    if results.empty:
        st.warning("No evaluation results available.")
        st.stop()

    col_h, col_m = st.columns(2)
    with col_h:
        sel_horizon_cmp = st.selectbox("Horizon (months)", HORIZONS, key="cmp_horizon")
    with col_m:
        sel_metric_cmp = st.selectbox("Metric", ["MAE", "RMSE", "MAPE"], key="cmp_metric")

    st.divider()
    sub_cmp = results[results["Horizon"] == sel_horizon_cmp].copy()

    st.subheader("Full Results Table")
    disp = sub_cmp[["Product", "Model", "Horizon", "MAE", "RMSE", "MAPE", "Folds"]].copy()
    disp["MAE"]  = disp["MAE"].round(2)
    disp["RMSE"] = disp["RMSE"].round(2)
    disp["MAPE"] = disp["MAPE"].round(2)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"{sel_metric_cmp} by Product & Model — Horizon {sel_horizon_cmp}m")
    fig_bar = px.bar(
        sub_cmp, x="Product", y=sel_metric_cmp, color="Model",
        barmode="group", template="plotly_dark",
        color_discrete_map={"ARIMA(1,0,0)": "#ea4335", "XGBoost": "#34a853"},
    )
    fig_bar.update_layout(
        height=420, xaxis_tickangle=-35,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,20,35,0.85)",
        legend=dict(orientation="h", y=1.06),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader(f"Heatmap — {sel_metric_cmp} (Horizon {sel_horizon_cmp}m)")
    pivot = sub_cmp.pivot(index="Product", columns="Model", values=sel_metric_cmp)
    fig_heat = px.imshow(
        pivot, text_auto=".1f",
        color_continuous_scale="Blues", template="plotly_dark", aspect="auto",
    )
    fig_heat.update_layout(
        height=420, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,35,0.85)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.subheader("Best Model per Product (by MAE)")
    best_rows = []
    for prod in PRODUCTS:
        for h in HORIZONS:
            best_m, best_val = get_best_model(results, prod, h, metric="MAE")
            if best_m:
                best_rows.append({"Product": prod, "Horizon": h,
                                   "Best Model": best_m, "MAE": round(best_val, 2)})
    best_df = pd.DataFrame(best_rows)
    if not best_df.empty:
        best_pivot = best_df.pivot(index="Product", columns="Horizon", values="Best Model")
        st.dataframe(best_pivot, use_container_width=True)

        model_counts = best_df["Best Model"].value_counts().reset_index()
        model_counts.columns = ["Model", "Count"]
        fig_pie = px.pie(
            model_counts, values="Count", names="Model",
            title="Best Model Frequency (all products & horizons)",
            template="plotly_dark",
            color_discrete_map={"ARIMA(1,0,0)": "#ea4335", "XGBoost": "#34a853"},
            hole=0.4,
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INVENTORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Inventory Management":
    st.title("Inventory Management")
    st.markdown(
        "Parameters: **Lead Time** = 2 months · **Service Level** = 95% (Z = 1.645) · "
        "**Holding Cost** = 20% of unit value · **Order Cost** = $50 · **Unit Cost** = $10"
    )

    df = load_data()

    col_s, col_l = st.columns([3, 1])
    with col_s:
        sel_inv_product = st.selectbox("Select Product", PRODUCTS, key="inv_product")
    with col_l:
        lead_time = st.number_input("Lead Time (months)", min_value=1, max_value=12, value=2)

    st.divider()

    md  = get_monthly_demand(df, sel_inv_product)
    inv = compute_inventory(md["Demand"].values, lead_time=lead_time)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Monthly Demand", f"{inv['Avg Monthly Demand']}")
    c2.metric("Std Deviation",      f"{inv['Std Demand']}")
    c3.metric("Safety Stock",       f"{int(inv['Safety Stock'])} units")
    c4.metric("Reorder Point",      f"{int(inv['Reorder Point (ROP)'])} units")
    c5.metric("EOQ",                f"{int(inv['EOQ'])} units")

    st.divider()

    st.subheader("Demand History with Inventory Thresholds")
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(
        x=md["Date"], y=md["Demand"],
        mode="lines+markers", name="Monthly Demand",
        line=dict(color="#4285f4", width=2), marker=dict(size=5),
    ))
    fig_inv.add_hline(
        y=inv["Reorder Point (ROP)"], line_dash="dash", line_color="#ea4335",
        annotation_text=f"ROP = {int(inv['Reorder Point (ROP)'])}",
        annotation_position="right",
    )
    fig_inv.add_hline(
        y=inv["Safety Stock"], line_dash="dot", line_color="#fbbc04",
        annotation_text=f"Safety Stock = {int(inv['Safety Stock'])}",
        annotation_position="right",
    )
    fig_inv.update_layout(
        template="plotly_dark", height=420,
        xaxis_title="Period", yaxis_title="Units",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,20,35,0.85)",
        legend=dict(orientation="h", y=1.06),
    )
    st.plotly_chart(fig_inv, use_container_width=True)

    st.divider()

    st.subheader("Inventory Summary — All Products")
    all_inv = []
    for prod in PRODUCTS:
        md_p  = get_monthly_demand(df, prod)
        inv_p = compute_inventory(md_p["Demand"].values, lead_time=lead_time)
        inv_p["Product"] = prod
        all_inv.append(inv_p)
    inv_df = pd.DataFrame(all_inv)[
        ["Product", "Avg Monthly Demand", "Std Demand",
         "Safety Stock", "Reorder Point (ROP)", "EOQ"]
    ]
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Economic Order Quantity (EOQ) per Product")
    fig_eoq = px.bar(
        inv_df.sort_values("EOQ", ascending=False),
        x="Product", y="EOQ", template="plotly_dark",
        color="EOQ", color_continuous_scale="Blues",
    )
    fig_eoq.update_layout(
        height=380, xaxis_tickangle=-30,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,20,35,0.85)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_eoq, use_container_width=True)

    st.subheader("Reorder Point (ROP) per Product")
    fig_rop = px.bar(
        inv_df.sort_values("Reorder Point (ROP)", ascending=False),
        x="Product", y="Reorder Point (ROP)", template="plotly_dark",
        color="Reorder Point (ROP)", color_continuous_scale="Teal",
    )
    fig_rop.update_layout(
        height=380, xaxis_tickangle=-30,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,20,35,0.85)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_rop, use_container_width=True)
