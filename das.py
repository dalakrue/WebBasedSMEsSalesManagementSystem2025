import sys
import datetime as dt
from typing import Dict, List, Optional, Tuple
import yaml
from yaml.loader import SafeLoader
import os
import pandas as pd
import numpy as np
import streamlit as st
import sqlalchemy as sa
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import io
import xlsxwriter
from functools import lru_cache          # NEW (used later for DB cache)
CSV_TABLES = [
    "users", "orders", "order_details",
    "products", "payments", "leads",
    "deals", "activities"
]
# Holds any uploaded CSVs when in CSV‐only mode or for overrides
csv_data: Dict[str, pd.DataFrame] = {}

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()
uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    df_csv = pd.read_csv(uploaded_file)
if st.button("📥 Export to Excel"):
    if 'df_csv' in locals():
        st.download_button("Download Excel", to_excel(df_csv), file_name="data_export.xlsx")
    else:
        st.warning("No data to export.")
from fpdf import FPDF
def generate_pdf_report(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary_text.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)
    return pdf.output(dest="S").encode("latin1")

if st.button("📄 Export Summary as PDF"):
    summary = "Sales Report\n\nTop 3 Products...\nRevenue Forecast...\n"
    pdf_bytes = generate_pdf_report(summary)
    st.download_button("Download PDF", pdf_bytes, file_name="summary.pdf")


# ============================== 2. Page Setup ================================
st.set_page_config(page_title="📊 SME Sales Management Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")
st.title("📊 Advanced Sales Management Dashboard for SMEs")
# ------------------ SESSION STATE SAFE DEFAULTS ------------------
# initialize keys we rely on so setting them later is safe
if "sqlite_path" not in st.session_state:
    st.session_state["sqlite_path"] = ""   # file path for SQLite DB
# Optional: keep engine key defined but leave value as-is if user connected earlier
# (get_engine already uses st.session_state.get(_DB_KEY), so no strict need to set it here)
# ----------------------------------------------------------------

# ============================== 3. Helper & Utility Functions ================

# ---------- 3.1 Database Connection Management -------------------------------
_DB_KEY = "db_engine"

def get_engine():
    """Return a cached engine or None."""
    return st.session_state.get(_DB_KEY)
def disconnect_db():
    """
    Dispose and forget the cached DB engine.
    """
    engine = st.session_state.pop(_DB_KEY, None)
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            pass
    st.success("🔌 Disconnected from DB")

def connect_db(db_type: Optional[str] = None):
    """
    Build and cache a DB engine from sidebar inputs.
    Supports MySQL and SQLite.
    """
    if _DB_KEY in st.session_state:
        st.info("Already connected.")
        return st.session_state[_DB_KEY]

    # Read DB type (default to MySQL if not set)
    db_type = db_type or st.session_state.get("db_type", "MySQL")

    try:
        if db_type == "SQLite":
    # Expect a local file path from the sidebar (or uploaded file saved to disk)
          sqlite_path = st.session_state.get("sqlite_path", "") or ""
          if not sqlite_path:
           st.error("Provide a SQLite file path (type it or upload a .db file).")
           return None

    # Check file exists and is readable
          if not os.path.exists(sqlite_path):
            st.error(f"SQLite file not found: {sqlite_path}. Please upload or correct the path.")
            return None

          uri = f"sqlite+pysqlite:///{sqlite_path}"
          engine = sa.create_engine(uri, connect_args={"check_same_thread": False})

        else:
            # MySQL path (values should already be in session_state from the sidebar)
            host     = st.session_state.get("mysql_host", "localhost")
            port     = st.session_state.get("mysql_port", "3306")
            db_name  = st.session_state.get("mysql_db", "sample_data")
            user     = st.session_state.get("mysql_user", "root")
            password = st.session_state.get("mysql_password", "root")

            if not all([host, port, db_name, user]):
                st.warning("Please fill in all database connection details.")
                return None

            uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
            engine = sa.create_engine(uri, pool_pre_ping=True, pool_recycle=280)

        # Test connection
        with engine.connect() as _:
            pass

        st.session_state[_DB_KEY] = engine
        st.success(f"✅ Connected to {db_type}")
        return engine

    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        return None



def merge_price_into_order_details(od: pd.DataFrame,
                                   prod: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure order_details has a 'price_each' column.
    If missing, attempt to merge with products['price'] (or 'unit_price').
    """
    if od.empty:
        st.warning("order_details is empty; skipping price merge.")
        return od
    if 'price_each' in od.columns:
        return od

    possible_price_cols = ['price', 'unit_price', 'unitPrice', 'unitprice',
                           'price_each', 'price_each_usd']
    price_col_in_products = next((col for col in possible_price_cols if col in prod.columns), None)

    if price_col_in_products is None:
        st.warning(
            "⚠️ Could not locate a price column in products; "
            "order_details will not have revenue calculations."
        )
        od['price_each'] = np.nan
        return od

    if 'product_id' in od.columns and 'product_id' in prod.columns:
        od['product_id'] = od['product_id'].astype(str)
        prod['product_id'] = prod['product_id'].astype(str)
    else:
        st.warning("product_id missing in either order_details or products.")
        od['price_each'] = np.nan
        return od

    od = pd.merge(
        od,
        prod[['product_id', price_col_in_products]].rename(
            columns={price_col_in_products: 'price_each'}
        ),
        on='product_id',
        how='left'
    )

    if od['price_each'].isna().all():
        st.warning("⚠️ After merge, all price_each values are NaN.")
    else:
        st.success(f"✅ price_each successfully merged from products['{price_col_in_products}'].")

    return od

def _coerce_key_cols(df_left, df_right,
                     left_key='customer_id',
                     right_key='user_id',
                     dtype='str'):
    """
    Make sure the two key columns have the same dtype
    so that pd.merge() does not raise the int64/object
    mismatch error.
    """
    df_left[left_key]  = df_left[left_key].astype(dtype)
    df_right[right_key] = df_right[right_key].astype(dtype)
    return df_left, df_right

# ---------- 3.3 KPI Calculation ----------------------------------------------
def kpi_leads_vs_target(leads_df, annual_target: int):
    """
    Return actual count, target and achievement %.
    Expects leads_df to hold one row per lead with 'created_at' datetime.
    """
    if 'created_date' not in leads_df.columns: return 0, annual_target, 0
    
    leads_df['created_date'] = pd.to_datetime(leads_df['created_date'])
    this_year = pd.Timestamp("now").year
    actual = leads_df[leads_df['created_date'].dt.year == this_year].shape[0]
    pct = (actual / annual_target) * 100 if annual_target else 0
    return actual, annual_target, pct

# ---------- 3.4 UI Components ------------------------------------------------
def kpi_card(value, label, delta=None, prefix="", help_text=""):
    """
    Render a single KPI card with optional delta and help text.
    """
    st.metric(label, f"{prefix}{value:,.0f}", delta=delta, help=help_text)

def spacer(lines: int = 1):
    """
    Produce vertical space in Streamlit.
    """
    for _ in range(lines):
        st.write("")

# ============================== 4. Sidebar & Data Loading ====================
st.sidebar.title("Configuration")
st.sidebar.header("1. Database")
# ------------------------------------------------------------------
# Choose whether we want to use the database or rely on CSV only
# ------------------------------------------------------------------
data_source = st.sidebar.radio(
    "Data source",
    ("Database", "CSV only"),
    horizontal=True,
    key="data_source_mode"
)
if data_source == "Database":
    st.sidebar.subheader("Database Connection")

    # Choose DB Type
    db_type = st.sidebar.radio(
        "DB Type",
        ("MySQL", "SQLite"),
        horizontal=True,
        key="db_type"
    )

    if db_type == "MySQL":
        st.sidebar.text_input("Host", value=st.session_state.get("mysql_host", "localhost"), key="mysql_host")
        st.sidebar.text_input("Port", value=st.session_state.get("mysql_port", "3306"), key="mysql_port")
        st.sidebar.text_input("Database Name", value=st.session_state.get("mysql_db", "sample_data"), key="mysql_db")
        st.sidebar.text_input("User", value=st.session_state.get("mysql_user", "root"), key="mysql_user")
        st.sidebar.text_input("Password", value=st.session_state.get("mysql_password", "root"), type="password", key="mysql_password")

    else:
        # SQLite path entry (this text_input writes to st.session_state["sqlite_path"] automatically)
     st.sidebar.text_input(
      "SQLite file path (or upload below)",
       value=st.session_state.get("sqlite_path", "sample_data.sqlite3"),
       key="sqlite_path"
)

# Upload a SQLite file and save it to a tmp folder. Use a safe write + fallback if session_state set fails.
up_db = st.sidebar.file_uploader(
    "…or upload a SQLite .db / .sqlite file",
    type=["db", "sqlite", "sqlite3"],
    key="sqlite_upload"
)

if up_db is not None:
    # create a small tmp folder to avoid writing to root
    tmp_dir = os.path.join(os.getcwd(), "tmp_sqlite")
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = os.path.join(tmp_dir, up_db.name)

    try:
        with open(save_path, "wb") as f:
            f.write(up_db.getbuffer())
    except Exception as e:
        st.sidebar.error(f"Failed to save uploaded file: {e}")
    else:
        # Try to update session_state (may fail on some Streamlit runtimes)
        try:
            with st.spinner("Saving uploaded DB..."):
             with open(save_path, "wb") as f:
              f.write(up_db.getbuffer())
            st.session_state["sqlite_path"] = save_path
            st.sidebar.success(f"Saved uploaded DB to: {save_path}")
        except Exception as e:
            # fallback UX: tell user to press a button to apply the uploaded file
            st.sidebar.warning(
                "Couldn't update session state automatically. Click 'Use uploaded DB' to apply."
            )
            if st.sidebar.button("Use uploaded DB", key="apply_uploaded_sqlite"):
                # attempt again inside a button callback (safer)
                try:
                    st.session_state["sqlite_path"] = save_path
                    st.sidebar.success(f"Applied uploaded DB: {save_path}")
                except Exception as e2:
                    st.sidebar.error(f"Still couldn't set path: {e2}")


    # Action buttons
    if st.sidebar.button("Connect DB"):
        connect_db(st.session_state.get("db_type", "MySQL"))

    if st.sidebar.button("Disconnect DB"):
        disconnect_db()


engine = get_engine() if data_source == "Database" else None

# --- Prepare empty DataFrames ------------------------------------------------
users = pd.DataFrame()
products = pd.DataFrame()
orders = pd.DataFrame()
order_details = pd.DataFrame()
payments = pd.DataFrame()
leads = pd.DataFrame()
deals = pd.DataFrame()
activities = pd.DataFrame()
customer_payments = pd.DataFrame()


# --- Load data from database if connected -------------------------------------
if engine:
    @st.cache_data
    def load_data_from_db(_engine):
        data = {}
        table_names = ["payments", "users", "orders", "order_details", "products",
                       "leads", "deals", "activities", "customer_payments_view"]
        for table in table_names:
            try:
                # Use inspect to check if table exists before querying
                if sa.inspect(_engine).has_table(table):
                    data[table] = pd.read_sql(f"SELECT * FROM {table}", _engine)
                else:
                    data[table] = pd.DataFrame() # Return empty df if table is missing
            except Exception as e:
                st.warning(f"Could not load table '{table}'. Reason: {e}")
                data[table] = pd.DataFrame()
        return data

    loaded_data = load_data_from_db(engine)
    payments = loaded_data.get("payments", pd.DataFrame())
    users = loaded_data.get("users", pd.DataFrame())
    orders = loaded_data.get("orders", pd.DataFrame())
    order_details = loaded_data.get("order_details", pd.DataFrame())
    products = loaded_data.get("products", pd.DataFrame())
    leads = loaded_data.get("leads", pd.DataFrame())
    deals = loaded_data.get("deals", pd.DataFrame())
    activities = loaded_data.get("activities", pd.DataFrame())
    customer_payments = loaded_data.get("customer_payments_view", pd.DataFrame())
    engine = get_engine() if data_source == "Database" else None
    csv_data: Dict[str, pd.DataFrame] = {} 

# ---------- PATCH-7  (new global guard) ----------
if data_source == "Database" and engine is None and not csv_data:
    st.info("Either connect to a database or upload CSV files to proceed.")
    st.stop()
# -------------------------------------------------

# ============================== 4. Sidebar & Data Loading ====================
...

# Add this HERE 👇 (just before pre-processing starts)
def safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Safely convert a column in a DataFrame to datetime, handling errors gracefully.
    """
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            st.warning(f"⚠️ Failed to convert {col} to datetime: {e}")
    else:
        st.warning(f"⚠️ Column `{col}` not found in dataframe.")
    return df
st.header("🔄 Data Pre-processing Status")
with st.expander("Show Details"):
    # --- Ensure dates are proper ---------------------------------------------
    payments = safe_to_datetime(payments, 'payment_date')
    orders = safe_to_datetime(orders, 'order_date')
    customer_payments = safe_to_datetime(customer_payments, 'payment_date')
    leads = safe_to_datetime(leads, 'created_date')
    deals = safe_to_datetime(deals, 'expected_close_date')
    activities = safe_to_datetime(activities, 'activity_date')
    st.write("✓ Date columns converted to datetime format.")
# ---------- PATCH-6  (CSV data take precedence) ----------
    for _name, _df in csv_data.items():
        if not _df.empty:
           locals()[_name] = _df
# ----------------------------------------------------------
    # --- Ensure IDs are strings for joins ------------------------------------
    for df in [orders, order_details, products, users, payments, customer_payments]:
        for col in ['order_id', 'user_id', 'product_id', 'customer_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    st.write("✓ Key columns (IDs) standardized to string type for reliable merges.")
    
    # --- Guarantee price_each in order_details -------------------------------
    order_details = merge_price_into_order_details(order_details, products)

    # --- Compute item_revenue safely -----------------------------------------
    if 'quantity' in order_details.columns and 'price_each' in order_details.columns:
        order_details['item_revenue'] = pd.to_numeric(order_details['quantity'], errors='coerce') * \
                                        pd.to_numeric(order_details['price_each'], errors='coerce')
        st.write("✓ 'item_revenue' calculated for order details.")
    else:
        order_details['item_revenue'] = np.nan

st.markdown("---")

# ============================== 6. KPI SECTION ===============================
st.header("🏆 Key Performance Indicators")

# Calculate KPIs only if the necessary data is available
total_revenue = payments['amount'].sum() if 'amount' in payments.columns else 0
total_orders = orders['order_id'].nunique() if not orders.empty and 'order_id' in orders.columns else 0
total_customers = users[users['user_type'] == 'customer']['user_id'].nunique() if not users.empty and 'user_type' in users.columns else 0
total_products = products['product_id'].nunique() if not products.empty else 0

# Time-based KPIs
revenue_this_month, rev_delta = (0, None)
if 'payment_date' in payments.columns:
    current_month_mask = payments['payment_date'].dt.to_period('M') == pd.Timestamp.today().to_period('M')
    last_month_mask = payments['payment_date'].dt.to_period('M') == (pd.Timestamp.today() - pd.DateOffset(months=1)).to_period('M')
    revenue_this_month = payments.loc[current_month_mask, 'amount'].sum()
    revenue_last_month = payments.loc[last_month_mask, 'amount'].sum()
    if revenue_last_month > 0:
        rev_delta = f"{((revenue_this_month - revenue_last_month) / revenue_last_month * 100):.0f}%"

lead_this_month, lead_delta = (0, None)
if 'created_date' in leads.columns:
    current_month_mask = leads['created_date'].dt.to_period('M') == pd.Timestamp.today().to_period('M')
    last_month_mask = leads['created_date'].dt.to_period('M') == (pd.Timestamp.today() - pd.DateOffset(months=1)).to_period('M')
    lead_this_month = current_month_mask.sum()
    lead_last_month = last_month_mask.sum()
    if lead_last_month > 0:
        lead_delta = f"{(lead_this_month - lead_last_month) / lead_last_month * 100:.0f}% vs last month"
    
pipeline_deals = deals[deals['stage'] != 'Closed Won'].shape[0] if not deals.empty and 'stage' in deals.columns else 0

# Display KPIs
kpi_cols = st.columns(4)
kpi_cols[0].metric("Leads This Month", f"{lead_this_month:,}", delta=lead_delta)
kpi_cols[1].metric("Revenue This Month", f"MMK {revenue_this_month:,.0f}", delta=rev_delta)
kpi_cols[2].metric("Deals in Pipeline", f"{pipeline_deals:,}")
kpi_cols[3].metric("Active Customers", f"{total_customers:,}")

spacer()

glob_cols = st.columns(4)
glob_cols[0].metric("💰 Total Lifetime Revenue", f"MMK {total_revenue:,.0f}")
glob_cols[1].metric("🛒 Total Orders", f"{total_orders:,}")
glob_cols[2].metric("📦 Total Products", f"{total_products:,}")
glob_cols[3].metric("👥 Total Customers", f"{total_customers:,}")

st.markdown("---")

# ============================== 7. Revenue Trend & Forecast ==================
st.header("📈 Revenue Analytics")

if 'payment_date' in payments.columns and not payments.empty:
    payments['month'] = payments['payment_date'].dt.to_period('M').astype(str)
    monthly_rev = payments.groupby('month')['amount'].sum().reset_index()

    tab_trend, tab_forecast = st.tabs(["Monthly Trend", "3-Month Forecast"])

    with tab_trend:
        fig_month = px.line(monthly_rev, x='month', y='amount', title="Actual Monthly Revenue",
                            markers=True, labels={'amount': 'Revenue (MMK)'})
        st.plotly_chart(fig_month, use_container_width=True)

    with tab_forecast:
        if len(monthly_rev) >= 3:
            monthly_rev['idx'] = np.arange(len(monthly_rev))
            model = LinearRegression().fit(monthly_rev[['idx']], monthly_rev['amount'])
            
            future_idx = range(len(monthly_rev), len(monthly_rev) + 3)
            future_amount = model.predict(np.array(future_idx).reshape(-1, 1))
            last_month = pd.to_datetime(monthly_rev['month'].iloc[-1])
            future_months = [(last_month + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(3)]

            future_df = pd.DataFrame({'month': future_months, 'amount': future_amount})
            combined = pd.concat([monthly_rev[['month', 'amount']], future_df])

            fig_fore = px.line(combined, x='month', y='amount', title="Actual + 3-Month Forecast", markers=True,
                               labels={'amount': 'Revenue (MMK)'})
            fig_fore.add_vline(x=monthly_rev['month'].iloc[-1], line_width=2, line_dash="dash", line_color="green")
            st.plotly_chart(fig_fore, use_container_width=True)
        else:
            st.info("📈 Need at least 3 months of revenue data for forecasting.")
else:
    st.info("📈 `payments` table with `payment_date` and `amount` columns is required for revenue analytics.")

st.markdown("---")

# ============================== 8. Target vs Actual ==========================
st.header("🎯 Target vs Actual")

cols_targets = st.columns(2)

with cols_targets[0]:
    if not leads.empty:
        target_leads = st.number_input("Annual Lead Target", min_value=1, value=5000, step=100)
        actual, target, pct = kpi_leads_vs_target(leads, annual_target=target_leads)
        st.metric("Leads (YTD)", f"{actual:,}", delta=f"{pct:0.1f}% of target ({target:,})")

        fig_lead = go.Figure(go.Indicator(
            mode="gauge+number", value=actual,
            title={'text': 'Lead Generation (YTD)'},
            gauge={'axis': {'range': [None, target]}, 'bar': {'color': '#2ECC71'}}))
        st.plotly_chart(fig_lead, use_container_width=True)
    else:
        st.info("🎯 `leads` table is required for this chart.")

with cols_targets[1]:
    if not payments.empty:
        target_rev = st.number_input("Annual Revenue Target (MMK)", min_value=1000, value=50_000_000, step=100_000)
        rev_ytd = payments[payments['payment_date'].dt.year == dt.datetime.today().year]['amount'].sum()
        
        fig_bullet = go.Figure(go.Indicator(
            mode="number+gauge+delta", value=rev_ytd,
            title={'text': "Revenue (YTD)"},
            gauge={'shape': "bullet", 'axis': {'range': [None, target_rev]}, 'bar': {'color': '#5DADE2'}},
            delta={'reference': target_rev, 'position': "bottom"}))
        st.plotly_chart(fig_bullet, use_container_width=True)
    else:
        st.info("🎯 `payments` table is required for this chart.")

st.markdown("---")

st.header("🤖 Customer Segmentation (RFM Lite)")

if not customer_payments.empty and 'customer_id' in customer_payments.columns:
    seg_data = customer_payments.groupby('customer_id').agg(
        recency_days=('payment_date', lambda x: (dt.datetime.now(x.dt.tz) - x.max()).days if x.dt.tz else (dt.datetime.now() - x.max()).days),
        frequency=('payment_id', 'count'),
        monetary=('amount', 'sum')
    ).reset_index()

    use_cols = st.multiselect("Select features for clustering",
                              ['recency_days', 'frequency', 'monetary'],
                              default=['frequency', 'monetary'])

    if len(use_cols) >= 2:
        X_scaled = StandardScaler().fit_transform(seg_data[use_cols])
        k = st.slider("Number of clusters (K)", 2, 6, 3, key="kmeans_k")
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        seg_data['segment'] = km.fit_predict(X_scaled)

        if not users.empty:
            # This is where the offending merge call was.
            # The new function is inserted just before it.
            seg_data, users = _coerce_key_cols(seg_data, users,
                                               left_key='customer_id',
                                               right_key='user_id',
                                               dtype='str')

            # … your original merge line now works
            seg_data = (
                seg_data.merge(
                    users[['user_id', 'name']],
                    left_on='customer_id',
                    right_on='user_id',
                    how='left'
                )
                .rename(columns={'name': 'customer_name'})
            )

        fig_seg = px.scatter(
            seg_data,
            x=use_cols[0], y=use_cols[1],
            color=seg_data['segment'].astype(str),
            size='monetary' if 'monetary' in seg_data.columns else None,
            hover_data=['customer_name', 'recency_days', 'frequency', 'monetary'],
            title="Customer Segments"
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        st.subheader("Segment Summary")
        summary = seg_data.groupby('segment').agg(
            customers=('customer_id', 'count'),
            avg_recency=('recency_days', 'mean'),
            avg_frequency=('frequency', 'mean'),
            avg_monetary=('monetary', 'mean')
        ).style.format({'customers': "{:,.0f}", 'avg_recency': "{:,.1f} days",
                          'avg_frequency': "{:,.1f}x", 'avg_monetary': "MMK {:,.0f}"})
        st.dataframe(summary, use_container_width=True)

    else:
        st.info("ℹ️ Select at least 2 dimensions for clustering.")
else:
    st.info("🤖 `customer_payments_view` is required for segmentation.")

st.markdown("---")


# ============================== 10. Product Performance ======================
st.header("🏅 Product Performance")

if not order_details.empty and not products.empty and 'quantity' in order_details.columns:
    prod_perf = order_details.groupby('product_id').agg(
        qty_sold=('quantity', 'sum'),
        revenue=('item_revenue', 'sum')
    ).reset_index()

    prod_perf = prod_perf.merge(products[['product_id', 'name']], on='product_id', how='left')

    n_top = st.slider("Show Top N Products", 3, min(20, len(prod_perf) or 21), 10, key="top_n_prod")

    tab_qty, tab_rev = st.tabs(["By Quantity Sold", "By Revenue Generated"])

    with tab_qty:
        top_qty = prod_perf.sort_values('qty_sold', ascending=False).head(n_top)
        fig_q = px.bar(top_qty, x='name', y='qty_sold', color='qty_sold',
                       color_continuous_scale='Plasma', title=f"Top {n_top} Products by Quantity",
                       labels={'name': 'Product', 'qty_sold': 'Total Quantity Sold'})
        st.plotly_chart(fig_q, use_container_width=True)

    with tab_rev:
        top_rev = prod_perf.sort_values('revenue', ascending=False).head(n_top)
        fig_r = px.bar(top_rev, x='name', y='revenue', color='revenue',
                       color_continuous_scale='Viridis', title=f"Top {n_top} Products by Revenue",
                       labels={'name': 'Product', 'revenue': 'Total Revenue (MMK)'})
        st.plotly_chart(fig_r, use_container_width=True)
else:
    st.info("🏅 `order_details` and `products` tables are required for product performance analysis.")

st.markdown("---")

# ============================== 11. Sales Funnel =============================
st.header("🔻 Sales Funnel")

if not deals.empty and 'stage' in deals.columns and 'value' in deals.columns:
    default_stages = ["Prospect", "Qualified", "Proposal", "Negotiation", "Won", "Lost"]
    # Filter to only stages that actually exist in the data to avoid errors
    existing_stages = [s for s in default_stages if s in deals['stage'].unique()]
    
    stage_counts = deals['stage'].value_counts().reindex(existing_stages).reset_index()
    stage_counts.columns = ['stage', 'count']

    fig_funnel = go.Figure(go.Funnel(
        y=stage_counts['stage'],
        x=stage_counts['count'],
        textinfo="value+percent initial"
    ))
    fig_funnel.update_layout(title_text="Deal Stage Funnel")
    st.plotly_chart(fig_funnel, use_container_width=True)
else:
    st.info("🔻 `deals` table with `stage` and `value` columns required for funnel visualization.")

st.markdown("---")

# ============================== 12. Advanced Data Tools ======================
st.header("🛠️ Advanced Data Tools")

st.sidebar.header("2. CSV files")
for tbl in CSV_TABLES:
    up = st.sidebar.file_uploader(f"Upload {tbl}.csv", type="csv", key=f"csv_{tbl}")
    if up:
        try:
            csv_data[tbl] = pd.read_csv(up)
            st.sidebar.success(f"{tbl}.csv loaded ({len(csv_data[tbl])} rows)")
        except Exception as e:
            st.sidebar.error(f"Error in {tbl}.csv: {e}")
# ------------------------ PATCH 9  conditional writer -----------------------
if data_source == "Database":
 with st.expander("Upload CSV Data to Database"):
    REQUIRED_TABLES = {
    "payments":        ["payment_id", "customer_id", "amount", "payment_date"],
    "deals":           ["deal_id", "customer_id", "stage", "value", "created_at"],
    "activities":      ["activity_id", "customer_id", "activity_date", "activity_type"],
    "leads":           ["lead_id", "customer_id", "source", "created_at"],
    "users":           ["user_id", "name", "email", "password", "phone", "user_type"],
    "products":        ["product_id", "name", "description", "price", "stock_quantity", "seller_id"],
    "orders":          ["order_id", "customer_id", "order_date", "status"],
    "order_details":   ["order_detail_id", "order_id", "product_id", "quantity", "price_each"]
}
   
    def check_columns(df, table_name):
        reqs = REQUIRED_TABLES.get(table_name)
        if not reqs: return # No requirements defined for this table
        missing = [c for c in reqs if c not in df.columns]
        if missing:
            raise ValueError(f"`{table_name}` is missing required columns: {missing}")
    def write_table(df, table_name, if_exists="append"):
        db_engine = get_engine()
        if db_engine is None:
            st.error("Connect to DB first before ingesting data."); return
        try:
            check_columns(df, table_name)
            df.to_sql(table_name, db_engine, if_exists=if_exists, index=False)
            st.success(f"✅ Ingested {len(df):,} rows into `{table_name}`.")
            st.cache_data.clear() # Clear cache to force a reload
        except Exception as e:
            st.error(f"❌ Failed to write to `{table_name}`. Reason: {e}")
    for table in ["payments", "deals", "activities", "leads"]:
        up = st.file_uploader(f"Upload `{table}.csv`", type="csv", key=f"upload_{table}")
        if up:
            try:
                df_csv = pd.read_csv(up)
                write_table(df_csv, table)
            except Exception as e:
                st.error(f"Could not process CSV for `{table}`. Error: {e}")
with st.expander("Generate Dummy Data"):
    st.write("Use this tool to populate your database with sample data if the tables are empty.")
    def _make_dummy_rows(db_engine):
        if not db_engine:
            st.error("Connect to a database first."); return
        inspector = sa.inspect(db_engine)
        if not inspector.has_table("payments"):
            pay = pd.DataFrame({
                "payment_id"  : range(1, 11), "customer_id" : np.random.randint(1, 6, 10),
                "amount"      : np.random.randint(5000, 50000, 10),
                "payment_date": pd.to_datetime(pd.date_range(end=dt.date.today(), periods=10, freq="M"))
            })
            pay.to_sql("payments", db_engine, index=False)
            st.write("✓ Created `payments` table.")
        if not inspector.has_table("deals"):
            deals_df = pd.DataFrame({
                "deal_id"     : range(1, 11), "customer_id" : np.random.randint(1, 6, 10),
                "stage"       : np.random.choice(["Prospect", "Qualified", "Proposal", "Won"], 10),
                "value"       : np.random.randint(10000, 100000, 10),
                "created_at"  : pd.to_datetime(pd.date_range(end=dt.date.today(), periods=10, freq="7D"))
            })
            deals_df.to_sql("deals", db_engine, index=False)
            st.write("✓ Created `deals` table.")
        if not inspector.has_table("activities"):
            acts = pd.DataFrame({
                "activity_id"   : range(1, 21), "customer_id"   : np.random.randint(1, 6, 20),
                "activity_date" : pd.to_datetime(pd.date_range(end=dt.date.today(), periods=20, freq="3D")),
                "activity_type" : np.random.choice(["Call", "Email", "Demo", "Meeting"], 20),
                "notes"         : [""]*20
            })
            acts.to_sql("activities", db_engine, index=False)
            st.write("✓ Created `activities` table.")
        st.success("Dummy data generation complete. Refresh the page to see changes.")
        st.cache_data.clear()
    if st.button("Generate Data"):
        _make_dummy_rows(engine)
