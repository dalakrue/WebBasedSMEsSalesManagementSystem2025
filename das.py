import os, io, sys, datetime as dt
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import sqlalchemy as sa
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import xlsxwriter
from fpdf import FPDF
import requests
from functools import lru_cache
# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "access" not in st.session_state:
    st.session_state["access"] = None    # <-- add this
if "role" not in st.session_state:
    st.session_state["role"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

# -------------------- Role-based Analysis --------------------
st.header("📊 Role-based Dashboard Analysis")

# Simulate getting user role from API (you already have access token)
def get_user_role(username: str, access_token: str) -> str:
    try:
        resp = requests.get(f"{DJANGO_BASE}/api/user_role/", headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("role", "buyer")
    except:
        pass
    return "buyer"

if st.session_state["access"]:
    user_role_val = get_user_role(st.session_state['username'], st.session_state['access'])
else:
    user_role_val = "buyer"  # default fallback

st.sidebar.subheader("User Role")
st.sidebar.write(f"**{user_role_val.capitalize()}**")

# Analysis Button
if st.button("Run Analysis"):

    if user_role_val.lower() == "buyer":
        st.subheader("📈 Buyer Dashboard")
        # Show Buyer-specific charts (limited)
        if not df_products.empty:
            top_quantity = df_products.sort_values("quantity", ascending=False).head(10)
            fig_buyer = px.bar(top_quantity, x='name', y='quantity', 
                               title="Top 10 Products by Stock Quantity (Buyer View)",
                               color='quantity', color_continuous_scale='Viridis')
            st.plotly_chart(fig_buyer, use_container_width=True)
        if not payments.empty and 'amount' in payments.columns:
            total_spent = payments['amount'].sum()
            st.metric("Total Spent", f"MMK {total_spent:,.0f}")
        st.info("Buyer sees only product availability and personal purchases.")

    elif user_role_val.lower() in ["admin", "seller"]:
        st.subheader("📈 Admin/Seller Dashboard")
        # Show full analysis (reuse your existing dashboard KPIs & charts)
        st.markdown("### 🔑 Key KPIs")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Leads This Month", f"{lead_this_month:,}", delta=lead_delta)
        kpi_cols[1].metric("Revenue This Month", f"MMK {revenue_this_month:,.0f}", delta=rev_delta)
        kpi_cols[2].metric("Deals in Pipeline", f"{pipeline_deals:,}")
        kpi_cols[3].metric("Active Customers", f"{total_customers:,}")

        st.markdown("### 📦 Product Overview")
        st.dataframe(df_products, use_container_width=True)

        # Top quantity chart
        if 'name' in df_products.columns and 'quantity' in df_products.columns:
            top_quantity = df_products.sort_values("quantity", ascending=False).head(10)
            fig_quantity = px.bar(top_quantity, x='name', y='quantity', 
                                 title="Top 10 Products by Stock Quantity",
                                 color='quantity', color_continuous_scale='Viridis')
            st.plotly_chart(fig_quantity, use_container_width=True)

        # Revenue trend
        if not payments.empty and 'payment_date' in payments.columns and 'amount' in payments.columns:
            payments['month'] = payments['payment_date'].dt.to_period('M').astype(str)
            monthly_rev = payments.groupby('month')['amount'].sum().reset_index()
            fig_month = px.line(monthly_rev, x='month', y='amount', title="Actual Monthly Revenue",
                                markers=True, labels={'amount': 'Revenue (MMK)'})
            st.plotly_chart(fig_month, use_container_width=True)
        
        # Buyer/Seller segmentation chart
        if not customer_payments.empty:
            seg_data = customer_payments.groupby('customer_id').agg(
                recency_days=('payment_date', lambda x: (pd.Timestamp.now() - x.max()).days),
                frequency=('payment_id', 'count'),
                monetary=('amount', 'sum')
            ).reset_index()
            fig_seg = px.scatter(seg_data, x='frequency', y='monetary', color='recency_days',
                                 title="Customer Segmentation (Admin/Seller View)")
            st.plotly_chart(fig_seg, use_container_width=True)

    else:
        st.warning("Unknown user role. Showing limited buyer view.")
        st.info("Buyer sees only product stock levels and purchases.")

st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;}
.chat-message.user {
    background-color: #2b313e;}
.chat-message.assistant {
    background-color: #475063;}
</style>
""", unsafe_allow_html=True)
def check_columns(df, table_name):
 pass    
def write_table(df, table_name, if_exists="append"):
    db_engine = get_engine()
    if db_engine is None:
        st.error("Connect to DB first.")
        return
    try:
        check_columns(df, table_name)
        df.to_sql(table_name, db_engine, if_exists=if_exists, index=False)
        st.success(f"✅ Ingested {len(df):,} rows into `{table_name}`.")
        st.cache_data.clear()
    except Exception as e:
        st.sidebar.error(f"Error in {table_name}.csv: {e}")
st.set_page_config(page_title="📊 SME Sales Management Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")
DJANGO_BASE = os.getenv("DJANGO_BASE", "http://127.0.0.1:8000")
if "access" not in st.session_state:
    st.session_state["access"] = None
if "sqlite_path" not in st.session_state:
    st.session_state["sqlite_path"] = "db.sqlite3"  
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if st.session_state["access"] is None:
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            login_url = f"{DJANGO_BASE}/api/token/"
            try:
                resp = requests.post(login_url, json={"username": username, "password": password}, timeout=20)
                resp.raise_for_status()
                tokens = resp.json()
                st.session_state["access"] = tokens["access"]
                st.session_state["username"] = username
                st.success(f"✅ Logged in as {username}")
                st.rerun()
            except requests.HTTPError:
                st.error("❌ Login failed. Check username/password")
else:
    st.sidebar.write(f"Logged in as **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.pop("access", None)
        st.session_state.pop("username", None)
        st.rerun()
df_products = pd.DataFrame()
orders = pd.DataFrame()
payments = pd.DataFrame()
users = pd.DataFrame()
order_details = pd.DataFrame()
leads = pd.DataFrame()
deals = pd.DataFrame()
activities = pd.DataFrame()
customer_payments = pd.DataFrame()
st.sidebar.title("Configuration")
st.sidebar.header("Choose Data Source")
data_source = st.sidebar.radio(
    "Data source",
    ("API", "Database"),
    horizontal=True,
    key="data_source_mode")
_DB_KEY = "db_engine"
def get_engine():
    return st.session_state.get(_DB_KEY)
def disconnect_db():
    engine = st.session_state.pop(_DB_KEY, None)
    if engine is not None:
        try: 
            engine.dispose()
        except Exception: 
            pass
    st.success("🔌 Disconnected from DB")
def connect_db(db_type: Optional[str] = None):
    if _DB_KEY in st.session_state:
        st.info("Already connected.")
        return st.session_state[_DB_KEY]
    db_type = db_type or st.session_state.get("db_type", "MySQL")
    try:
        if db_type == "SQLite":
            sqlite_path = st.session_state.get("sqlite_path", "") or ""
            if not sqlite_path:
                st.error("Provide a SQLite file path (type it or upload a .db file).")
                return None
            if not os.path.exists(sqlite_path):
                st.error(f"SQLite file not found: {sqlite_path}.")
                return None
            uri = f"sqlite+pysqlite:///{sqlite_path}"
            engine = sa.create_engine(uri, connect_args={"check_same_thread": False})
        else:
            host = st.session_state.get("mysql_host", "localhost")
            port = st.session_state.get("mysql_port", "3306")
            db_name = st.session_state.get("mysql_db", "sample_data")
            user = st.session_state.get("mysql_user", "root")
            password = st.session_state.get("mysql_password", "root")
            if not all([host, port, db_name, user]):
                st.warning("Please fill in all database connection details.")
                return None
            uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
            engine = sa.create_engine(uri, pool_pre_ping=True, pool_recycle=280)
        with engine.connect() as _:
            pass
        st.session_state[_DB_KEY] = engine
        st.success(f"✅ Connected to {db_type}")
        return engine
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
if data_source == "API":
    st.sidebar.subheader("Using Django API")
    if st.session_state["access"]:
        headers = {"Authorization": f"Bearer {st.session_state['access']}"}
        try:
            products_url = f"{DJANGO_BASE}/api/products/"
            resp = requests.get(products_url, headers=headers, timeout=20)
            if resp.status_code == 200 and resp.text and resp.text.strip():
                products_api = resp.json()
            else:
                products_api = []
            if products_api:
                df_products = pd.DataFrame(products_api)
                if 'id' in df_products.columns and 'product_id' not in df_products.columns:
                    df_products = df_products.rename(columns={'id': 'product_id'})
                st.success("✅ Products loaded from API")
            else:
                st.info("No products available from API")
        except Exception as e:
            st.error(f"❌ Could not fetch products from API: {e}")        
    else:
        st.error("Please login to use API data source")
elif data_source == "Database":
    st.sidebar.subheader("Database Connection")
    db_type = st.sidebar.radio("DB Type", ("MySQL", "SQLite"), horizontal=True, key="db_type")
    if db_type == "MySQL":
        st.sidebar.text_input("Host", value=st.session_state.get("mysql_host", "localhost"), key="mysql_host")
        st.sidebar.text_input("Port", value=st.session_state.get("mysql_port", "3306"), key="mysql_port")
        st.sidebar.text_input("Database Name", value=st.session_state.get("mysql_db", "sample_data"), key="mysql_db")
        st.sidebar.text_input("User", value=st.session_state.get("mysql_user", "root"), key="mysql_user")
        st.sidebar.text_input("Password", value=st.session_state.get("mysql_password", "root"), type="password", key="mysql_password")
    else:  # SQLite
        st.sidebar.text_input("SQLite file path (or upload below)",
                              value=st.session_state.get("sqlite_path", "sample_data.sqlite3"),
                              key="sqlite_path")
        up_db = st.sidebar.file_uploader("Upload a SQLite .db / .sqlite file",
                                         type=["db", "sqlite", "sqlite3"],
                                         key="sqlite_upload")
        if up_db:
            tmp_path = os.path.join(os.getcwd(), f"_uploaded_{up_db.name}")
            with open(tmp_path, "wb") as f: 
                f.write(up_db.getbuffer())
            st.session_state["sqlite_path"] = tmp_path
            st.sidebar.success(f"Loaded {tmp_path}")
    if st.sidebar.button("Connect DB"):
        connect_db(db_type)
    if st.sidebar.button("Disconnect DB"):
        disconnect_db()
    if "db_engine" in st.session_state:
        engine = st.session_state["db_engine"]
        @st.cache_data
        def load_data_from_db(_engine):
            data = {}
            table_names = ["payments", "users", "orders", "order_details", "products", 
                          "leads", "deals", "activities", "customer_payments_view"]
            for table in table_names:
                try:
                    if sa.inspect(_engine).has_table(table):
                        data[table] = pd.read_sql(f"SELECT * FROM {table}", _engine)
                    else:
                        data[table] = pd.DataFrame()
                except Exception as e:
                    st.warning(f"Could not load table '{table}'. Reason: {e}")
                    data[table] = pd.DataFrame()
            return data
        loaded = load_data_from_db(engine)
        payments = loaded.get("payments", pd.DataFrame())
        users = loaded.get("users", pd.DataFrame())
        orders = loaded.get("orders", pd.DataFrame())
        order_details = loaded.get("order_details", pd.DataFrame())
        df_products = loaded.get("products", pd.DataFrame())
        leads = loaded.get("leads", pd.DataFrame())
        deals = loaded.get("deals", pd.DataFrame())
        activities = loaded.get("activities", pd.DataFrame())
        customer_payments = loaded.get("customer_payments_view", pd.DataFrame())
        if 'id' in df_products.columns and 'product_id' not in df_products.columns:
            df_products = df_products.rename(columns={'id': 'product_id'})
st.sidebar.markdown("---")
if data_source == "API":
    st.sidebar.info("📡 Using API Data Source")
    if not df_products.empty:
        st.sidebar.success(f"✅ Loaded {len(df_products)} products from API")
    else:
        st.sidebar.warning("⚠️ No product data from API")
else:
    if "db_engine" in st.session_state:
        st.sidebar.success("🗄️ Using Database Data Source")
        if not df_products.empty:
            st.sidebar.success(f"✅ Loaded {len(df_products)} products from DB")
        else:
            st.sidebar.warning("⚠️ No product data from database")
    else:
        st.sidebar.info("🔌 Not connected to database")
st.title("📊 Advanced Sales Management Dashboard for SMEs")
st.header("💬 SME Chat Assistant")
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask about your sales data..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    def generate_chat_response(prompt, products_df, orders_df, payments_df):
        """Generate responses based on available data"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["product", "item", "stock", "inventory"]):
            if products_df.empty:
                return "I don't have product data available at the moment."
            if "how many products" in prompt_lower:
                return f"You have {len(products_df)} products in total."
            if "low stock" in prompt_lower or "out of stock" in prompt_lower:
                low_stock = products_df[products_df['quantity'] < 10]
                if len(low_stock) > 0:
                    return f"You have {len(low_stock)} products with low stock. Consider restocking soon."
                else:
                    return "All products have sufficient stock levels."
            if "most expensive" in prompt_lower:
                if 'price' in products_df.columns:
                    max_price = products_df.loc[products_df['price'].idxmax()]
                    return f"Your most expensive product is '{max_price['name']}' at MMK {max_price['price']:,.0f}"
            return "I can help with product inventory, pricing, and stock levels. Ask me about specific products!"
        elif any(word in prompt_lower for word in ["sale", "revenue", "income", "profit"]):
            if not payments_df.empty and 'amount' in payments_df.columns:
                total_revenue = payments_df['amount'].sum()
                return f"Your total revenue is MMK {total_revenue:,.0f}"
            return "Sales data is not available at the moment."
        elif any(word in prompt_lower for word in ["order", "purchase", "transaction"]):
            if not orders_df.empty:
                total_orders = len(orders_df)
                return f"You have {total_orders} total orders in the system."
            return "Order data is not available at the moment."
        elif any(word in prompt_lower for word in ["help", "what can you do", "assist"]):
            return """I can help you with:
- Product information and inventory levels
- Sales and revenue data
- Order statistics
- Stock alerts and recommendations
Ask me anything about your business data!"""
        return "I'm here to help with your sales data analysis. Ask me about products, sales, or inventory!"
    response = generate_chat_response(prompt, df_products, orders, payments)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
if st.button("Clear Chat"):
    st.session_state.chat_messages = []
    st.rerun()
if not df_products.empty:
    st.header("📦 Products Overview")
    st.dataframe(df_products, use_container_width=True)
    st.subheader("Product Analytics")
    if 'name' in df_products.columns and 'quantity' in df_products.columns:
        top_quantity = df_products.sort_values("quantity", ascending=False).head(10)
        fig_quantity = px.bar(top_quantity, x='name', y='quantity', 
                             title="Top 10 Products by Stock Quantity",
                             color='quantity', color_continuous_scale='Viridis')
        st.plotly_chart(fig_quantity, use_container_width=True)
    if 'name' in df_products.columns and 'price' in df_products.columns:
        fig_price = px.bar(df_products.sort_values("price", ascending=False), 
                          x='name', y='price', 
                          title="Product Price Distribution",
                          color='price', color_continuous_scale='Plasma')
        st.plotly_chart(fig_price, use_container_width=True)
    if 'category' in df_products.columns:
        category_counts = df_products['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        fig_category = px.pie(category_counts, values='count', names='category',
                             title="Products by Category")
        st.plotly_chart(fig_category, use_container_width=True)
    if 'approved' in df_products.columns:
        approved_counts = df_products['approved'].value_counts().reset_index()
        approved_counts.columns = ['approved', 'count']
        fig_approved = px.pie(approved_counts, values='count', names='approved',
                             title="Approval Status of Products")
        st.plotly_chart(fig_approved, use_container_width=True)
else:
    st.info("No product data available")
def merge_price_into_order_details(od: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    if od.empty: 
        st.warning("order_details is empty; skipping price merge.")
        return od
    if 'price_each' in od.columns: 
        return od
    possible = ['price','unit_price','unitPrice','unitprice','price_each','price_each_usd']
    price_col = next((c for c in possible if c in prod.columns), None)
    if price_col is None:
        st.warning("⚠️ No price column in products; setting price_each to NaN.")
        od['price_each'] = np.nan
        return od
    if 'product_id' in od.columns and 'product_id' in prod.columns:
        od['product_id'] = od['product_id'].astype(str)
        prod['product_id'] = prod['product_id'].astype(str)
    else:
        st.warning("product_id missing in order_details or products.")
        od['price_each'] = np.nan
        return od
    od = od.merge(prod[['product_id', price_col]].rename(columns={price_col: 'price_each'}),
                  on='product_id', how='left')
    if od['price_each'].isna().all():
        st.warning("⚠️ After merge, all price_each values are NaN.")
    else:
        st.success(f"✅ price_each merged from products['{price_col}'].")
    return od
def safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            st.warning(f"⚠️ Failed to convert {col} to datetime: {e}")
    else:
        st.warning(f"⚠️ Column `{col}` not found.")
    return df
st.header("🔄 Data Pre-processing Status")
with st.expander("Show Details"):
    if not payments.empty:
        payments = safe_to_datetime(payments, 'payment_date')
    if not orders.empty:
        orders = safe_to_datetime(orders, 'order_date')
    if not customer_payments.empty:
        customer_payments = safe_to_datetime(customer_payments, 'payment_date')
    if not leads.empty:
        leads = safe_to_datetime(leads, 'created_date')
    if not deals.empty:
        deals = safe_to_datetime(deals, 'expected_close_date')
    if not activities.empty:
        activities = safe_to_datetime(activities, 'activity_date')
    st.write("✓ Dates standardized.")  
    for df in [orders, order_details, df_products, users, payments, customer_payments]:
        if not df.empty:
            for col in ['order_id','user_id','product_id','customer_id']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
    st.write("✓ Key IDs coerced to string.")
    if not order_details.empty and not df_products.empty:
        order_details = merge_price_into_order_details(order_details, df_products)
        if 'quantity' in order_details.columns and 'price_each' in order_details.columns:
            order_details['item_revenue'] = pd.to_numeric(order_details['quantity'], errors='coerce') * \
                                            pd.to_numeric(order_details['price_each'], errors='coerce')
            st.write("✓ item_revenue computed.")
        else:
            order_details['item_revenue'] = np.nan
    else:
        st.write("⚠️ Could not compute item_revenue - missing order_details or products")
        st.header("📋 Published Table (Everyone Can See)")

if "db_engine" in st.session_state:
    engine = st.session_state["db_engine"]
    try:
        # Example: published table named `published_data`
        published_df = pd.read_sql("SELECT * FROM published_data", engine)
        if not published_df.empty:
            st.dataframe(published_df, use_container_width=True)
            
            # Optional: example chart from this table
            if 'amount' in published_df.columns and 'created_date' in published_df.columns:
                published_df['created_date'] = pd.to_datetime(published_df['created_date'], errors='coerce')
                monthly_pub = published_df.groupby(published_df['created_date'].dt.to_period('M'))['amount'].sum().reset_index()
                fig_pub = px.line(monthly_pub, x='created_date', y='amount', title="Published Data Revenue Trend")
                st.plotly_chart(fig_pub, use_container_width=True)
        else:
            st.info("No data available in the published table.")
    except Exception as e:
        st.error(f"Failed to load published table: {e}")
else:
    st.info("🔌 Connect to database to view published data.")

st.header("🏆 Key Performance Indicators")
total_revenue = payments['amount'].sum() if not payments.empty and 'amount' in payments.columns else 0
total_orders = orders['order_id'].nunique() if not orders.empty and 'order_id' in orders.columns else 0
total_customers = users[users['user_type'] == 'customer']['user_id'].nunique() if not users.empty and 'user_type' in users.columns else 0
total_products = 0
if not df_products.empty:
    product_id_col = "product_id" if "product_id" in df_products.columns else "id" if "id" in df_products.columns else None
    if product_id_col:
        total_products = df_products[product_id_col].nunique()
revenue_this_month, rev_delta = (0, None)
if not payments.empty and 'payment_date' in payments.columns:
    current_month_mask = payments['payment_date'].dt.to_period('M') == pd.Timestamp.today().to_period('M')
    last_month_mask = payments['payment_date'].dt.to_period('M') == (pd.Timestamp.today() - pd.DateOffset(months=1)).to_period('M')
    revenue_this_month = payments.loc[current_month_mask, 'amount'].sum()
    revenue_last_month = payments.loc[last_month_mask, 'amount'].sum()
    if revenue_last_month > 0:
        rev_delta = f"{((revenue_this_month - revenue_last_month) / revenue_last_month * 100):.0f}%"
lead_this_month, lead_delta = (0, None)
if not leads.empty and 'created_date' in leads.columns:
    current_month_mask = leads['created_date'].dt.to_period('M') == pd.Timestamp.today().to_period('M')
    last_month_mask = leads['created_date'].dt.to_period('M') == (pd.Timestamp.today() - pd.DateOffset(months=1)).to_period('M')
    lead_this_month = current_month_mask.sum()
    lead_last_month = last_month_mask.sum()
    if lead_last_month > 0:
        lead_delta = f"{(lead_this_month - lead_last_month) / lead_last_month * 100:.0f}% vs last month"
pipeline_deals = deals[deals['stage'] != 'Closed Won'].shape[0] if not deals.empty and 'stage' in deals.columns else 0
kpi_cols = st.columns(4)
kpi_cols[0].metric("Leads This Month", f"{lead_this_month:,}", delta=lead_delta)
kpi_cols[1].metric("Revenue This Month", f"MMK {revenue_this_month:,.0f}", delta=rev_delta)
kpi_cols[2].metric("Deals in Pipeline", f"{pipeline_deals:,}")
kpi_cols[3].metric("Active Customers", f"{total_customers:,}")
st.markdown("<br><br>", unsafe_allow_html=True)
glob_cols = st.columns(4)
glob_cols[0].metric("💰 Total Lifetime Revenue", f"MMK {total_revenue:,.0f}")
glob_cols[1].metric("🛒 Total Orders", f"{total_orders:,}")
glob_cols[2].metric("📦 Total Products", f"{total_products:,}")
glob_cols[3].metric("👥 Total Customers", f"{total_customers:,}")
st.header("📈 Revenue Analytics")
if not payments.empty and 'payment_date' in payments.columns and 'amount' in payments.columns:
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
st.header("🎯 Target vs Actual")
def kpi_leads_vs_target(leads_df, annual_target: int):
    """
    Returns:
        actual: number of leads this year
        target: annual target
        pct: achievement percentage
    """
    if 'created_date' not in leads_df.columns:
        return 0, annual_target, 0
    leads_df['created_date'] = pd.to_datetime(leads_df['created_date'], errors='coerce')
    this_year = pd.Timestamp.now().year
    actual = leads_df[leads_df['created_date'].dt.year == this_year].shape[0]
    pct = (actual / annual_target * 100) if annual_target else 0
    return actual, annual_target, pct
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
def _coerce_key_cols(df_left, df_right, left_key='customer_id', right_key='user_id', dtype='str'):
    df_left[left_key] = df_left[left_key].astype(dtype)
    df_right[right_key] = df_right[right_key].astype(dtype)
    return df_left, df_right
if not customer_payments.empty and 'customer_id' in customer_payments.columns:
    seg_data = customer_payments.groupby('customer_id').agg(
        recency_days=(
            'payment_date',
            lambda x: (pd.Timestamp.now(tz=x.dt.tz) - x.max()).days if x.dt.tz is not None
                      else (pd.Timestamp.now() - x.max()).days),
        frequency=('payment_id', 'count'),
        monetary=('amount', 'sum')
    ).reset_index()
    use_cols = st.multiselect("Select features for clustering",
                              ['recency_days','frequency','monetary'],
                              default=['frequency','monetary'])
    if len(use_cols) >= 2:
        X_scaled = StandardScaler().fit_transform(seg_data[use_cols])
        k = st.slider("Number of clusters (K)", 2, 6, 3, key="kmeans_k")
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        seg_data['segment'] = km.fit_predict(X_scaled)
        if not users.empty:
            seg_data, users = _coerce_key_cols(seg_data, users, 'customer_id', 'user_id', 'str')
            seg_data = (seg_data.merge(users[['user_id','name']],
                                       left_on='customer_id', right_on='user_id', how='left')
                               .rename(columns={'name':'customer_name'}))
        fig_seg = px.scatter(
            seg_data, 
            x=use_cols[0], 
            y=use_cols[1],
            color=seg_data['segment'].astype(str),
            size='monetary' if 'monetary' in seg_data.columns else None,
            hover_data=['customer_name', 'recency_days', 'frequency', 'monetary'],
            title="Customer Segments" )
        st.plotly_chart(fig_seg, use_container_width=True)
        st.subheader("Segment Summary")
        st.dataframe(
            seg_data.groupby('segment').agg(
                customers=('customer_id','count'),
                avg_recency=('recency_days','mean'),
                avg_frequency=('frequency','mean'),
                avg_monetary=('monetary','mean')
            ).style.format({'customers':"{:,.0f}",'avg_recency':"{:,.1f} days",
                            'avg_frequency':"{:,.1f}x",'avg_monetary':"MMK {:,.0f}"}),
            use_container_width=True )
    else:
        st.info("ℹ️ Select at least 2 dimensions for clustering.")
else:
    st.info("🤖 `customer_payments_view` is required for segmentation.")
