
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("Groceries_dataset.csv")
    data = data.drop_duplicates()
    data = data.dropna()
    return data

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛒 Market Basket Analysis")
st.sidebar.caption("Data Mining Mini Project")
st.sidebar.divider()

top_n = st.sidebar.selectbox(
    "Products to display",
    [5, 10, 15, 20],
    index=1
)

min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.005,
    max_value=0.05,
    value=0.01,
    step=0.005
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.05,
    max_value=0.80,
    value=0.05,
    step=0.05
)

# ---------------- HEADER ----------------
st.title("🛒 Market Basket Analysis")
st.write(
    "Discover frequently purchased grocery products "
    "using the Apriori algorithm and association rules."
)

st.caption("Dataset: Grocery Transactions | Algorithm: Apriori")
st.divider()

# ---------------- KEY METRICS ----------------
transaction_groups = df.groupby(
    ["Member_number", "Date"]
)

total_transactions = transaction_groups.ngroups
total_products = df["itemDescription"].nunique()

c1, c2, c3 = st.columns(3)

c1.metric("📋 Cleaned Records", f"{len(df):,}")
c2.metric("🧾 Transactions", f"{total_transactions:,}")
c3.metric("🛍️ Unique Products", f"{total_products:,}")

st.divider()

# ---------------- PRODUCT ANALYSIS ----------------
st.subheader("📊 Most Purchased Products")

product_counts = (
    df["itemDescription"]
    .value_counts()
    .head(top_n)
)

left, right = st.columns([3, 2])

with left:
    fig, ax = plt.subplots(figsize=(9, 5))
    product_counts.sort_values().plot(
        kind="barh",
        ax=ax
    )
    ax.set_xlabel("Number of Purchases")
    ax.set_ylabel("Product")
    ax.set_title(f"Top {top_n} Products")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("#### Product Summary")
    summary = product_counts.rename_axis(
        "Product"
    ).reset_index(name="Purchases")

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Product Data",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="product_frequency.csv",
        mime="text/csv"
    )

# ---------------- TRANSACTION BASKETS ----------------
st.divider()
st.subheader("🔗 Product Association Rules")

with st.spinner("Analyzing product combinations..."):
    baskets = transaction_groups[
        "itemDescription"
    ].apply(list)

    all_items = sorted(df["itemDescription"].unique())

    basket_data = pd.DataFrame(
        [
            [item in basket for item in all_items]
            for basket in baskets
        ],
        columns=all_items,
        dtype=bool
    )

    frequent_items = apriori(
        basket_data,
        min_support=min_support,
        use_colnames=True
    )

    if not frequent_items.empty:
        rules = association_rules(
            frequent_items,
            metric="confidence",
            min_threshold=min_confidence
        )
    else:
        rules = pd.DataFrame()

# ---------------- DISPLAY RULES ----------------
if not rules.empty:
    rules["Products Bought"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )

    rules["Products Associated"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(x))
    )

    rules = rules.sort_values(
        "lift",
        ascending=False
    )

    display_rules = rules[
        [
            "Products Bought",
            "Products Associated",
            "support",
            "confidence",
            "lift"
        ]
    ].head(20).copy()

    display_rules["support"] = (
        display_rules["support"] * 100
    ).round(2)

    display_rules["confidence"] = (
        display_rules["confidence"] * 100
    ).round(2)

    display_rules = display_rules.rename(
        columns={
            "support": "Support (%)",
            "confidence": "Confidence (%)",
            "lift": "Lift"
        }
    )

    st.caption(
        "Support and confidence are shown as percentages. "
        "Rules are sorted by lift, highest first."
    )

    st.dataframe(
        display_rules,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Association Rules",
        data=display_rules.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="association_rules.csv",
        mime="text/csv"
    )

    st.info(
        "Lift above 1 indicates a positive association; "
        "lift below 1 indicates a negative association "
        "relative to independence."
    )
else:
    st.warning(
        "No rules found for these settings. "
        "Try lowering Minimum Support or Minimum Confidence."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Market Basket Analysis | Data Mining Mini Project"
)
