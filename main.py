
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load dataset
df = pd.read_csv("Groceries_dataset.csv")

# Step 2: Clean dataset
df = df.drop_duplicates()
df = df.dropna()

print("Dataset Shape After Cleaning:", df.shape)

# Step 3: Create transactions
transactions = df.groupby(
    ["Member_number", "Date"]
)["itemDescription"].apply(list)

basket_list = transactions.tolist()
all_items = sorted(df["itemDescription"].unique())

# Step 4: Convert transactions into basket format
basket_data = []

for basket in basket_list:
    basket_data.append([
        1 if item in basket else 0
        for item in all_items
    ])

basket_df = pd.DataFrame(
    basket_data,
    columns=all_items,
    dtype=bool
)

print("Total Transactions:", len(basket_df))
print("Total Unique Products:", len(all_items))

# Step 5: Apply Apriori algorithm
frequent_items = apriori(
    basket_df,
    min_support=0.01,
    use_colnames=True
)

print("\nTop 10 Frequent Itemsets:")
print(
    frequent_items.sort_values(
        "support", ascending=False
    ).head(10).to_string(index=False)
)

# Step 6: Generate association rules
if not frequent_items.empty:
    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.05
    )

    if not rules.empty:
        rules = rules.sort_values(
            "lift", ascending=False
        )

        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )

        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )

        print("\nTop 10 Association Rules:")
        print(
            rules[
                ["antecedents", "consequents",
                 "support", "confidence", "lift"]
            ].head(10).round(3).to_string(index=False)
        )
    else:
        print("\nNo association rules found.")
else:
    print("\nNo frequent itemsets found.")

import matplotlib.pyplot as plt

# Top 10 most frequently purchased products
top_products = (
    df["itemDescription"]
    .value_counts()
    .head(10)
)

# Create bar chart
plt.figure(figsize=(10, 6))
top_products.sort_values().plot(kind="barh")

plt.title("Top 10 Most Frequently Purchased Products")
plt.xlabel("Number of Purchases")
plt.ylabel("Product Name")
plt.tight_layout()

# Save chart as an image
plt.savefig("top_products_chart.png")
plt.show()
