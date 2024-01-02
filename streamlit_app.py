import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv("Groceries_dataset.csv")

df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

df["month"] = df['Date'].dt.month
df["day"] = df['Date'].dt.weekday

df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], inplace=True)
df["day"].replace([i for i in range(6 + 1)], ['Monday', 'Tuesady', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)

st.title("Association rules pada data belanja dengan algoritma Apriori")

def get_data(month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if not filtered.empty else "No result"

def user_input_feature():
    item = st.selectbox("Item", ['tropical fruit', 'whole milk', 'pip fruit', 'other vegetables', 'rolls/buns', 'pot plants', 'citrus fruit', 'beef', 'frankfurter', 'chicken',
                                 'butter', 'fruit/vegetable juice', 'packaged fruit/vegetables', 'chocolate', 'specialty bar', 'butter milk', 'bottled water', 'yogurt', 'sausage',
                                 'brown bread', 'hamburger meat', 'root vegetables', 'pork', 'pastry', 'canned beer', 'berries', 'coffee', 'misc. beverages', 'ham', 'turkey',
                                 'curd cheese', 'red/blush wine', 'frozen potato products', 'flour', 'sugar', 'frozen meals', 'herbs', 'soda', 'detergent', 'grapes', 'processed cheese',
                                 'fish', 'sparkling wine', 'newspapers', 'curd', 'pasta', 'popcorn', 'finished products', 'beverages', 'bottled beer', 'dessert', 'dog food',
                                 'specialty chocolate', 'condensed milk', 'cleaner', 'white wine', 'meat', 'ice cream', 'hard cheese', 'cream cheese', 'liquor', 'pickled vegetables',
                                 'liquor (appetizer)', 'UHT-milk', 'candy', 'onions', 'hair spray', 'photo/film', 'domestic eggs', 'margarine', 'shopping bags', 'salt', 'oil',
                                 'whipped/sour cream', 'frozen vegetables', 'sliced cheese', 'dish cleaner', 'baking powder', 'specialty cheese', 'salty snack', 'Instant food products',
                                 'pet care', 'white bread', 'female sanitary products', 'cling film/bags', 'soap', 'frozen chicken', 'house keeping products', 'spread cheese',
                                 'decalcifier', 'frozen dessert', 'vinegar', 'nuts/prunes', 'potato products', 'frozen fish', 'hygiene articles', 'artif. sweetener', 'light bulbs',
                                 'canned vegetables', 'chewing gum', 'canned fish', 'cookware', 'semi-finished bread', 'cat food', 'bathroom cleaner', 'prosecco', 'liver loaf',
                                 'zwieback', 'canned fruit', 'frozen fruits', 'brandy', 'baby cosmetics', 'spices', 'napkins', 'waffles', 'sauces', 'rum', 'chocolate marshmallow',
                                 'long life bakery product', 'bags', 'sweet spreads', 'soups', 'mustard', 'specialty fat', 'instant coffee', 'snack products', 'organic sausage',
                                 'soft cheese', 'mayonnaise', 'dental care', 'roll products', 'kitchen towels', 'flower soil/fertilizer', 'cereals', 'meat spreads', 'dishes',
                                 'male cosmetics', 'candles', 'whisky', 'tidbits', 'cooking chocolate', 'seasonal products', 'liqueur', 'abrasive cleaner', 'syrup', 'ketchup',
                                 'cream', 'skin care', 'rubbing alcohol', 'nut snack', 'cocoa drinks', 'softener', 'organic products', 'cake bar', 'honey', 'jam', 'kitchen utensil',
                                 'flower (seeds)', 'rice', 'tea', 'salad dressing', 'specialty vegetables', 'pudding powder', 'ready soups', 'make up remover', 'toilet cleaner', 'preservation products'])
    month = st.select_slider("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.select_slider("Day", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], value="Mon")

    return item, month, day

item, month, day = user_input_feature()

data = get_data(month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

if type(data) != type("No Result"):
    item_count = df.groupby(["Member_number", "itemDescription"])["itemDescription"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Member_number', columns='itemDescription', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_treshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_treshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    filtered_data = data.loc[data["antecedents"] == item_antecedents]

    if not filtered_data.empty:
        return list(filtered_data.iloc[0, :])
    else:
        return []

if type(data) != type("No Result!"):
    st.markdown("Hasil Rekomendasi : ")
    result = return_item_df(item)
    if result :
        st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_df(item)[1]}** secara bersamaan")
    else:
        st.warning("Tidak ditemukan rekomendasi untuk item yang dipilih")
