import streamlit as st
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from databricks import sql
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define the HTML template for the front end with custom styles
html_temp = """
    <style>
    body {
        background-color: #F6F6F6;  /* Set the background color to gold */
        font-family: Arial, sans-serif;  /* Set the font family */
    }
    h1 {
        color: #1E56A0;  /* Set the h1 (title) color to dark gray */
    }
    </style>
    <div style="background-color:#F6F6F6;padding:13px;text-align:center;">
    <h1 style="color: #163172;">Food Recommendation System</h1>
    </div>
"""

# Set the page configuration
st.set_page_config(
    page_title="Food Recommendation System",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon=None,
)

# Display the front end aspect
st.markdown(html_temp, unsafe_allow_html=True)

user_input = st.text_input("Enter ingredients separated by commas:")

df = pd.read_csv('./Ingredients_Flattened.csv')

df = df[['Recipe_Name', 'Ingredients']]

# User inputs for Databricks credentials
st.sidebar.header('Databricks Connection Settings')
# server_hostname = st.sidebar.text_input('Databricks Hostname (without https://)', 'your-databricks-hostname')
# shttp_path = st.sidebar.text_input('Databricks HTTP Path', 'your-http-path')
access_token = st.sidebar.text_input('Databricks Token', 'your-access-token', type='password')

def recommend_dishes(ingredients, df, num_recommendations=5):
    """
    Recommend recipes based on matching ingredients.
 
    Args:
        df (pd.DataFrame): DataFrame containing recipe data.
        ingredients (list): List of ingredients to match.
        num_recommendations (int): Number of recommendations to return.
 
    Returns:
        recommended_recipes (pd.DataFrame): Top matching recipes.
    """
    # Split ingredients into individual ingredients
    df['Ingredient_List'] = df['Ingredients'].str.split(', ')
 
    # Calculate the number of matching ingredients for each recipe
    df['Match_Count'] = df['Ingredient_List'].apply(lambda x: sum(i in ingredients for i in x))
 
    # Calculate the total number of ingredients in each recipe
    df['Total_Ingredients'] = df['Ingredient_List'].apply(len)
 
    # Calculate the percentage of matching ingredients
    df['Match_Percentage'] = (df['Match_Count'] / df['Total_Ingredients']) * 100
 
    # Sort recipes by Match_Count in descending order
    recommended_recipes = df.sort_values(by=['Match_Count', 'Match_Percentage'], ascending=[False, False]).head(num_recommendations)
 
    return recommended_recipes

# def recommend_dishes(user_ingredients, data, top_n=3):
#         # Step 2: Initialize TF-IDF vectorizer
#     vectorizer = TfidfVectorizer()
    
#     # Fit and transform the cleaned ingredients for all recipes into TF-IDF vectors
#     tfidf_matrix = vectorizer.fit_transform(df['Ingredients'])

#     # Clean the user input (assuming it's already cleaned)
#     user_ingredients_cleaned = ' '.join(user_ingredients)
   
#     # Transform user input into the same TF-IDF space as the recipes
#     user_input_tfidf = vectorizer.transform([user_ingredients_cleaned])
   
#     # Calculate cosine similarity between the user input and all recipe vectors
#     cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)
   
#     # Flatten the similarity scores and get the indices of top N similar recipes
#     similar_indices = cosine_sim.flatten().argsort()[-top_n:][::-1]
   
#     # Retrieve the top N recommended recipes
#     recommended_dishes = data.iloc[similar_indices]
   
#     return recommended_dishes[['Recipe_Name', 'Ingredients']]

# def recommend_dishes(data, user_input):
#   # Preprocess user input
#   user_input = user_input.lower()

#   # Calculate the number of matching ingredients
#   # vectorizer = CountVectorizer()
#   # ingredients_matrix = vectorizer.fit_transform(data['Ingredients'])

# # Calculate the TF-IDF vectors for the cleaned ingredients
#   vectorizer = TfidfVectorizer()
#   ingredients_matrix = vectorizer.fit_transform(data['Ingredients'])

#   user_vector = vectorizer.transform([user_input])

#   similarities = cosine_similarity(user_vector, ingredients_matrix)
#   st.write(similarities)
#   # Find dishes with at least `threshold` matching ingredients
#   matching_dishes = [(index, row) for index, row in enumerate(similarities[0]) if row >= 0.3]

#   recommended_dishes = data.iloc[[index for index, _ in matching_dishes]]

#   return recommended_dishes[['Recipe_Name', 'Ingredients']]


if st.button("Recommend"):
  if user_input:
    recommended_dishes = recommend_dishes(user_input, df)
    st.subheader("Recommended Dishes:")

    if not recommended_dishes.empty:
            # Create a dictionary to store whether the ingredients expander is open for each dish
            expanders_open = {}
            for idx, row in recommended_dishes.iterrows():
                title = row['Recipe_Name']
                cleaned_ingredients = row['Ingredients']
                # Create an expander for each dish
                with st.expander(f"{title}", expanded=expanders_open.get(title, False)):
                    # Split the ingredients string at the comma
                    ingredients_list = [ingredient.lstrip("'") for ingredient in cleaned_ingredients.split("', ")]
                    # Remove "for serving" from each ingredient
                    ingredients_list = [ingredient.replace('for serving', '') for ingredient in ingredients_list]
                    # Check if the first ingredient starts with "[" and remove it
                    if ingredients_list[0].startswith("['"):
                        ingredients_list[0] = ingredients_list[0][2:]
                    # Check if the last ingredient ends with ']'
                    if ingredients_list[-1].endswith("']"):
                        ingredients_list[-1] = ingredients_list[-1][:-2]
                    st.markdown('\n'.join([f"- {ingredient}" for ingredient in ingredients_list]))
    else:
      st.write("No recommended dishes found. Please try a different combination of ingredients.")
              

  else:
    st.warning("Please enter ingredients to get recommendations.")

# st.sidebar.header("About This App")
# st.sidebar.info("Welcome to the Food Recommendation System! This web app suggests dishes based on the ingredients you provide.")
# st.sidebar.info("The more ingredients you specify, the more accurate the recommendations will be.")
# st.sidebar.info("To get your recommended dishes, simply enter the ingredients you'd like to use and click on the 'Recommend' button.")
# st.sidebar.info("Example 1: fish, oil, potato, yoghurt, salt, pepper, zucchini")
# st.sidebar.info("Example 2: chicken, salt, rice, tomato, lettuce, pepper, cucumber")
# st.sidebar.info("Be creative!")
# st.sidebar.info("We hope you find a delicious dish to enjoy. Don't forget to rate this app!")

# feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=5,step=1)

# if feedback:
#   st.header("Thank you for rating the app!")


# Connect to Databricks SQL warehouse/cluster
if st.sidebar.button('Connect'):
    try:
        conn = sql.connect(
            server_hostname='adb-7286099022455773.13.azuredatabricks.net',
            http_path='/sql/1.0/warehouses/ccee7adcb4d8c644',
            access_token=access_token
        )
        
        # Sample query to fetch data
        query = "SELECT * FROM test.sales.bronze_sales_orders LIMIT 10"
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()

        # If results are found, display in a DataFrame
        if result:
            df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
            st.write("Query Results:")
            st.dataframe(df)
        
        else:
            st.write("No Data Found")
        

    # Close connection
    # cursor.close()
    #     conn.close()

    except Exception as e:
        st.error(f"Failed to execute query: {e}")
    finally:
        cursor.close()
        conn.close()