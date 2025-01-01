import pandas as pd
import json
import requests
import matplotlib.pyplot as plt
import streamlit as st

# Firebase URL
FIREBASE_URL = "https://childmonitoring-951c3-default-rtdb.firebaseio.com/Users.json"

# Set up the Streamlit page configuration
st.set_page_config(page_title="Children's Health Monitoring", layout="wide")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Home", "User Data", "Statistics", "Visualizations"])

# Function to retrieve data from Firebase
def get_data_from_firebase():
    response = requests.get(FIREBASE_URL)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} {response.text}")
        return {}

# Retrieve data and process it
bh = pd.read_excel("HFA_Boys.xlsx")
bh['age']= bh['Month']
bh = bh.drop("Month",axis = 1)
print(bh)

data = get_data_from_firebase()
df = pd.DataFrame(data).T
df["age"] = pd.to_numeric(df["age"])
df["height"] = pd.to_numeric(df["height"])
df["weight"] = pd.to_numeric(df["weight"])
df["BMI"] = round((df["weight"] * 10000) / (df["height"] ** 2), 2)

# Pages based on the selection in the sidebar
if page == "Home":
    st.title("Welcome to the Children's Health Monitoring System")
    st.write("This system allows monitoring of children's growth, including their height, weight, and BMI.")
    st.write("Select a page from the sidebar to explore the data and visualizations.")
    
elif page == "User Data":
    st.title("User Data")
    st.write("Below is the user data fetched from the Firebase database.")
    st.write(df)

elif page == "Statistics":
    st.title("Statistics of Boys and Girls")
    
    # Separate data for boys and girls
    df_boy = df[df['gender'] == 'boy']
    df_girl = df[df['gender'] == 'girl']
    
    # Display summary statistics for boys and girls
    st.write("Boys Data")
    st.write(df_boy.describe())
    
    st.write("Girls Data")
    st.write(df_girl.describe())
    
elif page == "Visualizations":
    # st.title("Visualizations")
    
    # # Separate data for boys
    # df_boy = df[df['gender'] == 'boy']

    # # Scatter plot for boys' height vs age
    # st.subheader("Height vs Age for Boys")
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df_boy['age'], df_boy['height'], color='red', label='Height for Boys', marker='*', s=100)
    # plt.xlabel("Age of Boys")
    # plt.ylabel("Height (cm)")
    # plt.title("Scatter Plot - Height vs Age for Boys")
    # plt.legend(loc='upper left')
    # plt.grid(True)
    # st.pyplot(plt)

    # # Example: BMI vs Age plot
    # st.subheader("BMI vs Age for Boys")
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df_boy['age'], df_boy['BMI'], color='blue', label='BMI for Boys', marker='o', s=100)
    # plt.xlabel("Age of Boys")
    # plt.ylabel("BMI")
    # plt.title("Scatter Plot - BMI vs Age for Boys")
    # plt.legend(loc='upper left')
    # plt.grid(True)
    # st.pyplot(plt)
    
    # # Plotting Z-scores for Height for Age
    # st.subheader("Z-scores for Height for Age - Boys")
    # # Assuming bh is the DataFrame with the Z-scores data
    # # Here you can use your own Z-scores dataset for 'bh'
    # plt.figure(figsize=(10, 6))
    # plt.plot(bh['age'], bh['SD4neg'], label='-4 SD', color='blue', linestyle='--')
    # plt.plot(bh['age'], bh['SD3neg'], label='-3 SD', color='cyan', linestyle='--')
    # plt.plot(bh['age'], bh['SD2neg'], label='-2 SD', color='green', linestyle='--')
    # plt.plot(bh['age'], bh['SD1neg'], label='-1 SD', color='lime', linestyle='--')
    # plt.plot(bh['age'], bh['SD0'], label='Median', color='black', linestyle='-')
    # plt.plot(bh['age'], bh['SD1'], label='+1 SD', color='orange', linestyle='--')
    # plt.plot(bh['age'], bh['SD2'], label='+2 SD', color='red', linestyle='--')
    # plt.plot(bh['age'], bh['SD3'], label='+3 SD', color='magenta', linestyle='--')
    # plt.plot(bh['age'], bh['SD4'], label='+4 SD', color='purple', linestyle='--')
    # plt.scatter(df_boy['age'], df_boy['height'], color='red', label='Height for Boys', marker='*', s=100)
    # plt.xlabel("Age of boys")
    # plt.ylabel("Z-scores")
    # plt.title("Height for Age - Boys")
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.grid(True)
    # st.pyplot(plt)
   
    # Load your datasets
    height_df_girl = pd.read_excel('height_df_girl.xlsx')
    bmi_df_girl = pd.read_excel('bmi_df_girl.xlsx')
    weight_df_girl = pd.read_excel('weight_df_girl.xlsx')
    height_df_boy = pd.read_excel('height_df_boy.xlsx')
    bmi_df_boy = pd.read_excel('bmi_df_boy.xlsx')
    weight_df_boy = pd.read_excel('weight_df_boy.xlsx')

    # Load growth standards for boys and girls
    bh_boys = pd.read_excel("HFA_Boys.xlsx")
    bh_boys['age'] = bh_boys['Month']
    bh_boys = bh_boys.drop("Month", axis=1)

    bh_girls = pd.read_excel("HFA_Girls.xlsx")
    bh_girls['age'] = bh_girls['Month']
    bh_girls = bh_girls.drop("Month", axis=1)

    bw_boys = pd.read_excel('WFA_Boys.xlsx')
    bw_boys['age'] = bw_boys['Month']
    bw_boys = bw_boys.drop("Month", axis=1)

    bw_girls = pd.read_excel('WFA_Girls.xlsx')
    bw_girls['age'] = bw_girls['Month']
    bw_girls = bw_girls.drop("Month", axis=1)

    bmi_boys = pd.read_excel('BFA_Boys.xlsx')
    bmi_boys['age'] = bmi_boys['Month']
    bmi_boys = bmi_boys.drop("Month", axis=1)

    bmi_girls = pd.read_excel('BFA_Girls.xlsx')
    bmi_girls['age'] = bmi_girls['Month']
    bmi_girls = bmi_girls.drop("Month", axis=1)

    # Set up the Streamlit app
    st.title("Child Growth Analysis")

    # Boys' height visualization
    st.header("Height Data - Boys")
    st.dataframe(height_df_boy)

    st.header("Height for Age - Boys")
    plt.figure(figsize=(10, 6))
    for sd, color in zip(['SD4neg', 'SD3neg', 'SD2neg', 'SD1neg', 'SD0', 'SD1', 'SD2', 'SD3', 'SD4'], 
                        ['blue', 'cyan', 'green', 'lime', 'black', 'orange', 'red', 'magenta', 'purple']):
        plt.plot(bh_boys['age'], bh_boys[sd], label=f'{sd} SD', color=color, linestyle='--')

    plt.scatter(height_df_boy['age'], height_df_boy['height'], color='red', label='Height for Boys', marker='*', s=100)
    plt.xlabel("Age of Boys (Months)")
    plt.ylabel("Z-scores")
    plt.title("Height for Age - Boys")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    st.pyplot(plt)

    # Girls' height visualization
    st.header("Height Data - Girls")
    st.dataframe(height_df_girl)

    st.header("Height for Age - Girls")
    plt.figure(figsize=(10, 6))
    for sd, color in zip(['SD4neg', 'SD3neg', 'SD2neg', 'SD1neg', 'SD0', 'SD1', 'SD2', 'SD3', 'SD4'], 
                        ['blue', 'cyan', 'green', 'lime', 'black', 'orange', 'red', 'magenta', 'purple']):
        plt.plot(bh_girls['age'], bh_girls[sd], label=f'{sd} SD', color=color, linestyle='--')

    plt.scatter(height_df_girl['age'], height_df_girl['height'], color='pink', label='Height for Girls', marker='*', s=100)
    plt.xlabel("Age of Girls (Months)")
    plt.ylabel("Z-scores")
    plt.title("Height for Age - Girls")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    st.pyplot(plt)

    # Boys' weight visualization
    st.header("Weight Data - Boys")
    st.dataframe(weight_df_boy)

    st.header("Weight for Age - Boys Visualization")
    plt.figure(figsize=(10, 6))
    for sd, color in zip(['SD4neg', 'SD3neg', 'SD2neg', 'SD1neg', 'SD0', 'SD1', 'SD2', 'SD3', 'SD4'], 
                        ['blue', 'cyan', 'green', 'lime', 'black', 'orange', 'red', 'magenta', 'purple']):
        plt.plot(bw_boys['age'], bw_boys[sd], label=f'{sd} SD', color=color, linestyle='--')

    plt.scatter(weight_df_boy['age'], weight_df_boy['weight'], color='brown', label='Weight for Boys', marker='o', s=100)
    plt.xlabel("Age of Boys (Months)")
    plt.ylabel("Weight (kg)")
    plt.title("Weight for Age - Boys")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    st.pyplot(plt)

    # Girls' weight visualization
    st.header("Weight Data - Girls")
    st.dataframe(weight_df_girl)

    st.header("Weight for Age - Girls Visualization")
    plt.figure(figsize=(10, 6))
    for sd, color in zip(['SD4neg', 'SD3neg', 'SD2neg', 'SD1neg', 'SD0', 'SD1', 'SD2', 'SD3', 'SD4'], 
                        ['blue', 'cyan', 'green', 'lime', 'black', 'orange', 'red', 'magenta', 'purple']):
        plt.plot(bw_girls['age'], bw_girls[sd], label=f'{sd} SD', color=color, linestyle='--')

    plt.scatter(weight_df_girl['age'], weight_df_girl['weight'], color='brown', label='Weight for Girls', marker='o', s=100)
    plt.xlabel("Age of Girls (Months)")
    plt.ylabel("Weight (kg)")
    plt.title("Weight for Age - Girls")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    st.pyplot(plt)

    # Boys' BMI visualization
    st.header("BMI Data - Boys")
    st.dataframe(bmi_df_boy)

    st.header("BMI for Age - Boys Visualization")
    plt.figure(figsize=(10, 6))
    for sd, color in zip(['SD4neg', 'SD3neg', 'SD2neg', 'SD1neg', 'SD0', 'SD1', 'SD2', 'SD3', 'SD4'], 
                        ['blue', 'cyan', 'green', 'lime', 'black', 'orange', 'red', 'magenta', 'purple']):
        plt.plot(bmi_boys['age'], bmi_boys[sd], label=f'{sd} SD', color=color, linestyle='--')

    plt.scatter(bmi_df_boy['age'], bmi_df_boy['BMI'], color='brown', label='BMI for Boys', marker='o', s=100)
    plt.xlabel("Age of Boys (Months)")
    plt.ylabel("BMI (kg/m²)")
    plt.title("BMI for Age - Boys")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    st.pyplot(plt)

    # Girls' BMI visualization
    st.header("BMI Data - Girls")
    st.dataframe(bmi_df_girl)

    st.header("BMI for Age - Girls Visualization")
    plt.figure(figsize=(10, 6))
    for sd, color in zip(['SD4neg', 'SD3neg', 'SD2neg', 'SD1neg', 'SD0', 'SD1', 'SD2', 'SD3', 'SD4'], 
                        ['blue', 'cyan', 'green', 'lime', 'black', 'orange', 'red', 'magenta', 'purple']):
        plt.plot(bmi_girls['age'], bmi_girls[sd], label=f'{sd} SD', color=color, linestyle='--')

    plt.scatter(bmi_df_girl['age'], bmi_df_girl['BMI'], color='brown', label='BMI for Girls', marker='o', s=100)
    plt.xlabel("Age of Girls (Months)")
    plt.ylabel("BMI (kg/m²)")
    plt.title("BMI for Age - Girls")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    st.pyplot(plt)
