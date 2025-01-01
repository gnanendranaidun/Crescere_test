import pandas as pd
import json
import requests
import matplotlib.pyplot as plt
import streamlit as st

# Firebase URL
FIREBASE_URL = "https://childmonitoring-951c3-default-rtdb.firebaseio.com/Users.json"

data = {
    'goutham': {
        'age': '78',
        'id':'2',
        'gender':'boy',
        'height': '110',
        'mail': 'goutham@gmail.com',
        'name': 'goutham',
        'password': 'G@210457',
        'weight': '20'
    },
    'john123': {
        'age': '65',
        'id':'3',
        'gender':'boy',
        'height': '130',
        'mail': 'john123@gmail.com',
        'name': 'John',
        'password': 'pass456',
        'weight': '19'
    },
    'bharani': {
        'age': '83',
        'id':'4',
        'gender':'girl',
        'height': '141',
        'mail': 'bharani@gmail.com',
        'name': 'Bharani',
        'password': 'Bha@210457',
        'weight': '28'
    }
}

# Sending the data to Firebase
response = requests.put(FIREBASE_URL, json=data)

# Retrieve data from Firebase
response = requests.get(FIREBASE_URL)
if response.status_code == 200:
    data = response.json()
    st.write("Retrieved data:", data)
else:
    st.write(f"Error: {response.status_code} {response.text}")

df = pd.DataFrame(data).T
df["age"] = pd.to_numeric(df["age"])
df["height"] = pd.to_numeric(df["height"])
df["weight"] = pd.to_numeric(df["weight"])
df["BMI"] = round((df["weight"] * 10000) / (df["height"] ** 2), 2)
st.write("Updated DataFrame with BMI:", df)

# Separate data for boys and girls
df_boy = df[df['gender'] == 'boy']
df_girl = df[df['gender'] == 'girl']
st.write('Boys data:', df_boy)
st.write('Girls data:', df_girl)

# Plotting
bh = pd.read_excel("HFA_Boys.xlsx")
bh['age'] = bh['Month']
bh = bh.drop("Month", axis=1)
st.write("Height for Boys DataFrame:", bh)

plt.figure(figsize=(10, 6))
plt.plot(bh['age'], bh['SD4neg'], label='-4 SD', color='blue', linestyle='--')
plt.plot(bh['age'], bh['SD3neg'], label='-3 SD', color='cyan', linestyle='--')
plt.plot(bh['age'], bh['SD2neg'], label='-2 SD', color='green', linestyle='--')
plt.plot(bh['age'], bh['SD1neg'], label='-1 SD', color='lime', linestyle='--')
plt.plot(bh['age'], bh['SD0'], label='Median', color='black', linestyle='-')
plt.plot(bh['age'], bh['SD1'], label='+1 SD', color='orange', linestyle='--')
plt.plot(bh['age'], bh['SD2'], label='+2 SD', color='red', linestyle='--')
plt.plot(bh['age'], bh['SD3'], label='+3 SD', color='magenta', linestyle='--')
plt.plot(bh['age'], bh['SD4'], label='+4 SD', color='purple', linestyle='--')
plt.scatter(df_boy['age'], df_boy['height'], color='red', label='Height for Boys', marker='*', s=100)
plt.xlabel("Age of boys")
plt.ylabel("Z-scores")
plt.title("Height for Age - Boys")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
st.pyplot(plt)



# You can add additional charts and interactive visualizations as needed
