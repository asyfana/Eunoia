import streamlit as st
import random
from streamlit_lottie import st_lottie
import json
import pandas as pd
import numpy as np 
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import plotly.express as px
from textblob import TextBlob
from streamlit_lottie import st_lottie
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import streamlit as st
from textblob import TextBlob


def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        return None

def get_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Load credentials from Streamlit secrets
    credentials_dict = st.secrets["gcp_service_account"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)

    client = gspread.authorize(credentials)
    spreadsheet = client.open("Eunoia_Data_streamlit")

    sheets = {
        "User": spreadsheet.worksheet("User"),
        "Task_user": spreadsheet.worksheet("Task_user"),
        "ML_task": spreadsheet.get_worksheet(2),
        "User_Mood": spreadsheet.get_worksheet(3),
        "Recommended_task_priority": spreadsheet.get_worksheet(4),
    }

    return sheets


# ✅ Then call it like this:
sheets = get_google_sheets()

@st.cache_data(ttl=60)
def get_user_data():
    return pd.DataFrame(sheets["User"].get_all_records())

@st.cache_data(ttl=60)
def get_task_user_data():
    return pd.DataFrame(sheets["Task_user"].get_all_records())

@st.cache_data(ttl=60)
def get_mood_data():
    return pd.DataFrame(sheets["User_Mood"].get_all_records())

@st.cache_data(ttl=60)
def get_predicted_task_data():
    return pd.DataFrame(sheets["Recommended_task_priority"].get_all_records())

@st.cache_data(ttl=60)
def get_ml_task_data():
    return pd.DataFrame(sheets["ML_task"].get_all_records())


# Make sure 'Recommended_task_priority' sheet has headers
priority_sheet = sheets["Recommended_task_priority"]
if not priority_sheet.get_all_values():
    priority_sheet.append_row([
        "Username", "Task_Name", "Task_Type", "Priority", "Completion_Status", "Dependency",
        "Task_Importance", "Estimated_Time", "Interruptions", "Workload",
        "Hours_To_Deadline", "Predicted_Urgency_Score", "Suggested_Order"
    ])





import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# Load dataset
df = pd.read_csv("Realistic_Task_Dataset_With_Urgency.csv")

# Drop unneeded columns
df = df.drop(columns=["Task_ID", "Task_Creation_Time", "Deadline"])

# Encode categorical features
cat_cols = ["Task_Type", "Priority", "Completion_Status", "Dependency"]
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[cat_cols] = encoder.fit_transform(df[cat_cols])

# Split features and target
X = df.drop(columns=["Urgency_Score"])
y = df["Urgency_Score"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model and encoder
joblib.dump(model, "urgency_model.pkl")
joblib.dump(encoder, "urgency_encoder.pkl")





# Load model and encoder
model = joblib.load("urgency_model.pkl")
encoder = joblib.load("urgency_encoder.pkl")

# Define categorical options
task_types = [
    "Meeting", "Planning", "Analysis", "Review", "Development", "Deployment",
    "Testing", "Research", "Customer Support", "Documentation", "Report Writing",
    "Exam Preparation", "Group Project", "Case Study", "Design", "Lecture Notes Review",
    "Brainstorming", "Bug Fixing", "Training", "Presentation", "Assignment", "Others"
]
priorities = ["Low", "Medium", "High"]
statuses = ["Pending", "In Progress", "Completed"]
dependencies = ["None", "External", "Internal"]

st.write("-----------------------------------------------")



# Initialize session state
if "mood" not in st.session_state:
    st.session_state.mood = None
if "schedule" not in st.session_state:
    st.session_state.schedule = []
if "character" not in st.session_state:
    st.session_state.character = "Eun"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:  
    st.session_state.username = ""



# Sidebar Login System
st.sidebar.title("Hello!")
st.sidebar.write("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Enter"):
    if username and password:
        st.session_state.logged_in = True
        st.session_state.username = username  # ✅ Ensure it's set here
        st.sidebar.success("Logged in successfully!")


# Display title and quote
st.markdown("""
    <h1 style='text-align: center;'>EunoiaVR</h1>
    <h3 style='text-align: center; color: gray;'>Your Virtual Companion! </h3>
    <hr>
""", unsafe_allow_html=True)

if st.session_state.logged_in:

    # Character Selection in Sidebar
    st.sidebar.write("---")
    st.sidebar.markdown("### Character")
    if st.sidebar.button("Eun"):
        st.session_state.character = "Eun"
    if st.sidebar.button("Noia"):
        st.session_state.character = "Noia"

    st.sidebar.title("choose pages")
    page = st.sidebar.selectbox('choose a page', ['Mood', 'mood Progress', 'Task Priority', 'Chat with companion'])
    
    if page == 'Mood':
        # Load mood messages
        Mood_messages = pd.read_csv("(latest)mood_messages.csv")

        # Header Section
        st.markdown("""
            <div style='text-align: center; padding: 10px 0;'>
                <h2>💖 How are you feeling today?</h2>
                <p style='font-size: 16px; color: #555;'>Let your companion know how your heart feels right now (DD/MM/YY).</p>
            </div>
        """, unsafe_allow_html=True)

        # Mood selection
        mood_options = Mood_messages["Mood"].unique().tolist()
        button_container = st.columns(3)

        st.markdown("<h4 style='text-align: center;'>Click the mood that matches your feeling:</h4>", unsafe_allow_html=True)
        
        for i, mood in enumerate(mood_options):
            if button_container[i % 3].button(f"🌀 {mood}"):
                st.session_state.mood = mood
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheets["User_Mood"].append_row([username, now, mood])
                st.success("✅ Mood saved! Your companion is here for you.")

        # If mood selected
        if "mood" in st.session_state and st.session_state.mood:
            current_mood = st.session_state.mood
            filtered = Mood_messages[Mood_messages["Mood"] == current_mood]

            if not filtered.empty:
                row = filtered.sample(1).iloc[0]
                message = row["Encouraging Message"]
                suggestion = row["Suggested Activity"]
            else:
                message = "You're doing your best. Keep going!"
                suggestion = "Take a moment to relax and reflect."

            st.markdown(f"<h3 style='text-align: center; color: #6C63FF;'>{st.session_state.character} is here for you 💬</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center;'>Current Mood: <span style='color:#FF8C42'>{current_mood}</span></h4>", unsafe_allow_html=True)

            # Display character video and mood-based content
            col1, col2 = st.columns([3, 2.5])
            with col1:
                # Define mood-to-image mapping
                mood_image_map = {
                    "Miserable": {"Eun": "E1.jpg", "Noia": "N1.jpg"},
                    "Sad": {"Eun": "E2.jpg", "Noia": "N2.jpg"},
                    "Happy": {"Eun": "E3.jpg", "Noia": "N3.jpg"},
                    "Default": {"Eun": "E4.jpg", "Noia": "N4.jpg"}
                }

                # Get current character
                character = st.session_state.character  # should be "Eun" or "Noia"

                # Select the appropriate image
                if current_mood in mood_image_map:
                    image_file = mood_image_map[current_mood].get(character, "E4.jpg")
                else:
                    image_file = mood_image_map["Default"].get(character, "E4.jpg")

                # Full path to the image inside the character subfolder
                image_path = f"Characters/{character}/{image_file}"

                # Display the image
                with col1:
                    st.image(image_path, use_column_width=True)


            with col2:
                st.markdown(
                    f"""
                    <div style='border: 2px solid #cfcfcf; background-color: #f9f9f9; padding: 20px; border-radius: 15px;'>
                        <h4 style='text-align: center; color: #4B4453;'>🌱 Here's something you might enjoy:</h4>
                        <ul style='list-style-position: inside; color: #333; font-size: 16px;'>
                            <li>{suggestion}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div style='margin-top: 20px; border: 2px solid #ffd6d6; background-color: #fff0f0; padding: 15px 20px; border-radius: 15px;'>
                    <p style='text-align: center; font-size: 18px;'>
                        <strong>{st.session_state.character} says:</strong><br>
                        <em>“{message}”</em>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )

            

    elif page == 'mood Progress':
        st.subheader("📊 Mood Progress Over Time")

        mood_df = get_mood_data()
        mood_df["DateTime"] = pd.to_datetime(mood_df["DateTime"])
        user_mood_df = mood_df[mood_df["Username"].str.lower() == st.session_state.username.lower()]

        if not user_mood_df.empty:
            import plotly.graph_objects as go
            from io import BytesIO
            import base64

            # 📅 Date filter
            min_date = user_mood_df["DateTime"].min().date()
            max_date = user_mood_df["DateTime"].max().date()
            start_date, end_date = st.date_input("Filter by date range:",
                                                value=[min_date, max_date],
                                                min_value=min_date,
                                                max_value=max_date)

            user_mood_df = user_mood_df[
                (user_mood_df["DateTime"].dt.date >= start_date) &
                (user_mood_df["DateTime"].dt.date <= end_date)
            ]

            # 🧠 Mood valence with emoji labels
            mood_valence = {
                "Miserable": -3,
                "Sad": -2,
                "Tired": -1,
                "Bored": -1,
                "Neutral": 0,
                "Calm": 1,
                "Relaxed": 1,
                "Happy": 2,
                "Content": 2,
                "Excited": 3,
                "Elated": 3
            }

            mood_emoji_labels = {
                -3: "😢 Miserable",
                -2: "😞 Sad",
                -1: "🥱 Tired/Bored",
                0: "😐 Neutral",
                1: "😌 Calm/Relaxed",
                2: "😊 Happy/Content",
                3: "🤩 Excited/Elated"
            }

            user_mood_df["Valence"] = user_mood_df["Mood"].map(mood_valence).fillna(0)
            user_mood_df = user_mood_df.sort_values("DateTime")

            def valence_to_color(val):
                if val <= -2:
                    return "red"
                elif val == -1:
                    return "orange"
                elif val == 0:
                    return "gray"
                elif val == 1:
                    return "lightgreen"
                else:
                    return "green"

            colors = user_mood_df["Valence"].apply(valence_to_color)
            user_mood_df["RollingAverage"] = user_mood_df["Valence"].rolling(window=3, min_periods=1).mean()

            mood_trace = go.Scatter(
                x=user_mood_df["DateTime"],
                y=user_mood_df["Valence"],
                mode="lines+markers",
                line=dict(color="royalblue", width=2),
                marker=dict(color=colors, size=10),
                name="Mood Valence",
                customdata=user_mood_df["Mood"],
                hovertemplate="<b>%{x}</b><br>Mood: %{customdata}<br>Valence: %{y}"
            )

            trend_trace = go.Scatter(
                x=user_mood_df["DateTime"],
                y=user_mood_df["RollingAverage"],
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="3-Entry Moving Average"
            )

            layout = go.Layout(
                title="Mood Timeline with Emotional Valence",
                xaxis_title="Date/Time",
                yaxis_title="Emotional Valence",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(mood_emoji_labels.keys()),
                    ticktext=list(mood_emoji_labels.values())
                ),
                template="plotly_white"
            )

            fig = go.Figure(data=[mood_trace, trend_trace], layout=layout)
            st.plotly_chart(fig, use_container_width=True)

            

            # ---------------- Mood Summary ----------------
            st.markdown("### 🧠 Mood Summary")

            # Count most common mood
            most_common = user_mood_df["Mood"].mode().values[0]
            count = user_mood_df["Mood"].value_counts()[most_common]

            # Calculate average valence
            avg_valence = user_mood_df["Valence"].mean().round(2)

            # Calculate longest same-mood streak
            streak = 1
            max_streak = 1
            prev = None

            for mood in user_mood_df["Mood"]:
                if mood == prev:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 1
                    prev = mood

            # Summary layout
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Most Frequent Mood", most_common, f"{count} times")

            with col2:
                st.metric("Average Valence", avg_valence)

            with col3:
                st.metric("Longest Mood Streak", f"{max_streak} days")


            # 📊 Show data table
            with st.expander("See raw data"):
                st.dataframe(user_mood_df.sort_values("DateTime", ascending=False))

        else:
            st.warning("No mood data available.")

    
    
    elif page == 'Task Priority':
        st.markdown("""
            <h3 style='text-align: center;'>Analyze your task</h3>
        """, unsafe_allow_html=True)


        st.title("🔍 EunoiaVR Task Priority Analyzer")

        st.markdown("Input your tasks below. Once done, click **Analyze Task Priority** to get sorted recommendations.")

        # --- Task log section ---
        task_log = st.session_state.get("task_log", [])

        # Input form
        with st.form("task_form"):
            col1, col2 = st.columns(2)
            with col1:
                task_name = st.text_input("Task Name", placeholder="e.g. Write report")
                task_type = st.selectbox("Task Type", task_types)
                priority = st.selectbox("Priority", priorities)
                completion_status = st.selectbox("Completion Status", statuses)
                dependency = st.selectbox("Dependency", dependencies)
            with col2:
                task_importance = st.slider("Task Importance", 1, 10, 5)
                estimated_time = st.number_input("Estimated Time (hours)", 0.5, 10.0, step=0.5)
                interruptions = st.number_input("Interruptions (today)", 0.0, 10.0, step=0.1)
                workload = st.slider("Workload", 1.0, 10.0, 5.0)

            deadline = st.date_input("Deadline")
            creation_time = st.date_input("Task Creation Time")

            submitted = st.form_submit_button("Add Task")

            if submitted:
                hours_to_deadline = (pd.to_datetime(deadline) - pd.to_datetime(creation_time)).total_seconds() / 3600
                task_row = {
                    "Username": username,
                    "Task_Name": task_name,
                    "Task_Type": task_type,
                    "Priority": priority,
                    "Completion_Status": completion_status,
                    "Dependency": dependency,
                    "Task_Importance": task_importance,
                    "Estimated_Time": estimated_time,
                    "Interruptions": interruptions,
                    "Workload": workload,
                    "Hours_To_Deadline": hours_to_deadline,
                    "Deadline": deadline.strftime("%Y-%m-%d"),
                    "Task_Creation_Time": creation_time.strftime("%Y-%m-%d")
                }
                sheets["Task_user"].append_row(list(task_row.values()))
                st.success("Task saved to Google Sheet!")

           

        # --- Analyze Priority ---
        if st.button("📊 Analyze Task Priority"):
            all_tasks_df = get_task_user_data()
            user_tasks_df = all_tasks_df[all_tasks_df["Username"].str.lower() == st.session_state.username.lower()]


            if not user_tasks_df.empty:
                task_df = user_tasks_df.copy()

                # Encode categorical columns
                cat_cols = ["Task_Type", "Priority", "Completion_Status", "Dependency"]
                
                task_df[cat_cols] = encoder.transform(task_df[cat_cols])

                predictions = model.predict(task_df[[
                    "Task_Type", "Priority", "Completion_Status", "Dependency",
                    "Task_Importance", "Estimated_Time", "Interruptions",
                    "Workload", "Hours_To_Deadline"
                ]])

                task_df["Predicted_Urgency_Score"] = np.round(predictions, 2)
                task_df["Suggested_Order"] = task_df["Predicted_Urgency_Score"].rank(method="first", ascending=False).astype(int)
                task_df_sorted = task_df.sort_values(by="Predicted_Urgency_Score", ascending=False)

                # Show predictions
                st.subheader("🔽 Recommended Task Priority")
                st.dataframe(task_df_sorted[[
                    "Task_Name", "Task_Type", "Priority", "Task_Importance", "Estimated_Time",
                    "Hours_To_Deadline", "Predicted_Urgency_Score", "Suggested_Order"
                ]])

                # ✅ Push predictions to Google Sheet
                for _, row in task_df_sorted.iterrows():
                    sheets["Recommended_task_priority"].append_row([
                        st.session_state.username,
                        row["Task_Name"],
                        row["Task_Type"],
                        row["Priority"],
                        row["Completion_Status"],
                        row["Dependency"],
                        row["Task_Importance"],
                        row["Estimated_Time"],
                        row["Interruptions"],
                        row["Workload"],
                        round(row["Hours_To_Deadline"], 2),
                        round(row["Predicted_Urgency_Score"], 2),
                        int(row["Suggested_Order"])
                    ])
            else:
                st.warning("⚠️ You have no tasks in the sheet to analyze. Please add some first.")


        # Show all original user tasks
        st.subheader("📋 Your Logged Tasks from Google Sheet")
        all_tasks_df = get_task_user_data()
        all_tasks_df.columns = [col.strip() for col in all_tasks_df.columns]  # Clean up column names
        user_tasks_df = all_tasks_df[all_tasks_df["Username"].str.lower() == st.session_state.username.lower()]
        st.dataframe(user_tasks_df)

        # Show all predicted priorities
        
        priority_df = get_predicted_task_data()
        priority_df.columns = [col.strip() for col in priority_df.columns]

        if not priority_df.empty and "Username" in priority_df.columns:
            user_priority_df = priority_df[priority_df["Username"].str.lower() == st.session_state.username.lower()]
            st.subheader("📌 Your Predicted Priorities from Google Sheet")
            st.dataframe(user_priority_df)
        else:
            st.info("⏳ No predicted tasks yet or 'Username' column is missing.")



        # --- Finish Task ---
        if task_log:
            st.subheader("✅ Mark Task as Completed")
            task_names = [f"{i+1}. {t['Task_Type']} ({t['Priority']})" for i, t in enumerate(task_log)]
            finished_task = st.selectbox("Select completed task", task_names)
            if st.button("Mark as Done"):
                idx = task_names.index(finished_task)
                task_log.pop(idx)
                st.session_state["task_log"] = task_log
                st.success("🎉 Task marked as completed!")

    
  
    elif page == 'Chat with companion':
        st.title("🧸  You can chat with your companion here!")

        st.write("Chat with me about anything — I'm here to listen 💬") 

        @st.cache_data
        def load_chat_data():
            try:
                df = pd.read_csv('Chat_NLP.csv')  # Updated to your new dataset
                df.dropna(subset=['Context', 'Response'], inplace=True)
                return df
            except FileNotFoundError:
                st.error("The 'Chat_NLP.csv' file was not found.")
                return None

        def detect_sentiment(text):
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.3:
                return "positive"
            elif polarity < -0.3:
                return "negative"
            else:
                return "neutral"

        def personalize_response(response, sentiment):
            if sentiment == "positive":
                return "😊 I'm glad to hear that! " + response
            elif sentiment == "negative":
                return "💙 It sounds like you're going through something. " + response
            else:
                return "🧠 Got it. " + response

        def get_response(user_input, data):
            if data is None or data.empty:
                return "I'm sorry, I don't have any information to share at the moment."

            vectorizer = TfidfVectorizer(stop_words='english')
            question_vectors = vectorizer.fit_transform(data['Context'])
            user_input_vector = vectorizer.transform([user_input])

            similarities = cosine_similarity(user_input_vector, question_vectors)
            most_similar_index = similarities.argmax()
            similarity_score = similarities[0, most_similar_index]

            if similarity_score > 0.1:
                return data['Response'].iloc[most_similar_index]
            else:
                return "I'm not sure how to answer that. Can you please try rephrasing?"

        def main():
            st.title("Your Virtual Companion")
            st.write("Hello! I'm here to listen. Feel free to share what's on your mind.")

            faq_data = load_data()

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("What would you like to talk about?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    sentiment = detect_sentiment(prompt)
                    raw_response = get_response(prompt, faq_data)
                    final_response = personalize_response(raw_response, sentiment)
                    st.markdown(final_response)

                st.session_state.messages.append({"role": "assistant", "content": final_response})

        if __name__ == "__main__":
            main()








