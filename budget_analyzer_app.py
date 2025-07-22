import streamlit as st
import pyrebase
from firebase_config import firebase_config
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import plotly.express as px
from datetime import datetime
import uuid

# --------------------------
# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

# --------------------------
# Session State Setup
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "email" not in st.session_state:
    st.session_state.email = None
if "username" not in st.session_state:
    st.session_state.username = None
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "analysis_unsaved" not in st.session_state:
    st.session_state.analysis_unsaved = False
if "selected_analysis" not in st.session_state:
    st.session_state.selected_analysis = None


# Text Cleaning Function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        return text.translate(str.maketrans('', '', string.punctuation))
    return ""

# --------------------
# ML Model for Needs/Wants
# --------------------
@st.cache_resource(show_spinner=True)
def load_and_train_model():
    df_train = pd.read_csv('expanded_expenses_with_dates.csv')
    df_train['CleanDescription'] = df_train['ItemDescription'].apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_train['CleanDescription'])
    y = df_train['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, vectorizer

model, vectorizer = load_and_train_model()

# --------------------
# Save Summary to Firebase
# --------------------
def convert_keys_to_str(d):
    return {str(k): v for k, v in d.items()}

def save_analysis_to_db(uid, summary_data, filename, full_df, monthly_spend, weekly_spend):
    analysis_id = str(uuid.uuid4())

    need_total = float(full_df[full_df['PredictedLabel'] == 'Need']['TotalCost'].sum())
    want_total = float(full_df[full_df['PredictedLabel'] == 'Want']['TotalCost'].sum())

    summary_data.update({
        "filename": filename,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "need_total": need_total,
        "want_total": want_total,
        "monthly_spend": convert_keys_to_str(monthly_spend.to_dict()),
        "weekly_spend": convert_keys_to_str(weekly_spend.to_dict())
    })

    db.child("users").child(uid).child("saved_analyses").child(analysis_id).set(summary_data)
    st.session_state.analysis_unsaved = False


# --------------------
# Sidebar with User Info and History
# --------------------
def user_sidebar(user_info):
    st.sidebar.markdown(f"👤 Username: **{user_info['username']}**")
    st.sidebar.markdown(f"📧 Email: **{user_info['email']}**")
    st.sidebar.markdown("### 📂 Your Saved Analyses:")

    user_id = user_info.get('uid')
    ref = db.child('users').child(user_id).child('saved_analyses')
    data = ref.get().val()

    if "selected_analysis" not in st.session_state:
        st.session_state.selected_analysis = None

    if data:
        for key, val in data.items():
            filename = val.get('filename', 'Unnamed')
            date = val.get('date', 'No Date')
            if st.sidebar.button(f"{filename} - {date}"):
                st.session_state.selected_analysis = val
                #st.experimental_rerun()
    else:
        st.sidebar.info("No saved analyses found.")

    # ✅ Show "Upload New File" if a saved analysis is selected
    if st.session_state.selected_analysis:
        if st.sidebar.button("📤 Upload New File"):
            st.session_state.selected_analysis = None
            st.experimental_rerun()

    if st.sidebar.button("🚪 Logout"):
        if not st.session_state.get("analysis_saved", True):
            if not st.sidebar.checkbox("⚠ I don't want to save this analysis"):
                st.warning("Please save your analysis before logging out or confirm you don't want to save it.")
                st.stop()
        st.session_state.clear()
        st.experimental_rerun()

    return st.session_state.selected_analysis



def display_saved_analysis(analysis):
    st.subheader(f"📁 Viewing Saved Analysis: {analysis.get('filename', 'Unnamed')}")
    st.markdown(f"*Date:* {analysis.get('date', 'Unknown')}")
    st.markdown(f"*Total Spending:* AED {analysis.get('total_expense', 'N/A')}")
    st.markdown(f"*Needs %:* {analysis.get('needs_percent', 'N/A')}%")
    st.markdown(f"*Wants %:* {analysis.get('wants_percent', 'N/A')}%")
    st.markdown(f"*Top Category:* {analysis.get('top_category', 'N/A')}")

       # Needs vs Wants Pie Chart
    if 'need_total' in analysis and 'want_total' in analysis:
        st.markdown("Needs vs Wants Breakdown")
        chart_data = pd.DataFrame({
            'Category': ['Needs', 'Wants'],
            'Amount': [analysis['need_total'], analysis['want_total']]
        })
        fig = px.pie(chart_data, names='Category', values='Amount', title="Needs vs Wants")
        st.plotly_chart(fig)

    # Weekly Spending Chart
    if 'weekly_spend' in analysis:
        st.markdown("### 📈 Weekly Spending Trends")
        weekly_df = pd.DataFrame.from_dict(analysis['weekly_spend'], orient='index', columns=['TotalCost'])
        weekly_df.index.name = 'Week'
        weekly_df = weekly_df.sort_index()
        st.line_chart(weekly_df)

    # Monthly Spending Chart
    if 'monthly_spend' in analysis:
        st.markdown("### 📈 Monthly Spending Trends")
        monthly_df = pd.DataFrame.from_dict(analysis['monthly_spend'], orient='index', columns=['TotalCost'])
        monthly_df.index.name = 'Month'
        monthly_df = monthly_df.sort_index()
        st.line_chart(monthly_df)



# --------------------
# Dashboard for Analysis
# --------------------
def dashboard(user_info):
    selected_analysis = user_sidebar(user_info)

    if selected_analysis:
        display_saved_analysis(selected_analysis)
        return

    # Regular upload interface
    st.title("💸 Budget Analyzer Dashboard")
    uploaded_file = st.file_uploader("📁 Upload your expense CSV", type=["csv"])
    

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = {'ItemDescription', 'NumberOfItemsPurchased', 'CostPerItem', 'Date'}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must include: {', '.join(required_cols)}")
            return

        df['CleanDescription'] = df['ItemDescription'].apply(clean_text)
        df['TotalCost'] = df['NumberOfItemsPurchased'] * df['CostPerItem']
        df['Date'] = pd.to_datetime(df['Date'])

        X_upload = vectorizer.transform(df['CleanDescription'])
        df['PredictedLabel'] = model.predict(X_upload)

        st.session_state.analysis_unsaved = True

        if 'Label' not in df.columns:
            st.info("✅ Your file has no labels — we used our trained model to predict Needs/Wants.")
        else:
            y_true = df['Label']
            y_pred = df['PredictedLabel']
            st.markdown("Model Performance on Your File")
            st.text(classification_report(y_true, y_pred))

        st.subheader("Analyzed Transactions")
        st.dataframe(df[['Date', 'ItemDescription', 'TotalCost', 'PredictedLabel']])

        need_total = df[df['PredictedLabel'] == 'Need']['TotalCost'].sum()
        want_total = df[df['PredictedLabel'] == 'Want']['TotalCost'].sum()
        total = need_total + want_total

        st.markdown(f"Total Spending: AED {total:.2f}")
        st.markdown(f"- Needs: AED {need_total:.2f}")
        st.markdown(f"- Wants: AED {want_total:.2f}")

        recommended_needs = 0.7 * total
        recommended_wants = 0.3 * total
        st.markdown("70-30 Budget Recommendation")
        st.markdown(f"- Recommended for Needs: AED {recommended_needs:.2f}")
        st.markdown(f"- Recommended for Wants: AED {recommended_wants:.2f}")

        if want_total > recommended_wants:
            st.warning("You are overspending on wants! Consider reducing luxury spending.")
        else:
            st.success("Good job! Your spending on wants is within a healthy range.")

        st.markdown("Set a Monthly Savings Goal")
        savings_goal = st.number_input("Enter your desired savings amount (AED)", min_value=0.0, step=100.0)

        if savings_goal:
            if savings_goal >= total:
                st.info("Amazing! You're saving more than you spent.")
            elif total - savings_goal < 0:
                st.warning("Warning: Spending exceeds your savings goal.")
            else:
                st.success(f"You're spending AED {total:.2f} and aiming to save AED {savings_goal:.2f}. Keep tracking!")

        st.markdown("Expense Insights")
        top_expenses = df.sort_values(by='TotalCost', ascending=False).head(3)
        st.write("Top 3 Most Expensive Items:")
        st.table(top_expenses[['ItemDescription', 'TotalCost']])

        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Day'] = df['Date'].dt.date

        daily_spend = df.groupby('Day')['TotalCost'].sum()
        weekly_spend = df.groupby('Week')['TotalCost'].sum()
        monthly_spend = df.groupby('Month')['TotalCost'].sum()

        st.write(f"Average Daily Spend: AED {daily_spend.mean():.2f}")
        st.write(f"Average Weekly Spend: AED {weekly_spend.mean():.2f}")
        st.write(f"Average Monthly Spend: AED {monthly_spend.mean():.2f}")


        daily_spend = df.groupby('Day')['TotalCost'].sum()
        weekly_spend = df.groupby('Week')['TotalCost'].sum()
        monthly_spend = df.groupby('Month')['TotalCost'].sum()

        st.markdown("Spending Visualizations")
        chart_data = pd.DataFrame({
            'Category': ['Needs', 'Wants'],
            'Amount': [need_total, want_total]
        })

        chart_type = st.radio("Choose chart type:", ["Pie Chart", "Bar Chart"])
        if chart_type == "Pie Chart":
            fig = px.pie(chart_data, names='Category', values='Amount', title="Needs vs Wants")
            st.plotly_chart(fig)
        elif chart_type == "Bar Chart":
            bar_data = chart_data.set_index('Category')
            st.bar_chart(bar_data)

        st.subheader("📈 Monthly Spending Trends")
        st.line_chart(monthly_spend)

        st.subheader("📈 Weekly Spending Trends")
        st.line_chart(weekly_spend)

        st.markdown("Predict Label for a Single Item")
        custom_text = st.text_input("Enter item description:")
        if custom_text:
            cleaned_text = clean_text(custom_text)
            vec = vectorizer.transform([cleaned_text])
            prediction = model.predict(vec)[0]
            st.info(f"Prediction: {prediction}")


        filename_input = st.text_input("💾 Enter a name for this analysis to save:")
        if st.button("Save Analysis"):
            if not filename_input:
                st.warning("Please provide a name to save this analysis.")
            else:
                summary_data = {
                    "total_expense": float(total),
                    "needs_percent": round((need_total / total) * 100, 2),
                    "wants_percent": round((want_total / total) * 100, 2),
                    "top_category": df['PredictedLabel'].value_counts().idxmax()
                }
                save_analysis_to_db(user_info['uid'], summary_data, filename_input, df, monthly_spend, weekly_spend)

                st.success("Analysis saved successfully!")

# --------------------------
# Signup Page
# --------------------------
def signup_page():
    st.title("Create an Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    username = st.text_input("Username (unique)")

    if st.button("Sign Up"):
        if not email or not password or not username:
            st.warning("Please fill in all fields.")
            return
        try:
            user = auth.create_user_with_email_and_password(email, password)
            uid = user["localId"]
            db.child("users").child(uid).set({
                "email": email,
                "username": username
            })
            st.success("Account created successfully! Please log in.")
        except Exception as e:
            st.error(f"Error: {e}")

# --------------------------
# Login Page
# --------------------------
def login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not email or not password:
            st.warning("Please enter both email and password.")
            return
        try:
            
            auth_data = auth.sign_in_with_email_and_password(email, password)
            uid = auth_data["localId"]
            user_data = db.child("users").child(uid).get().val()
            if user_data:
                st.session_state.logged_in = True
                st.session_state.email = email
                st.session_state.username = user_data.get("username", "User")
                st.session_state.user_info = {
                    "uid": uid,
                    "email": email,
                    "username": user_data.get("username", "User")
                }
                st.success("Login successful!")
                #st.experimental_rerun()
            else:
                st.error("User data not found!")
        except Exception as e:
            st.error(f"Login failed: {e}")

# --------------------------
# Main Function
# --------------------------
def main():
    if not st.session_state.logged_in:
        choice = st.radio("Choose:", ["Login", "Sign Up"])
        if choice == "Sign Up":
            signup_page()
        else:
            login_page()
    else:
        dashboard(st.session_state.user_info)

#if __name__ == '_main_':
main()




