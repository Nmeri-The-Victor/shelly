import streamlit as st
from time import sleep
from navigation import make_sidebar


st.title("Shelly, Your Personal Study Assistant.")

st.write("Please log in to continue (username `test`, password `test`).")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

option = st.selectbox(
    "How would you like to be contacted?",
    ("Signup", "Login"))


if option == "Login":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        if username == "test" and password == "test":
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.session_state["auth"] = True
            sleep(0.5)
            st.switch_page("app.py")
        else:
            st.session_state["auth"] = False
            st.error("Incorrect username or password")
else:
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Signup", type="primary"):
        if username == "test" and password == "test":
            st.success("Signup successfully!")
            st.session_state["auth"] = True
            sleep(0.5)
            st.switch_page("app.py")
        