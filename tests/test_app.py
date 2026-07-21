import streamlit as st

st.title("🧪 Streamlit Test App")
st.write("If you can see this, Streamlit is working!")

if st.button("Click me"):
    st.success("✅ Button clicked successfully!")
    
st.sidebar.write("Sidebar is working too!")

