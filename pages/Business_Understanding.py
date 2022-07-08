import streamlit as st

st.markdown("<h1 style='text-align: center'>Our Business Understanding Process</h1>", unsafe_allow_html=True)
st.markdown("""<p style='text-align: center'>While many teams hurry through this phase, establishing a strong business understanding
         is like building the foundation of a house – absolutely essential.</p>""", unsafe_allow_html=True)


with st.expander("Motivation"):
    st.write("""What was our motivation? We are excited about machine learning in the context of visual challenges, for example, autonomous driving, which can 
             provide more safety on the road. For this reason, we were eager to develop an application in the field of labeling. Through the project, we hoped to be 
             able to practically apply our theoretical foundations and expand our knowledge in the field of image recognition and labeling.""")

with st.expander("Determining Business Objectives"):
    st.write("""At the beginning, we had many ideas that usually had to be changed during the project due to factors such as time, feasibility and difficulty. 
             The goal, of course, was to train a model that could generate subtitles as accurately as possible. Finding a model, as well as training and transferring 
             it to the Streamlit application, proved in some cases to be more difficult than expected in some cases. Nonetheless, we can now present a relatively good 
             model that can present some features to engage the user in the application.""")

with st.expander("Assess Situation"):
    st.write("""Of course, we thought about what resources we would need and what problems we might encounter beforehand.
            For research, Google Schoolar and Papers with code were of course of great importance. As for software, we settled on.
            Kaggle, TensorFlow, Pycharm, and Jupyter, among others. Concrete libraries we used for implementation were Streamlit and Torch, of course.
            We identified the quality of the images and labels that came with the original LAION dataset as a potential problem.""")

with st.expander("Determine Data Mining Goals"):
    st.write("""We did not set ourselves a specific technical goal. First of all, it was important for us to develop an understanding
             and thus to have a model that can roughly and comprehensibly recognize the image content. Based on this, we definitely wanted to include 
             the user as well as some features.""")

with st.expander("Produce Project Plan"):
    st.write("""We designed our project plan to be relatively flexible. We had set ourselves specific deadlines early on, divided the tasks among ourselves accordingly,
             and consulted each other at least weekly on how to proceed with the current development. Roughly speaking, it was a matter of first doing research,
             preparing the data, training a model, and finally evaluating it. Of course, there were always problems along the way, which we solved together.""")
