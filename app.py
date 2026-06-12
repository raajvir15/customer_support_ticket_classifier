import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Smart Ticket Classifier",
    page_icon = "🎫",
    layout="wide"
)

## loading all the models
@st.cache_resource   # so that the sit does not keep on refreshing 

def load_models():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    type_classifier = joblib.load("models/type_classifier.pkl")
    priority_classifier = joblib.load("models/priority_classifier.pkl")
    le_type = joblib.load("models/le_type.pkl")
    le_priority = joblib.load('models/le_priority.pkl')

    return encoder,type_classifier, priority_classifier, le_type, le_priority

encoder, type_classifier, priority_classifier, le_type, le_priority = load_models()


## headet
st.title("🎫 Smart Customer Support Ticket Classifier")
st.markdown("Automatically classifier support tickets by type and priority using BERT embeddings")
st.divider()

# select mode for single ticket or bulk csv/excel uploading
mode =st.radio(   ## st.radio creates a set of radio buttons so only one option can be selected
    "Select Mode: ",
    ["Single Ticket", "Bulk Upload"],
    horizontal=True
)

## for single mode
if mode == "Single Ticket":
    st.header("Single Classificatin")

    col1, col2 = st.columns([3,1])

    with col1:
        subject = st.text_input("Ticket Subject",
                                placeholder="e.g. App keeps crashing on login")
        description = st.text_area("Ticket Description",
                                   height = 150,
                                   placeholder="Describe the issue in detail...")
        
    with col2:
        channel = st.selectbox("Channel",
                               ["Email", "Chat", "Phone", "Social Media"])
        
    if st.button("Classify Ticket", type ="primary"):
        if subject and description:
            ##combining subject and description like before
            combined = subject.lower().strip() + " "+ description.lower().strip()

            ##embedding the text using encoder (sentence transformer)
            with st.spinner("Analyzing ticket..."):
                embedding = encoder.encode([combined])

            ##predicting the ticket type
            type_pred = type_classifier.predict(embedding)[0]
            type_label = le_type.inverse_transform([type_pred])[0]
            type_proba = type_classifier.predict_proba(embedding)[0]
            type_confidence = type_proba.max() * 100

            ## predicting the ticket priority
            priority_pred = priority_classifier.predict(embedding)[0]
            priority_label = le_priority.inverse_transform([priority_pred])[0]
            priority_proba = priority_classifier.predict_proba(embedding)[0]
            priority_confidence = priority_proba.max() * 100

            st.divider()


            ## displaying the results
            st.subheader("Classification Results: ")
            r1, r2, r3 = st.columns(3)
            
            r1.metric("Ticket Type", type_label)
            r2.metric("Priority", priority_label)
            r3.metric("Channel", channel)
            
            st.progress(int(type_confidence), 
                       text=f"Type Confidence: {type_confidence:.1f}%")
            st.progress(int(priority_confidence), 
                       text=f"Priority Confidence: {priority_confidence:.1f}%")
            
            # routing recommendation
            routing = {
                "Technical Issue": "Engineering Team",
                "Billing Inquiry": "Finance Team", 
                "Refund Request": "Refund Processing Team",
                "Account Access": "Account Management Team",
                "Product Inquiry": "Product Support Team"
            }
            
            st.info(f"**Recommended Routing:** {routing.get(type_label, 'General Support')}")
            
            if priority_confidence < 50:
                st.warning("Low confidence on priority — recommend manual review!!")
                
        else:
            st.error("Please fill in both subject and description.")

else:
    st.header("Bulk Ticket Classification")
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx']
    )
    
    if uploaded_file is not None:
        # load file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Loaded {len(df)} tickets")
        st.dataframe(df.head(), use_container_width=True)
        
        # check required columns
        if 'Ticket Subject' not in df.columns or 'Ticket Description' not in df.columns:
            st.error("File must have 'Ticket Subject' and 'Ticket Description' columns")
        else:
            if st.button("Classify All Tickets", type="primary"):
                with st.spinner(f"Classifying {len(df)} tickets... this may take a minute"):
                    
                    # preprocess the text jaise pehle kiya combine krna
                    df['combine_text'] = (
                        df['Ticket Subject'].str.lower().str.strip() + " " + 
                        df['Ticket Description'].str.lower().str.strip()
                    )
                    
                    # generating the embeddings
                    embeddings = encoder.encode(
                        df['combine_text'].tolist(),
                        batch_size=32,
                        show_progress_bar=False
                    )
                    
                    # predict type
                    type_preds = type_classifier.predict(embeddings)
                    df['Predicted Type'] = le_type.inverse_transform(type_preds)
                    type_probas = type_classifier.predict_proba(embeddings)
                    df['Type Confidence'] = (type_probas.max(axis=1) * 100).round(1)
                    
                    # predict priority
                    priority_preds = priority_classifier.predict(embeddings)
                    df['Predicted Priority'] = le_priority.inverse_transform(priority_preds)
                    priority_probas = priority_classifier.predict_proba(embeddings)
                    df['Priority Confidence'] = (priority_probas.max(axis=1) * 100).round(1)
                    
                    # routing
                    routing_map = {
                        "Technical Issue": "Engineering Team",
                        "Billing Inquiry": "Finance Team",
                        "Refund Request": "Refund Processing Team",
                        "Account Access": "Account Management Team",
                        "Product Inquiry": "Product Support Team"
                    }
                    df['Recommended Team'] = df['Predicted Type'].map(routing_map)
                
                st.success("Classification complete!")
                st.divider()
                
                # summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Tickets", len(df))
                m2.metric("Avg Type Confidence", 
                         f"{df['Type Confidence'].mean():.1f}%")
                m3.metric("High/Critical Priority", 
                         len(df[df['Predicted Priority'].isin(['High','Critical'])]))
                m4.metric("Needs Manual Review", 
                         len(df[df['Priority Confidence'] < 50]))
                
                st.divider()
                
                # charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.bar(
                        df['Predicted Type'].value_counts().reset_index(),
                        x='Predicted Type', y='count',
                        title='Ticket Distribution by Type',
                        color='Predicted Type'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.pie(
                        df, names='Predicted Priority',
                        title='Priority Distribution'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # results table
                st.subheader("Classified Tickets")
                st.dataframe(
                    df[['Ticket Subject', 'Predicted Type', 
                        'Type Confidence', 'Predicted Priority',
                        'Priority Confidence', 'Recommended Team']],
                    use_container_width=True
                )
                
                # download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "classified_tickets.csv",
                    "text/csv"
                )

st.divider()

st.caption("""
    **Model Performance:** Ticket Type: 97.6% accuracy | 
    Priority: 45.1% accuracy (directional guidance only)  
    Built with BERT sentence embeddings + Logistic Regression
""")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Smart Ticket Classifier** uses BERT-based 
    sentence embeddings to automatically classify 
    customer support tickets.
    
    **Models:**
    - Ticket Type: 97.6% accuracy
    - Priority: 45.1% accuracy
    
    **How to use:**
    - Single Ticket: classify one ticket instantly
    - Bulk Upload: process entire CSV/Excel files
    """)