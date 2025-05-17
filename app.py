import streamlit as st
import pandas as pd
import numpy as np # Using numpy for simulating data issues

# --- Configuration ---
st.set_page_config(page_title="AI-Enhanced CRM Data Accuracy", layout="wide")

# --- Helper Functions (Simulating AI/ML checks) ---

def identify_potential_duplicates(df):
    """
    Simulates identifying potential duplicate records based on Name and Email.
    In a real app, this would use more sophisticated matching algorithms (e.g., fuzzy matching, machine learning).
    """
    st.subheader("Potential Duplicates")
    if df.empty:
        st.warning("No data loaded to check for duplicates.")
        return pd.DataFrame()

    # Simple example: Group by Name and Email and find groups with more than one record
    duplicate_groups = df.groupby(['Name', 'Email']).filter(lambda x: len(x) > 1)

    if not duplicate_groups.empty:
        st.write("The following records might be duplicates:")
        st.dataframe(duplicate_groups)
    else:
        st.info("No potential duplicates found based on simple Name and Email matching.")

    return duplicate_groups

def check_data_consistency(df):
    """
    Simulates checking for data consistency issues (e.g., invalid email format, inconsistent phone numbers).
    In a real app, this would involve regex checks, data type validation, etc.
    """
    st.subheader("Data Consistency Issues")
    if df.empty:
        st.warning("No data loaded to check consistency.")
        return pd.DataFrame()

    issues = []
    # Simulate checking for missing emails
    missing_emails = df[df['Email'].isnull()]
    if not missing_emails.empty:
        issues.append(("Missing Email", missing_emails))

    # Simulate checking for potentially invalid phone numbers (simple check)
    # Assuming phone numbers should be strings and not empty
    invalid_phones = df[df['Phone'].isnull() | (df['Phone'].astype(str).str.len() < 5)] # Example check
    if not invalid_phones.empty:
        issues.append(("Potentially Invalid Phone", invalid_phones))

    if issues:
        st.write("The following data consistency issues were found:")
        for issue_type, issue_df in issues:
            st.write(f"- **{issue_type}**")
            st.dataframe(issue_df)
    else:
        st.info("No major consistency issues found based on simple checks.")

    # Combine all issue dataframes for potential export/review
    all_issues_df = pd.concat([issue_df for _, issue_df in issues]) if issues else pd.DataFrame()
    return all_issues_df

def suggest_corrections(df):
    """
    Simulates suggesting corrections (e.g., filling missing values, standardizing formats).
    In a real app, this would use AI/ML models for imputation or standardization.
    """
    st.subheader("Suggested Corrections")
    if df.empty:
        st.warning("No data loaded to suggest corrections.")
        return pd.DataFrame()

    st.info("This section would display AI-suggested corrections (e.g., imputing missing values, standardizing formats).")
    st.write("Example: Suggesting to fill missing 'City' values or standardize 'State' abbreviations.")

    # Simulate a simple suggestion: fill missing 'City' with 'Unknown'
    df_corrected = df.copy()
    if 'City' in df_corrected.columns:
        missing_city_count = df_corrected['City'].isnull().sum()
        if missing_city_count > 0:
            st.write(f"- Suggesting to fill {missing_city_count} missing 'City' values.")
            # In a real scenario, you might show the rows affected or allow user review
            # df_corrected['City'] = df_corrected['City'].fillna('Unknown') # This would modify the dataframe

    return df_corrected # Return the potentially corrected dataframe (or show diff)


# --- Main App Logic ---

st.title("AI-Enhanced Data Accuracy in CRM Systems")

st.write("""
This application demonstrates how AI and machine learning techniques can be used
to improve the accuracy and consistency of your CRM data.
Upload your CRM data (CSV format) to get started.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your CRM data (CSV)", type="csv")

df = pd.DataFrame() # Initialize an empty dataframe

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Data Preview:")
        st.dataframe(df.head())

        # --- Data Accuracy Checks ---
        st.header("Data Accuracy Analysis")

        potential_duplicates_df = identify_potential_duplicates(df.copy())
        consistency_issues_df = check_data_consistency(df.copy())
        df_with_suggestions = suggest_corrections(df.copy())

        # --- Actionable Insights/Export ---
        st.header("Actionable Insights")
        if not potential_duplicates_df.empty or not consistency_issues_df.empty:
            st.write("Review the findings above to improve your data quality.")
            # Option to download the identified issues
            all_issues_for_download = pd.concat([potential_duplicates_df, consistency_issues_df]).drop_duplicates()
            if not all_issues_for_download.empty:
                 csv = all_issues_for_download.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     label="Download Identified Issues (CSV)",
                     data=csv,
                     file_name='crm_data_issues.csv',
                     mime='text/csv',
                 )
        else:
            st.info("No significant data quality issues found based on the checks performed.")


    except Exception as e:
        st.error(f"Error loading or processing file: {e}")

else:
    st.info("Please upload a CSV file to analyze.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit")

