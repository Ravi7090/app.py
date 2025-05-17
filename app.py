{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "93a0dae3-aafa-463e-ab8a-57efd450b800",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import re\n",
    "from sklearn.ensemble import IsolationForest\n",
    "\n",
    "st.title(\"🔍 AI-Enhanced Data Accuracy in CRM Systems\")\n",
    "\n",
    "st.markdown(\"Upload a CRM CSV file and this app will flag anomalies in email and phone data using AI-enhanced techniques.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Upload CRM Data (CSV)\", type=\"csv\")\n",
    "\n",
    "def is_valid_email(email):\n",
    "    return re.match(r\"[^@]+@[^@]+\\.[^@]+\", str(email))\n",
    "\n",
    "def is_valid_phone(phone):\n",
    "    return re.match(r\"^\\+?\\d{7,15}$\", str(phone))\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    df = pd.read_csv(uploaded_file)\n",
    "    st.subheader(\"📋 Original Data\")\n",
    "    st.dataframe(df)\n",
    "\n",
    "    # Basic Rule-based validation\n",
    "    df[\"Email_Valid\"] = df[\"Email\"].apply(is_valid_email)\n",
    "    df[\"Phone_Valid\"] = df[\"Phone\"].apply(is_valid_phone)\n",
    "\n",
    "    # AI-based anomaly detection using Isolation Forest\n",
    "    features = df.select_dtypes(include=['float64', 'int64']).fillna(0)\n",
    "    if not features.empty:\n",
    "        model = IsolationForest(contamination=0.1)\n",
    "        df[\"Anomaly\"] = model.fit_predict(features)\n",
    "        df[\"Anomaly\"] = df[\"Anomaly\"].map({1: \"Normal\", -1: \"Anomaly\"})\n",
    "    else:\n",
    "        df[\"Anomaly\"] = \"N/A (no numeric features)\"\n",
    "\n",
    "    st.subheader(\"✅ Cleaned & Analyzed Data\")\n",
    "    st.dataframe(df)\n",
    "\n",
    "    csv = df.to_csv(index=False).encode(\"utf-8\")\n",
    "    st.download_button(\"Download Processed CSV\", csv, \"cleaned_crm_data.csv\", \"text/csv\")\n",
    "else:\n",
    "    st.info(\"Please upload a CSV file with CRM data. Required columns: `Email`, `Phone`, plus optional numeric fields.\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71999040-88b7-4cab-b5f2-6584026cd82e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
