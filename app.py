import streamlit as st

import pandas as pd

import time



from policy_engine import (

    rules_based_search,

    llm_without_vector,

    llm_with_vector

)





# =========================================================

# PAGE CONFIGURATION

# =========================================================



st.set_page_config(

    page_title="Company Policy Assistant",

    page_icon="📚",

    layout="wide"

)





# =========================================================

# LOAD DATA

# =========================================================



df = pd.read_csv("company_policies.csv")





# =========================================================

# HELPER FUNCTION

# =========================================================



def check_unsupported(answer, policy, error=False):

    """

    Simple indicator used for the comparison table.



    Yes = the method did not find sufficient support in the

    policy database.



    No = the method returned an answer linked to a policy.



    N/A = the API failed, so the result cannot be evaluated.

    """



    if error:

        return "N/A"


