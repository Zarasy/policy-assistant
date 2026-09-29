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

    answer_lower = str(answer).lower()

    rejection_phrases = [
        "no relevant policy was found",
        "does not contain enough information",
        "not enough information",
        "cannot be found",
        "not available"
    ]

    for phrase in rejection_phrases:
        if phrase in answer_lower:
            return "No unsupported answer"

    if (
        policy == "None"
        or policy == "Not identified"
        or policy == "Not available"
    ):
        return "Potentially unsupported"

    return "No"


# =========================================================
# HEADER
# =========================================================

st.title("Company Policy Assistant")

st.write(
    "Compare three approaches for answering questions "
    "using a database of company policies."
)

st.caption(
    f"Policy database: {len(df)} company policies"
)


# =========================================================
# EXPLAIN THE THREE METHODS
# =========================================================

with st.expander("How do the three approaches work?"):

    st.markdown(
        """
        **1. Rules-Based Search**  
        Searches the policy database using keywords from the
        user's question. No large language model is used.

        **2. LLM Without Vector Index**  
        Sends the complete policy database together with the
        user's question to Gemini.

        **3. LLM With Vector Index**  
        Uses Gemini Embeddings and cosine similarity to retrieve
        the three most relevant policies. Only those policies are
        then sent to Gemini to generate the final answer.
        """
    )


# =========================================================
# QUESTION FORM
# =========================================================

st.subheader("Ask a policy question")

with st.form("policy_question_form"):

    question = st.text_input(
        "Question",
        placeholder="Example: How many days can I work remotely?"
    )

    submitted = st.form_submit_button(
        "Run comparison",
        type="primary"
    )


# =========================================================
# RUN COMPARISON
# =========================================================

if submitted:

    if not question.strip():

        st.warning(
            "Please enter a policy question."
        )

    else:

        st.divider()

        st.subheader("Results")

        # =================================================
        # APPROACH 1
        # =================================================

        rules_result = rules_based_search(
            question,
            df
        )

        rules_unsupported = check_unsupported(
            rules_result["answer"],
            rules_result["policy"],
            rules_result["error"]
        )

        # =================================================
        # APPROACH 2
        # =================================================

        with st.spinner(
            "Running LLM without vector index..."
        ):

            llm_result = llm_without_vector(
                question,
                df
            )

        llm_unsupported = check_unsupported(
            llm_result["answer"],
            llm_result["policy"],
            llm_result["error"]
        )

        # Small pause between Gemini LLM calls
        if not llm_result["error"]:
            time.sleep(5)

        # =================================================
        # APPROACH 3
        # =================================================

        with st.spinner(
            "Running embedding-based vector retrieval..."
        ):

            vector_result = llm_with_vector(
                question,
                df
            )

        vector_unsupported = check_unsupported(
            vector_result["answer"],
            vector_result["policy"],
            vector_result["error"]
        )


        # =================================================
        # APPROACH 1 DISPLAY
        # =================================================

        st.header("1. Rules-Based Search")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Response time",
                f'{rules_result["response_time"]:.4f} s'
            )

        with col2:
            st.metric(
                "Total tokens",
                "0"
            )

        with col3:
            st.metric(
                "Unsupported answer",
                rules_unsupported
            )

        st.markdown("**Answer**")
        st.write(
            rules_result["answer"]
        )

        st.markdown("**Relevant policy**")
        st.write(
            rules_result["policy"]
        )


        # =================================================
        # APPROACH 2 DISPLAY
        # =================================================

        st.divider()

        st.header(
            "2. LLM Without Vector Index"
        )

        if llm_result["error"]:

            st.warning(
                llm_result["answer"]
            )

            st.info(
                "This approach could not be evaluated because "
                "the Gemini LLM API was unavailable or its "
                "quota had been reached."
            )

        else:

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Response time",
                    f'{llm_result["response_time"]:.4f} s'
                )

            with col2:
                st.metric(
                    "Total tokens",
                    llm_result["total_tokens"]
                )

            with col3:
                st.metric(
                    "Unsupported answer",
                    llm_unsupported
                )

            st.markdown("**Answer**")
            st.write(
                llm_result["answer"]
            )

            st.markdown(
                "**Relevant policy**"
            )
            st.write(
                llm_result["policy"]
            )

            st.markdown(
                "**Token details**"
            )

            st.write(
                f'Input: {llm_result["input_tokens"]} | '
                f'Output: {llm_result["output_tokens"]} | '
                f'Total: {llm_result["total_tokens"]}'
            )


        # =================================================
        # APPROACH 3 DISPLAY
        # =================================================

        st.divider()

        st.header(
            "3. LLM With Vector Index"
        )

        if vector_result["error"]:

            st.warning(
                vector_result["answer"]
            )

            if vector_result[
                "retrieved_policies"
            ]:

                st.success(
                    "Vector retrieval was completed successfully."
                )

                st.markdown(
                    "**Policies retrieved using Gemini Embeddings**"
                )

                for number, policy in enumerate(
                    vector_result[
                        "retrieved_policies"
                    ],
                    start=1
                ):

                    st.write(
                        f"{number}. {policy}"
                    )

            st.info(
                "The final LLM answer could not be evaluated "
                "because Gemini 3.8 Flash was unavailable or "
                "its quota had been reached."
            )

        else:

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Response time",
                    f'{vector_result["response_time"]:.4f} s'
                )

            with col2:
                st.metric(
                    "Total tokens",
                    vector_result[
                        "total_tokens"
                    ]
                )

            with col3:
                st.metric(
                    "Unsupported answer",
                    vector_unsupported
                )

            st.markdown("**Answer**")
            st.write(
                vector_result["answer"]
            )

            st.markdown(
                "**Relevant policy**"
            )
            st.write(
                vector_result["policy"]
            )

            st.markdown(
                "**Policies retrieved using Gemini Embeddings**"
            )

            for number, policy in enumerate(
                vector_result[
                    "retrieved_policies"
                ],
                start=1
            ):

                st.write(
                    f"{number}. {policy}"
                )

            st.markdown(
                "**Token details**"
            )

            st.write(
                f'Input: {vector_result["input_tokens"]} | '
                f'Output: {vector_result["output_tokens"]} | '
                f'Total: {vector_result["total_tokens"]}'
            )


        # =================================================
        # COMPARISON TABLE
        # =================================================

        st.divider()

        st.header("Comparison")

        comparison_data = {
            "Approach": [
                "Rules-Based Search",
                "LLM Without Vector Index",
                "LLM With Vector Index"
            ],

            "Response Time (s)": [
                round(
                    rules_result[
                        "response_time"
                    ],
                    4
                ),

                (
                    round(
                        llm_result[
                            "response_time"
                        ],
                        4
                    )
                    if not llm_result["error"]
                    else "N/A"
                ),

                (
                    round(
                        vector_result[
                            "response_time"
                        ],
                        4
                    )
                    if not vector_result["error"]
                    else "N/A"
                )
            ],

            "Total Tokens": [
                0,

                (
                    llm_result[
                        "total_tokens"
                    ]
                    if not llm_result["error"]
                    else "N/A"
                ),

                (
                    vector_result[
                        "total_tokens"
                    ]
                    if not vector_result["error"]
                    else "N/A"
                )
            ],

            "Relevant Policy": [
                rules_result["policy"],

                (
                    llm_result["policy"]
                    if not llm_result["error"]
                    else "N/A"
                ),

                (
                    vector_result["policy"]
                    if not vector_result["error"]
                    else "N/A"
                )
            ],

            "Unsupported Answer": [
                rules_unsupported,
                llm_unsupported,
                vector_unsupported
            ]
        }

        comparison_df = pd.DataFrame(
            comparison_data
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # INTERPRETATION PLACEHOLDER
        # =================================================

        st.divider()

        st.header("Evaluation")

        st.info(
            "The final two-paragraph comparison will be added "
            "after the approaches have been tested on both "
            "supported and unsupported policy questions."
        )