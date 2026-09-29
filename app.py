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

    if any(
        phrase in answer_lower
        for phrase in rejection_phrases
    ):
        return "No unsupported answer"

    if policy in [
        None,
        "None",
        "Not identified",
        "Not available"
    ]:
        return "Potentially unsupported"

    return "No"


# =========================================================
# HEADER
# =========================================================

st.title("📚 Company Policy Assistant")

st.write(
    """
    This application compares three different approaches
    for answering questions about company policies:
    rules-based search, an LLM without a vector index,
    and an LLM with a vector index.
    """
)


# =========================================================
# INFORMATION BOX
# =========================================================

with st.expander("About the three approaches"):

    st.markdown(
        """
        **1. Rules-based search**

        Searches the policy database using keywords from
        the user's question. It does not use an LLM.

        **2. LLM without vector index**

        Sends the complete policy database together with
        the question to the LLM.

        **3. LLM with vector index**

        Uses embeddings to retrieve the most relevant
        policies first. Only the retrieved policies are
        then provided to the LLM.
        """
    )


# =========================================================
# QUESTION FORM
# =========================================================

st.divider()

st.subheader("Ask a policy question")

with st.form("policy_question_form"):

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: How many days per week can I work remotely?"
    )

    submitted = st.form_submit_button(
        "Compare approaches"
    )


# =========================================================
# RUN THE THREE APPROACHES
# =========================================================

if submitted:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        st.divider()

        st.header("Results")

        # -------------------------------------------------
        # APPROACH 1
        # -------------------------------------------------

        with st.spinner(
            "Running rules-based search..."
        ):

            rules_result = rules_based_search(
                question,
                df
            )

        # -------------------------------------------------
        # APPROACH 2
        # -------------------------------------------------

        with st.spinner(
            "Running LLM without vector index..."
        ):

            llm_result = llm_without_vector(
                question,
                df
            )

        if not llm_result.get(
            "error",
            False
        ):
            time.sleep(5)

        # -------------------------------------------------
        # APPROACH 3
        # -------------------------------------------------

        with st.spinner(
            "Running LLM with vector index..."
        ):

            vector_result = llm_with_vector(
                question,
                df
            )


        # =================================================
        # RULES-BASED RESULT
        # =================================================

        st.subheader(
            "1. Rules-Based Search"
        )

        st.write("**Answer:**")

        st.write(
            rules_result["answer"]
        )

        st.write(
            "**Relevant policy:**",
            rules_result["policy"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Response time",
            f"{rules_result['response_time']:.4f} sec"
        )

        col2.metric(
            "Total tokens",
            rules_result["total_tokens"]
        )

        col3.metric(
            "Unsupported answer",
            check_unsupported(
                rules_result["answer"],
                rules_result["policy"],
                rules_result.get(
                    "error",
                    False
                )
            )
        )


        # =================================================
        # LLM WITHOUT VECTOR RESULT
        # =================================================

        st.divider()

        st.subheader(
            "2. LLM Without Vector Index"
        )

        st.write("**Answer:**")

        st.write(
            llm_result["answer"]
        )

        st.write(
            "**Relevant policy:**",
            llm_result["policy"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Response time",
            f"{llm_result['response_time']:.4f} sec"
        )

        col2.metric(
            "Total tokens",
            llm_result["total_tokens"]
        )

        col3.metric(
            "Unsupported answer",
            check_unsupported(
                llm_result["answer"],
                llm_result["policy"],
                llm_result.get(
                    "error",
                    False
                )
            )
        )

        st.caption(
            f"Input tokens: {llm_result['input_tokens']} | "
            f"Output tokens: {llm_result['output_tokens']}"
        )


        # =================================================
        # VECTOR RESULT
        # =================================================

        st.divider()

        st.subheader(
            "3. LLM With Vector Index"
        )

        st.write("**Answer:**")

        st.write(
            vector_result["answer"]
        )

        st.write(
            "**Relevant policy:**",
            vector_result["policy"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Response time",
            f"{vector_result['response_time']:.4f} sec"
        )

        col2.metric(
            "Total tokens",
            vector_result["total_tokens"]
        )

        col3.metric(
            "Unsupported answer",
            check_unsupported(
                vector_result["answer"],
                vector_result["policy"],
                vector_result.get(
                    "error",
                    False
                )
            )
        )

        st.caption(
            f"Input tokens: {vector_result['input_tokens']} | "
            f"Output tokens: {vector_result['output_tokens']}"
        )


        # =================================================
        # RETRIEVED POLICIES
        # =================================================

        retrieved_policies = vector_result.get(
            "retrieved_policies",
            []
        )

        similarity_scores = vector_result.get(
            "similarity_scores",
            []
        )

        if retrieved_policies:

            with st.expander(
                "Policies retrieved by the vector index"
            ):

                for i, policy in enumerate(
                    retrieved_policies
                ):

                    if i < len(
                        similarity_scores
                    ):

                        st.write(
                            f"{i + 1}. "
                            f"{policy} "
                            f"(similarity: "
                            f"{similarity_scores[i]:.3f})"
                        )

                    else:

                        st.write(
                            f"{i + 1}. {policy}"
                        )


        # =================================================
        # COMPARISON TABLE
        # =================================================

        st.divider()

        st.header(
            "Comparison"
        )

        comparison_df = pd.DataFrame(
            [
                {
                    "Approach":
                        "Rules-Based Search",
                    "Relevant Policy":
                        rules_result["policy"],
                    "Response Time (sec)":
                        round(
                            rules_result[
                                "response_time"
                            ],
                            4
                        ),
                    "Input Tokens":
                        rules_result[
                            "input_tokens"
                        ],
                    "Output Tokens":
                        rules_result[
                            "output_tokens"
                        ],
                    "Total Tokens":
                        rules_result[
                            "total_tokens"
                        ],
                    "Unsupported Answer":
                        check_unsupported(
                            rules_result[
                                "answer"
                            ],
                            rules_result[
                                "policy"
                            ],
                            rules_result.get(
                                "error",
                                False
                            )
                        )
                },
                {
                    "Approach":
                        "LLM Without Vector Index",
                    "Relevant Policy":
                        llm_result["policy"],
                    "Response Time (sec)":
                        round(
                            llm_result[
                                "response_time"
                            ],
                            4
                        ),
                    "Input Tokens":
                        llm_result[
                            "input_tokens"
                        ],
                    "Output Tokens":
                        llm_result[
                            "output_tokens"
                        ],
                    "Total Tokens":
                        llm_result[
                            "total_tokens"
                        ],
                    "Unsupported Answer":
                        check_unsupported(
                            llm_result[
                                "answer"
                            ],
                            llm_result[
                                "policy"
                            ],
                            llm_result.get(
                                "error",
                                False
                            )
                        )
                },
                {
                    "Approach":
                        "LLM With Vector Index",
                    "Relevant Policy":
                        vector_result["policy"],
                    "Response Time (sec)":
                        round(
                            vector_result[
                                "response_time"
                            ],
                            4
                        ),
                    "Input Tokens":
                        vector_result[
                            "input_tokens"
                        ],
                    "Output Tokens":
                        vector_result[
                            "output_tokens"
                        ],
                    "Total Tokens":
                        vector_result[
                            "total_tokens"
                        ],
                    "Unsupported Answer":
                        check_unsupported(
                            vector_result[
                                "answer"
                            ],
                            vector_result[
                                "policy"
                            ],
                            vector_result.get(
                                "error",
                                False
                            )
                        )
                }
            ]
        )

        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True
        )