import re
import time
import os
import json
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types

from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()


# =========================================================
# GEMINI CLIENT
# Works locally with .env and online with Streamlit Secrets
# =========================================================

def get_api_key():

    # First try local environment / .env
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    # If running on Streamlit Community Cloud,
    # try Streamlit Secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]

        if api_key:
            return api_key

    except Exception:
        pass

    return None


def get_client():

    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Add it to .env locally or Streamlit Secrets online."
        )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# GEMINI LLM WITH RETRY
# =========================================================

def call_gemini_with_retry(
    client,
    prompt,
    max_retries=3
):

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            return response

        except Exception as e:

            error_message = str(e)

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < max_retries - 1:

                    wait_time = 5 * (
                        attempt + 1
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

            return None

    return None


# =========================================================
# LOAD SAVED POLICY EMBEDDINGS
# =========================================================

def load_policy_embeddings():

    with open(
        "policy_embeddings.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# CREATE EMBEDDING FOR QUESTION
# =========================================================

def create_question_embedding(
    client,
    question
):

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    return result.embeddings[0].values


# =========================================================
# APPROACH 1
# RULES-BASED SEARCH
# =========================================================

def rules_based_search(
    question,
    df
):

    start_time = time.time()

    question_lower = question.lower()

    question_words = re.findall(
        r"\b\w+\b",
        question_lower
    )

    stop_words = {
        "the", "a", "an", "is", "are",
        "can", "i", "we", "you", "how",
        "what", "when", "where", "who",
        "many", "much", "do", "does",
        "to", "of", "in", "for", "and",
        "on", "my"
    }

    keywords = [
        word
        for word in question_words
        if word not in stop_words
        and len(word) > 2
    ]

    best_score = 0
    best_policy = None

    for _, row in df.iterrows():

        searchable_text = (
            str(row["title"]) + " " +
            str(row["department"]) + " " +
            str(row["category"]) + " " +
            str(row["policy_text"])
        ).lower()

        score = sum(
            1
            for keyword in keywords
            if keyword in searchable_text
        )

        if score > best_score:

            best_score = score
            best_policy = row

    response_time = (
        time.time() - start_time
    )

    if best_policy is None:

        return {
            "answer":
                "No relevant policy was found.",
            "policy": "None",
            "response_time":
                response_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": False
        }

    return {
        "answer":
            best_policy["policy_text"],
        "policy":
            best_policy["title"],
        "response_time":
            response_time,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "error": False
    }


# =========================================================
# APPROACH 2
# LLM WITHOUT VECTOR INDEX
# =========================================================

def llm_without_vector(
    question,
    df
):

    start_time = time.time()

    client = get_client()

    policies = ""

    for _, row in df.iterrows():

        policies += f"""
Policy title: {row['title']}
Department: {row['department']}
Category: {row['category']}
Policy text: {row['policy_text']}

"""

    prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the company policies
provided below.

Do not use outside knowledge.

If the answer cannot be found in the policies, say:
"The policy database does not contain enough information to answer this question."

Identify the single most relevant policy.

Return your response exactly in this format:

ANSWER: [your answer]
POLICY: [policy title]

COMPANY POLICIES:

{policies}

USER QUESTION:

{question}
"""

    response = call_gemini_with_retry(
        client,
        prompt
    )

    response_time = (
        time.time() - start_time
    )

    if response is None:

        return {
            "answer":
                "Gemini 3.5 Flash-Lite is currently unavailable or its quota has been reached.",
            "policy":
                "Not available",
            "response_time":
                response_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error": True
        }

    text = response.text

    answer = text
    policy = "Not identified"

    if "ANSWER:" in text:

        answer = text.split(
            "ANSWER:",
            1
        )[1]

        if "POLICY:" in answer:

            answer, policy = (
                answer.split(
                    "POLICY:",
                    1
                )
            )

            answer = answer.strip()
            policy = policy.strip()

    usage = response.usage_metadata

    return {
        "answer": answer,
        "policy": policy,
        "response_time":
            response_time,
        "input_tokens":
            usage.prompt_token_count,
        "output_tokens":
            usage.candidates_token_count,
        "total_tokens":
            usage.total_token_count,
        "error": False
    }


# =========================================================
# APPROACH 3
# LLM WITH SAVED GEMINI VECTOR INDEX
# =========================================================

def llm_with_vector(
    question,
    df
):

    start_time = time.time()

    client = get_client()

    try:

        saved_policies = (
            load_policy_embeddings()
        )

    except Exception:

        return {
            "answer":
                "policy_embeddings.json was not found. Run create_embeddings.py first.",
            "policy":
                "Not available",
            "response_time":
                time.time() - start_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retrieved_policies": [],
            "similarity_scores": [],
            "error": True
        }

    try:

        question_embedding = (
            create_question_embedding(
                client,
                question
            )
        )

    except Exception:

        return {
            "answer":
                "Gemini could not create the question embedding.",
            "policy":
                "Not available",
            "response_time":
                time.time() - start_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retrieved_policies": [],
            "similarity_scores": [],
            "error": True
        }

    policy_vectors = [
        policy["embedding"]
        for policy
        in saved_policies
    ]

    similarities = cosine_similarity(
        [question_embedding],
        policy_vectors
    ).flatten()

    top_indices = (
        similarities.argsort()[
            -3:
        ][::-1]
    )

    retrieved_titles = []
    similarity_scores = []
    relevant_policies = ""

    for index in top_indices:

        policy_data = (
            saved_policies[index]
        )

        retrieved_titles.append(
            policy_data["title"]
        )

        similarity_scores.append(
            float(
                similarities[index]
            )
        )

        relevant_policies += f"""
Policy title: {policy_data['title']}
Department: {policy_data['department']}
Category: {policy_data['category']}
Policy text: {policy_data['policy_text']}

"""

    prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the retrieved company
policies provided below.

Do not use outside knowledge.

If the answer cannot be found in the retrieved policies, say:
"The policy database does not contain enough information to answer this question."

Identify the single most relevant policy.

Return your response exactly in this format:

ANSWER: [your answer]
POLICY: [policy title]

RETRIEVED POLICIES:

{relevant_policies}

USER QUESTION:

{question}
"""

    response = call_gemini_with_retry(
        client,
        prompt
    )

    response_time = (
        time.time() - start_time
    )

    if response is None:

        return {
            "answer":
                "Vector retrieval completed successfully, but Gemini 3.5 Flash-Lite is currently unavailable or its quota has been reached.",
            "policy":
                retrieved_titles[0],
            "response_time":
                response_time,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retrieved_policies":
                retrieved_titles,
            "similarity_scores":
                similarity_scores,
            "error": True
        }

    text = response.text

    answer = text
    policy = "Not identified"

    if "ANSWER:" in text:

        answer = text.split(
            "ANSWER:",
            1
        )[1]

        if "POLICY:" in answer:

            answer, policy = (
                answer.split(
                    "POLICY:",
                    1
                )
            )

            answer = answer.strip()
            policy = policy.strip()

    usage = response.usage_metadata

    return {
        "answer":
            answer,
        "policy":
            policy,
        "response_time":
            response_time,
        "input_tokens":
            usage.prompt_token_count,
        "output_tokens":
            usage.candidates_token_count,
        "total_tokens":
            usage.total_token_count,
        "retrieved_policies":
            retrieved_titles,
        "similarity_scores":
            similarity_scores,
        "error":
            False
    }