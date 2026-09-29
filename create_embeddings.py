import pandas as pd
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# =========================================================
# LOAD POLICIES
# =========================================================

df = pd.read_csv(
    "company_policies.csv"
)

print(
    f"Found {len(df)} policies."
)

print(
    "Creating Gemini embeddings..."
)


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

saved_policies = []

for index, row in df.iterrows():

    document_text = (
        f"Department: {row['department']}\n"
        f"Category: {row['category']}\n"
        f"Policy: {row['policy_text']}"
    )

    try:

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=document_text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                title=str(row["title"])
            )
        )

        embedding = result.embeddings[0].values

        saved_policies.append(
            {
                "title": row["title"],
                "department": row["department"],
                "category": row["category"],
                "policy_text": row["policy_text"],
                "embedding": embedding
            }
        )

        print(
            f"{index + 1}/{len(df)} - "
            f"{row['title']}"
        )

        # Small pause to reduce API pressure
        time.sleep(0.2)

    except Exception as e:

        print(
            f"ERROR on policy {index + 1}: "
            f"{row['title']}"
        )

        print(e)

        print(
            "Embedding creation stopped."
        )

        break


# =========================================================
# ONLY SAVE IF ALL POLICIES WERE EMBEDDED
# =========================================================

if len(saved_policies) == len(df):

    with open(
        "policy_embeddings.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            saved_policies,
            file
        )

    print()
    print(
        "SUCCESS!"
    )

    print(
        f"Saved {len(saved_policies)} policy embeddings."
    )

    print(
        "File created: policy_embeddings.json"
    )

else:

    print()
    print(
        "The embedding file was NOT created because "
        "not all policies were successfully embedded."
    )