from datasets import load_dataset
import pandas as pd
import ast
import re

df = load_dataset("modelbiome/ai_ecosystem_withmodelcards", split="train").to_pandas()

useful_columns = [
    "model_id",
    "likes",
    "downloads",
    "tags",
    "pipeline_tag",
    "library_name",
    "createdAt",
    "languages",
    "modelCard"
]
df = df[useful_columns]

df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
df = df.sort_values(by=["downloads", "likes"], ascending=False)

# delete empty models and models with useless description
df = df[df["modelCard"].fillna("").str.len() > 20]

pattern = r"<!--\s*This model card has been generated automatically"
df = df[~df["modelCard"].str.contains(pattern, na=False, regex=True)]


# save only 600k samples: 75% random, 25% most popular
popular = df.head(150_000)
remaining = df.drop(popular.index)
random = remaining.sample(n=450_000, random_state=42)
df = pd.concat([popular, random])

# remove duplicates just in case
df = df.drop_duplicates(subset=["model_id"])

# fix the columns that was read as strings while actually being lists
for col in ['tags', 'languages']:
    df[col] = df[col].apply(ast.literal_eval)

# convert the date and remove the time
df['createdAt'] = pd.to_datetime(df['createdAt']).dt.date


def clean_model_card(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. Remove code blocks (``` ... ```)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # 2. Remove markdown headers
    text = re.sub(r"#+\s*", " ", text)

    # 3. Remove markdown links [text](url)
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)

    # 4. Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # 5. Remove bullet symbols
    text = re.sub(r"[-*•`]", " ", text)

    # 6. Normalize spaces but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)

    # 7. Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8. Remove html formatting
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()


df["modelCard"] = df["modelCard"].apply(clean_model_card)

df.to_parquet("clean_dataset.parquet", index=False)

