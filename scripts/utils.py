import re

def clean_text(text):
    """
    Minimal text cleaning function for BERT.

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9.,!?\'\`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
