from typing import List
import regex
import unicodedata


# -----------------------------------------------------------------------------------------------
class Normalizer:
    def __init__(self, lang):
        self.lang = lang
    
    def normalize_arabic(self, text: str) -> str:
        # Remove Arabic diacritics (tashkeel)
        text = regex.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)

        # Normalize Alef variants
        text = regex.sub("[إأآٱ]", "ا", text)

        # Normalize Yeh & Alef Maqsura
        text = regex.sub("[ىئ]", "ي", text)

        # Normalize Waw with Hamza
        text = regex.sub("ؤ", "و", text)

        # Normalize Teh Marbuta
        text = regex.sub("ة", "ه", text)

        # Remove Tatweel
        text = regex.sub("ـ", "", text)

        return text

    def __call__(self, transcripts: List[str]) -> List[str]:
        normalized_transcripts = []

        for text in transcripts:

            # unicode normalization
            text = unicodedata.normalize("NFC", text)

            # convert to lowercase
            text = text.lower()

            # remove Unicode punctuation only
            text = regex.sub(r"\p{P}+", "", text)

            # normalize whitespace (preserve ZWNJ)
            text = regex.sub(r"[ \t\r\n]+", " ", text)

            # strip leading/trailing whitespace
            text = text.strip()

            if self.lang == "ar":
                text = self.normalize_arabic(text)

            normalized_transcripts.append(text)

        return normalized_transcripts


# -----------------------------------------------------------------------------------------------
