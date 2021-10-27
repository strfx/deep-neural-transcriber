"""
Translate transcripts using commercial translation service DeepL.
"""
from typing import Protocol

import requests  # type: ignore

DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"


class Translator(Protocol):

    def translate(self, text: str, target_lang: str = 'DE') -> str:
        # TODO: Should there be a default implementation that NOPs?
        pass


class DeepL(Translator):
    """
    Translate text using the DeepL API.

    You can find the DeepL's complete API documentation at:
    https://www.deepl.com/en/docs-api/

    You can get a free API key when creating an account. The free account
    allows to translate up to 500k characters per month (as of June, 2021).
    This should be enough for demo purposes.

    If you want to test the API, you can use cURL like:
        curl https://api-free.deepl.com/v2/translate \ 
            -d auth_key=<your API key> \ 
            -d "text=Hello, world!"  \ 
            -d "target_lang=DE"

    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def translate(self, text: str, target_lang: str = 'DE') -> str:
        """
        Translate text using the API.

        Args:
            text: The source text you want to translate.
            target_lang: Language code for the desired translation direction.

        Raises:
            HTTPError, if the API call went wrong (i.e, HTTP status code is not 
            2xx or something went wrong before the HTTP request could be
            established).

        Returns:
            The text translated in the traget language.

        """
        response = requests.post(url=DEEPL_API_URL,
                                 data={
                                     'target_lang': target_lang,
                                     'auth_key': self.api_key,
                                     'text': text,
                                 })

        response.raise_for_status()
        translation = response.json()

        # API response JSON looks like:
        # "translations": [{
        # 		"detected_source_language":"EN",
        # 		"text":"Hallo, Welt!"
        # 	}]
        # }

        return translation['translations'][0]['text']


class NopTranslator(Translator):
    """
    Translator that skips translation.

    Mostly used for testing or when you don't want to drain your DeepL API
    plan.
    """

    def translate(self, text: str, target_lang: str = 'DE') -> str:
        return text
