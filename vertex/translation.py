#!/usr/bin/env python3

############################################
# Registry for different languages
############################################

LANGUAGE_REGISTRY = {
    "fr": {
        "Yes": "Oui",
        "No": "Non",
        "Unknown": "Inconnu",
        "Other": "Autre",
        "Female": "Femme",
        "Male": "Homme",
        "Years": "Années",
        "Months": "Mois",
        "Days": "Jours",
        "Presentation": "Presentation",
        "Daily": "Quotidien",
        "Outcome": "Résultat",
        "Discharged alive": "Sorti vivant",
        "Discharged": "Sorti vivant",
        "Discharged against medical advice": "Sorti contre l'avis médical",
        "Death": "Décès",
        "Palliative care": "Soins palliatifs",
        "Censored": "Censuré",
    },
}


REMOVE_ACCENT_MAPPING_DICT = {
    "À": "A", "Á": "A", "Â": "A", "Ã": "A", "Ä": "A", "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "ª": "A",
    "È": "E", "É": "E", "Ê": "E", "Ë": "E", "è": "e", "é": "e", "ê": "e", "ë": "e",
    "Í": "I", "Ì": "I", "Î": "I", "Ï": "I", "í": "i", "ì": "i", "î": "i", "ï": "i",
    "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O", "Ö": "O", "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o",
    "Ù": "U", "Ú": "U", "Û": "U", "Ü": "U", "ù": "u", "ú": "u", "û": "u", "ü": "u",
    "Ñ": "N", "ñ": "n",
    "Ç": "C", "ç": "c"
}


def translated_phrase(phrase, language):
    if language not in LANGUAGE_REGISTRY:
        raise ValueError(f"Unknown method: {language}")
    if phrase not in LANGUAGE_REGISTRY[language]:
        raise ValueError(f"Unknown language-specific function in {language}: {phrase}")
    return LANGUAGE_REGISTRY[language][phrase]


def translate(phrase, language):
    return phrase if (language == "en") else translated_phrase(phrase, language)


def remove_accents(string):
    '''Solution from https://stackoverflow.com/questions/65833714/how-to-remove-accents-from-a-string-in-python'''
    normalize = str.maketrans(REMOVE_ACCENT_MAPPING_DICT)
    return string.translate(normalize)
