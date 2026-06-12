def get_disclaimer():
    """Return standard disclaimers for the breast cancer recommender system."""
    return {
        "type": "medical_disclaimer",
        "title": "Important Disclaimer",
        "statements": [
            "These are statistical recommendations based on a machine learning model — NOT medical advice.",
            "The model estimates a proxy outcome and does not prove that a treatment causes better outcomes.",
            "The model has not been externally validated or cleared for clinical decision-making.",
            "Quality depends entirely on the trained model and preprocessing; test and validate with clinical oversight before use.",
            "Do not use as a substitute for professional clinical judgment or consultation with qualified healthcare providers.",
            "Always consult with oncologists and clinical teams before making treatment decisions.",
            "Results may not apply to all patient populations; validate on your specific clinical context.",
            "The system is provided as-is; the developers assume no liability for outcomes."
        ]
    }
