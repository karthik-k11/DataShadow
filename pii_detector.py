import re

import pandas as pd


COLUMN_RULES = {
    "name": "Potential Name",
    "email": "Potential Email",
    "phone": "Potential Phone Number",
    "mobile": "Potential Phone Number",
    "address": "Potential Address",
    "aadhaar": "Potential Government ID",
    "ssn": "Potential Government ID",
    "passport": "Potential Government ID",
    "pan": "Potential Government ID",
}


def detect_pii(df):
    findings = []

    for column in df.columns:
        if df[column].dropna().empty:
            continue

        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(column).lower(),
        )

        detected_type = None

        for keyword, label in COLUMN_RULES.items():
            if keyword in normalized:
                detected_type = label
                break

        if detected_type is None:
            sample = (
                df[column]
                .dropna()
                .astype(str)
                .head(100)
            )

            if sample.empty:
                continue

            email_matches = sample.str.match(
                r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
            )

            if email_matches.mean() >= 0.8:
                detected_type = "Potential Email"

            else:
                phone_matches = sample.str.match(
                    r"^\+?[\d\s().-]{7,20}$"
                )

                if phone_matches.mean() >= 0.8:
                    detected_type = "Potential Phone Number"

        if detected_type:
            findings.append(
                {
                    "column": str(column),
                    "type": detected_type,
                    "confidence": "Rule-based indicator",
                }
            )

    return findings