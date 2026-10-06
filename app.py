from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from profiler import profile_dataset

from pii_detector import detect_pii

from privacy_analyzer import analyze_privacy_risks

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MAX_ROWS = 10_000
MAX_COLUMNS = 100


@app.route("/", methods=["GET", "POST"])
def index():
    dataset_info = None
    profile = None
    pii_findings = []
    privacy_report = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a CSV file."

        elif Path(uploaded_file.filename).suffix.lower() != ".csv":
            error = "Only CSV files are allowed."

        elif not secure_filename(uploaded_file.filename):
            error = "Invalid filename."

        else:
            try:
                df = pd.read_csv(
                    uploaded_file.stream,
                    nrows=MAX_ROWS + 1,
                )

                if len(df) > MAX_ROWS:
                    error = f"CSV files cannot exceed {MAX_ROWS} rows."

                elif len(df.columns) == 0:
                    error = "The CSV must contain columns."

                elif len(df.columns) > MAX_COLUMNS:
                    error = (
                        f"CSV files cannot exceed "
                        f"{MAX_COLUMNS} columns."
                    )

                else:
                    dataset_info = {
                        "filename": secure_filename(
                            uploaded_file.filename
                        )
                    }

                    profile = profile_dataset(df)
                    pii_findings = detect_pii(df)
                    privacy_report = analyze_privacy_risks(df)

            except (
                ValueError,
                UnicodeDecodeError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
            ):
                error = "Unable to read this file as a valid CSV."

            except Exception:
                app.logger.exception("CSV processing failed.")
                error = (
                    "An unexpected error occurred "
                    "while processing the CSV."
                )

    return render_template(
        "index.html",
        dataset_info=dataset_info,
        profile=profile,
        pii_findings=pii_findings,
        privacy_report=privacy_report,
        error=error,
    )

@app.errorhandler(413)
def file_too_large(_error):
    return (
        render_template(
            "index.html",
            dataset_info=None,
            profile=None,
            pii_findings=[],
            privacy_report=None,
            error="File exceeds the 5 MB upload limit.",
        ),
        413,
    )

if __name__ == "__main__":
    app.run(debug=True)