
import json
import os
import secrets
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from profiler import profile_dataset
from pii_detector import detect_pii
from privacy_analyzer import analyze_privacy_risks
from synthetic_generator import generate_synthetic_data, describe_generation
from utility_evaluator import compare_utility
from privacy_evaluator import evaluate_privacy
from history_db import clear_history, get_history, save_analysis
from data_quality import analyze_data_quality


app = Flask(__name__)

INSTANCE_DIR = Path("instance")
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
SECRET_KEY_PATH = INSTANCE_DIR / "datashadow_secret_key"


def load_secret_key():
    """Load a stable Flask session signing key."""
    env_key = os.environ.get("DATASHADOW_SECRET_KEY")

    if env_key:
        return env_key

    try:
        with SECRET_KEY_PATH.open("x", encoding="utf-8") as key_file:
            key = secrets.token_hex(32)
            key_file.write(key)
            return key
    except FileExistsError:
        key = SECRET_KEY_PATH.read_text(encoding="utf-8").strip()

        if not key:
            raise RuntimeError("The DataShadow secret key file is empty.")

        return key


app.config["SECRET_KEY"] = load_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MAX_ROWS = 10_000
MAX_COLUMNS = 100
MAX_CACHED_EXPORTS = 5
ACTIVE_DATASET_TTL_SECONDS = 30 * 60

# Temporary application memory only. Original CSV data is not stored in SQLite.
ACTIVE_DATASETS = {}
ACTIVE_GENERATIONS = {}
SYNTHETIC_EXPORTS = {}
REPORT_EXPORTS = {}
SYNTHETIC_EXPORT_EXPIRY = {}
REPORT_EXPORT_EXPIRY = {}
EXPORT_TTL_SECONDS = 30 * 60


def safe_csv_bytes(df):
    """Protect exported text cells against spreadsheet formula injection."""
    safe_df = df.copy()
    dangerous_prefixes = ("=", "+", "-", "@", "\t", "\r")

    for column in safe_df.columns:
        if (
            pd.api.types.is_object_dtype(safe_df[column])
            or pd.api.types.is_string_dtype(safe_df[column])
        ):
            safe_df[column] = safe_df[column].map(
                lambda value: (
                    "'" + value
                    if isinstance(value, str)
                    and value.startswith(dangerous_prefixes)
                    else value
                )
            )

    return safe_df.to_csv(index=False).encode("utf-8")


def cleanup_active_datasets():
    """Remove expired datasets and their generated results from memory."""
    now = time.time()

    expired_tokens = [
        token
        for token, entry in ACTIVE_DATASETS.items()
        if now - entry["last_access"] > ACTIVE_DATASET_TTL_SECONDS
    ]

    for token in expired_tokens:
        ACTIVE_DATASETS.pop(token, None)
        ACTIVE_GENERATIONS.pop(token, None)


def get_active_dataset():
    """Retrieve the current browser session's temporary dataset."""
    cleanup_active_datasets()

    token = session.get("active_dataset_id")

    if not token:
        return None

    entry = ACTIVE_DATASETS.get(token)

    if entry is None:
        session.pop("active_dataset_id", None)
        session.pop("active_row_count", None)
        return None

    entry["last_access"] = time.time()
    entry["generation"] = ACTIVE_GENERATIONS.get(token)
    entry["row_count"] = session.get("active_row_count", "100")

    return entry


def store_active_dataset(df, filename):
    """Keep a validated dataset in temporary server memory."""
    cleanup_active_datasets()

    token = session.get("active_dataset_id")

    # A stale session token must not overwrite an unrelated entry.
    if token not in ACTIVE_DATASETS:
        token = secrets.token_urlsafe(32)

    ACTIVE_DATASETS[token] = {
        "df": df.copy(),
        "filename": filename,
        "last_access": time.time(),
        "row_count": "100",
        "generation": None,
        "privacy_options": {},
    }

    ACTIVE_GENERATIONS.pop(token, None)
    session["active_dataset_id"] = token
    session["active_row_count"] = "100"

    return ACTIVE_DATASETS[token]


def build_dataset_context(
    df,
    filename,
    row_count="100",
    privacy_options=None,
):
    """Build the analysis results for a dataset."""
    privacy_options = privacy_options or {}
    return {
        "dataset_info": {"filename": filename},
        "profile": profile_dataset(df),
        "pii_findings": detect_pii(df),
        "privacy_report": analyze_privacy_risks(df, **privacy_options),
        "data_quality_report": analyze_data_quality(df),
        "row_count": row_count,
        "has_active_dataset": True,
        "privacy_options": {
            "rare_threshold": privacy_options.get("rare_threshold", 2),
            "uniqueness_threshold": privacy_options.get("uniqueness_threshold", 0.9),
            "quasi_identifiers": privacy_options.get("quasi_identifiers", []),
            "minimum_group_size": privacy_options.get("minimum_group_size", 5),
        },
    }


def parse_privacy_options(form, df):
    """Validate privacy-analysis settings submitted by the dashboard."""
    try:
        rare_threshold = int(form.get("rare_threshold", 2))
        minimum_group_size = int(form.get("minimum_group_size", 5))
        uniqueness_threshold = float(form.get("uniqueness_threshold", 0.9))
    except (TypeError, ValueError) as error:
        raise ValueError("Privacy thresholds must be valid numbers.") from error
    quasi_identifiers = form.getlist("quasi_identifiers")
    if rare_threshold < 1:
        raise ValueError("Rare-category threshold must be at least 1.")
    if minimum_group_size < 1:
        raise ValueError("Minimum group size must be at least 1.")
    if not 0 < uniqueness_threshold <= 1:
        raise ValueError("Uniqueness threshold must be greater than 0 and at most 1.")
    unknown = set(quasi_identifiers) - set(df.columns)
    if unknown:
        raise ValueError("Selected quasi-identifiers must be existing columns.")
    return {
        "rare_threshold": rare_threshold,
        "minimum_group_size": minimum_group_size,
        "uniqueness_threshold": uniqueness_threshold,
        "quasi_identifiers": quasi_identifiers,
    }


def parse_generation_options(form, df):
    """Parse generation settings and explicitly validate every constraint."""
    try:
        row_count = int(form.get("row_count", "100"))
        seed = int(form.get("random_seed", "42"))
    except (TypeError, ValueError) as error:
        raise ValueError("Row count and random seed must be valid integers.") from error
    if not 1 <= row_count <= MAX_ROWS:
        raise ValueError(f"Choose between 1 and {MAX_ROWS} synthetic rows.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("Random seed must be between 0 and 4294967295.")

    ranges = {}
    categories = {}
    for index, column in enumerate(df.columns):
        minimum = form.get(f"range_min_{index}", "").strip()
        maximum = form.get(f"range_max_{index}", "").strip()
        if minimum or maximum:
            ranges[column] = (minimum or None, maximum or None)
        allowed_text = form.get(f"category_values_{index}", "").strip()
        if allowed_text:
            try:
                requested = json.loads(allowed_text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Category constraint for {column} must be a JSON array of values."
                ) from error
            if not isinstance(requested, list) or not requested or any(
                not isinstance(value, str) for value in requested
            ):
                raise ValueError(
                    f"Category constraint for {column} must be a non-empty JSON array of strings."
                )
            observed = df[column].dropna().tolist()
            resolved = []
            for value in requested:
                matches = [
                    original for original in observed
                    if str(original) == value
                ]
                if not matches:
                    raise ValueError(
                        f"Category constraint for {column} includes a value not present in the source."
                    )
                resolved.append(matches[0])
            categories[column] = resolved

    return {
        "row_count": row_count,
        "random_state": seed,
        "strategy": form.get("strategy", "independent"),
        "missing_strategy": form.get("missing_strategy", "preserve"),
        "numerical_strategy": form.get("numerical_strategy", "normal"),
        "categorical_strategy": form.get("categorical_strategy", "weighted"),
        "stratify_by": form.getlist("stratify_by"),
        "constraints": {"ranges": ranges, "categories": categories},
    }


def build_history_summary(history):
    """Summarize the saved analysis records."""
    return {
        "recent_analyses": len(history),
        "analyses_with_warnings": sum(
            1 for item in history
            if item["privacy_warning_count"] > 0
        ),
        "synthetic_generations": sum(
            1 for item in history
            if item["synthetic_row_count"] is not None
        ),
        "synthetic_rows_generated": sum(
            item["synthetic_row_count"] or 0
            for item in history
        ),
    }


def render_dashboard(page="overview", **overrides):
    """Render one page using the shared application context."""
    history = get_history(limit=20)

    context = {
        "page": page,
        "dataset_info": None,
        "profile": None,
        "pii_findings": [],
        "privacy_report": None,
        "data_quality_report": None,
        "synthetic_columns": [],
        "synthetic_preview": [],
        "synthetic_row_count": None,
        "download_token": None,
        "utility_report": None,
        "utility_warning": None,
        "privacy_evaluation": None,
        "privacy_utility_report": None,
        "generation_config": None,
        "generation_form_constraints": None,
        "privacy_options": {
            "rare_threshold": 2,
            "uniqueness_threshold": 0.9,
            "quasi_identifiers": [],
            "minimum_group_size": 5,
        },
        "report_download_token": None,
        "row_count": "100",
        "has_active_dataset": False,
        "error": None,
        "history": history,
        "history_summary": build_history_summary(history),
        "history_message": None,
    }

    context.update(overrides)
    return render_template("index.html", **context)


def get_history_message():
    """Read the optional history-cleared message from the URL."""
    cleared_count = request.args.get("history_cleared")

    if cleared_count is None:
        return None

    try:
        count = int(cleared_count)
    except ValueError:
        return None

    return f"Analysis history cleared. {count} record(s) removed."


def render_current_dataset(page="overview", **overrides):
    """Render a page with the current dataset, when one is available."""
    active = get_active_dataset()
    context = {}

    if active is not None:
        context.update(
            build_dataset_context(
                active["df"],
                active["filename"],
                active.get("row_count", "100"),
                active.get("privacy_options", {}),
            )
        )

        if active.get("generation"):
            context.update(active["generation"])

    context.update(overrides)

    if "history_message" not in context:
        context["history_message"] = get_history_message()

    return render_dashboard(page=page, **context)


@app.route("/", methods=["GET", "POST"])
def index():
    """Overview page and compatibility endpoint for existing forms."""
    if request.method == "GET":
        return render_current_dataset(page="overview")

    uploaded_file = request.files.get("file")
    action = request.form.get("action", "analyze")
    row_count_input = request.form.get("row_count", "100")

    try:
        has_new_file = bool(uploaded_file and uploaded_file.filename)

        if has_new_file:
            if Path(uploaded_file.filename).suffix.lower() != ".csv":
                return render_current_dataset(
                    page="upload",
                    error="Only CSV files are allowed.",
                )

            filename = secure_filename(uploaded_file.filename)

            if not filename:
                return render_current_dataset(
                    page="upload",
                    error="Invalid filename.",
                )

            df = pd.read_csv(
                uploaded_file.stream,
                nrows=MAX_ROWS + 1,
            )

            if len(df) > MAX_ROWS:
                return render_current_dataset(
                    page="upload",
                    error=f"CSV files cannot exceed {MAX_ROWS} rows.",
                )

            if len(df.columns) == 0:
                return render_current_dataset(
                    page="upload",
                    error="The CSV must contain columns.",
                )

            if len(df.columns) > MAX_COLUMNS:
                return render_current_dataset(
                    page="upload",
                    error=f"CSV files cannot exceed {MAX_COLUMNS} columns.",
                )

            # Replace the current dataset only after validation succeeds.
            active = store_active_dataset(df, filename)

        else:
            active = get_active_dataset()

            if active is None:
                return render_dashboard(
                    page="upload",
                    error=(
                        "Please select a CSV file. Upload it again "
                        "to start a new analysis workflow."
                    ),
                )

            df = active["df"]
            filename = active["filename"]

        df = active["df"]
        token = session["active_dataset_id"]

        if action == "privacy":
            try:
                privacy_options = parse_privacy_options(request.form, df)
            except ValueError as error:
                return render_current_dataset(page="upload", error=str(error))
            active["privacy_options"] = privacy_options
            context = build_dataset_context(
                df, filename, row_count_input, privacy_options
            )
            if active.get("generation"):
                context.update(active["generation"])
            return render_dashboard(page="upload", **context)

        privacy_options = active.get("privacy_options", {})
        if action == "generate":
            try:
                generation_config = parse_generation_options(request.form, df)
                privacy_options = parse_privacy_options(request.form, df)
                requested_rows = generation_config.pop("row_count")
                seed = generation_config.pop("random_state")
                synthetic_df = generate_synthetic_data(
                    df,
                    row_count=requested_rows,
                    random_state=seed,
                    **generation_config,
                )
            except (ValueError, TypeError) as error:
                context = build_dataset_context(
                    df, filename, row_count_input, privacy_options
                )
                return render_dashboard(
                    page="synthetic", **context, error=str(error)
                )

            context = build_dataset_context(
                df,
                filename,
                str(requested_rows),
                privacy_options,
            )

            if synthetic_df.empty or len(synthetic_df.columns) == 0:
                return render_dashboard(
                    page="synthetic",
                    **context,
                    error=(
                        "No eligible columns remain after excluding "
                        "direct identifiers. No synthetic file was created."
                    ),
                )

            utility_report = compare_utility(df, synthetic_df)
            requested_qis = privacy_options["quasi_identifiers"]
            disclosure_qis = [
                column for column in requested_qis
                if column in synthetic_df.columns
            ]
            sensitive_attributes = request.form.getlist("sensitive_attributes")
            if sensitive_attributes and not requested_qis:
                return render_dashboard(
                    page="synthetic",
                    **context,
                    error="Select at least one quasi-identifier for attribute-disclosure analysis.",
                )
            if set(sensitive_attributes) - set(df.columns):
                return render_dashboard(
                    page="synthetic",
                    **context,
                    error="Selected sensitive attributes must be existing columns.",
                )
            if set(sensitive_attributes) & set(requested_qis):
                return render_dashboard(
                    page="synthetic",
                    **context,
                    error="Quasi-identifiers and sensitive attributes must be different columns.",
                )
            if sensitive_attributes and len(disclosure_qis) == len(requested_qis):
                privacy_evaluation = evaluate_privacy(
                    df,
                    synthetic_df,
                    quasi_identifiers=disclosure_qis,
                    sensitive_attributes=sensitive_attributes,
                    rare_threshold=privacy_options["rare_threshold"],
                )
            else:
                privacy_evaluation = evaluate_privacy(
                    df,
                    synthetic_df,
                    rare_threshold=privacy_options["rare_threshold"],
                )
                if sensitive_attributes:
                    privacy_evaluation["limitations"].append(
                        "Attribute-disclosure analysis was not run because selected quasi-identifiers "
                        "were excluded from generated columns."
                    )
            privacy_evaluation["generation_configuration"] = {
                "strategy": generation_config["strategy"],
                "missing_strategy": generation_config["missing_strategy"],
                "numerical_strategy": generation_config["numerical_strategy"],
                "categorical_strategy": generation_config["categorical_strategy"],
                "per_column_strategies_applied": generation_config["strategy"] == "independent",
                "row_count": requested_rows,
                "random_seed": seed,
                "range_constraint_count": len(generation_config["constraints"]["ranges"]),
                "categorical_constraint_count": len(generation_config["constraints"]["categories"]),
                "range_constrained_columns": [
                    str(column) for column in generation_config["constraints"]["ranges"]
                ],
                "categorical_constrained_columns": [
                    str(column) for column in generation_config["constraints"]["categories"]
                ],
                "quasi_identifier_columns": [str(column) for column in requested_qis],
                "sensitive_attribute_columns": [str(column) for column in sensitive_attributes],
                "privacy_thresholds": {
                    "uniqueness_ratio": privacy_options["uniqueness_threshold"],
                    "rare_value_count": privacy_options["rare_threshold"],
                    "minimum_group_size": privacy_options["minimum_group_size"],
                },
                "stratification_columns": [
                    str(column) for column in generation_config["stratify_by"]
                ],
            }

            utility_warning = None
            if len(df) < 20:
                utility_warning = (
                    "The original dataset contains fewer than 20 rows. "
                    "Utility comparisons may be unstable and should be "
                    "interpreted cautiously."
                )

            while len(SYNTHETIC_EXPORTS) >= MAX_CACHED_EXPORTS:
                oldest_token = next(iter(SYNTHETIC_EXPORTS))
                del SYNTHETIC_EXPORTS[oldest_token]
                SYNTHETIC_EXPORT_EXPIRY.pop(oldest_token, None)

            export_token = secrets.token_urlsafe(24)
            SYNTHETIC_EXPORTS[export_token] = safe_csv_bytes(synthetic_df)
            SYNTHETIC_EXPORT_EXPIRY[export_token] = time.time() + EXPORT_TTL_SECONDS
            while len(REPORT_EXPORTS) >= MAX_CACHED_EXPORTS:
                oldest_token = next(iter(REPORT_EXPORTS))
                REPORT_EXPORTS.pop(oldest_token)
                REPORT_EXPORT_EXPIRY.pop(oldest_token, None)
            report_token = secrets.token_urlsafe(24)
            report_rows = [{
                "category": "Generation configuration",
                "experiment": "Applied settings",
                "result": json.dumps(
                    privacy_evaluation["generation_configuration"],
                    ensure_ascii=False,
                ),
                "interpretation": "Configuration metadata for interpreting the following privacy experiments.",
                "severity": "not rated",
                "limitations": "Settings describe the generator; they are not privacy guarantees.",
                "recommended_mitigation": "Review strategy-specific record-reuse and relationship-preservation tradeoffs.",
            }] + [
                {
                    "category": item["category"],
                    "experiment": item["name"],
                    "result": json.dumps(item["result"], ensure_ascii=False),
                    "interpretation": item["interpretation"],
                    "severity": item["severity"] or "not rated",
                    "limitations": json.dumps(item["limitations"], ensure_ascii=False),
                    "recommended_mitigation": item["recommended_mitigation"],
                }
                for item in privacy_evaluation["experiments"]
            ]
            REPORT_EXPORTS[report_token] = safe_csv_bytes(pd.DataFrame(report_rows))
            REPORT_EXPORT_EXPIRY[report_token] = time.time() + EXPORT_TTL_SECONDS
            sanitized_generation_config = {
                "strategy": generation_config["strategy"],
                "strategy_description": describe_generation(generation_config["strategy"]),
                "missing_strategy": generation_config["missing_strategy"],
                "numerical_strategy": generation_config["numerical_strategy"],
                "categorical_strategy": generation_config["categorical_strategy"],
                "per_column_strategies_applied": generation_config["strategy"] == "independent",
                "random_seed": seed,
                "range_constraint_count": len(generation_config["constraints"]["ranges"]),
                "categorical_constraint_count": len(generation_config["constraints"]["categories"]),
                "range_constrained_columns": [
                    str(column) for column in generation_config["constraints"]["ranges"]
                ],
                "categorical_constrained_columns": [
                    str(column) for column in generation_config["constraints"]["categories"]
                ],
                "quasi_identifier_columns": [str(column) for column in requested_qis],
                "sensitive_attribute_columns": [str(column) for column in sensitive_attributes],
                "privacy_thresholds": {
                    "uniqueness_ratio": privacy_options["uniqueness_threshold"],
                    "rare_value_count": privacy_options["rare_threshold"],
                    "minimum_group_size": privacy_options["minimum_group_size"],
                },
                "stratification_columns": [
                    str(column) for column in generation_config["stratify_by"]
                ],
            }
            generation_form_constraints = {
                "ranges": {
                    str(column): [
                        "" if lower is None else str(lower),
                        "" if upper is None else str(upper),
                    ]
                    for column, (lower, upper) in generation_config["constraints"]["ranges"].items()
                },
                "categories": {
                    str(column): json.dumps(values, ensure_ascii=False)
                    for column, values in generation_config["constraints"]["categories"].items()
                },
            }
            recommendations = list(privacy_evaluation["recommendations"])
            overlap_experiment = next(
                (
                    item for item in privacy_evaluation["experiments"]
                    if item["category"] == "Record overlap"
                ),
                None,
            )
            overlap = (
                overlap_experiment["result"]["overlapping_synthetic_rows"]
                if overlap_experiment else 0
            )
            if overlap:
                recommendations.append(
                    "Exact source-row overlap was observed; avoid joint or stratified row resampling if record reproduction is unacceptable."
                )
            if context["pii_findings"]:
                recommendations.append(
                    "Review potential direct identifiers and remove or transform them before any disclosure; name-based detection can miss identifiers."
                )
            if context["privacy_report"]["quasi_identifier_analysis"]["records_in_small_groups"]:
                recommendations.append(
                    "Review, coarsen, or suppress selected quasi-identifiers that create groups below the configured minimum size."
                )
            if context["data_quality_report"]["findings"]:
                recommendations.append(
                    "Resolve or explicitly account for the reported quality issues before relying on utility comparisons."
                )
            privacy_utility_report = {
                "data_quality_findings": context["data_quality_report"]["findings"],
                "potential_pii_findings": context["pii_findings"],
                "privacy_risks": context["privacy_report"],
                "utility_metrics": utility_report,
                "generation_settings": sanitized_generation_config,
                "privacy_evaluation": privacy_evaluation,
                "warnings": (
                    context["privacy_report"]["warnings"]
                    + [
                        {
                            "type": "Evaluation limitation",
                            "severity": "info",
                            "columns": [],
                            "message": limitation,
                        }
                        for limitation in privacy_evaluation["limitations"]
                    ]
                ),
                "recommendations": recommendations,
            }

            generation_context = {
                "synthetic_columns": list(synthetic_df.columns),
                "synthetic_preview": (
                    synthetic_df.head(10).to_dict(orient="records")
                ),
                "synthetic_row_count": len(synthetic_df),
                "download_token": export_token,
                "utility_report": utility_report,
                "utility_warning": utility_warning,
                "privacy_evaluation": privacy_evaluation,
                "privacy_utility_report": privacy_utility_report,
                "generation_config": sanitized_generation_config,
                "generation_form_constraints": generation_form_constraints,
                "report_download_token": report_token,
                "privacy_options": privacy_options,
                "row_count": str(requested_rows),
                "has_active_dataset": True,
            }

            ACTIVE_GENERATIONS[token] = generation_context
            active["generation"] = generation_context
            active["privacy_options"] = privacy_options
            active["row_count"] = str(requested_rows)
            session["active_row_count"] = str(requested_rows)
            context.update(generation_context)

        else:
            # Analyzing again clears results from a previous generation.
            ACTIVE_GENERATIONS.pop(token, None)
            active["generation"] = None
            active["row_count"] = row_count_input
            session["active_row_count"] = row_count_input
            context = build_dataset_context(
                df, filename, row_count_input, privacy_options
            )

        save_analysis(
            filename=filename,
            row_count=len(df),
            column_count=len(df.columns),
            privacy_warning_count=len(
                context["privacy_report"].get("warnings", [])
            ),
            pii_finding_count=len(context["pii_findings"]),
            synthetic_row_count=(
                context.get("synthetic_row_count")
                if action == "generate"
                else None
            ),
        )

        page = "synthetic" if action == "generate" else "upload"
        return render_dashboard(page=page, **context)

    except (
        ValueError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ):
        return render_current_dataset(
            page="upload",
            error="Unable to read this file as a valid CSV.",
        )

    except Exception:
        app.logger.exception("CSV processing failed.")
        return render_current_dataset(
            page="upload",
            error="An unexpected error occurred while processing the CSV.",
        )


@app.route("/upload", methods=["GET"])
def upload_page():
    """Upload and dataset analysis page."""
    return render_current_dataset(page="upload")


@app.route("/synthetic", methods=["GET"])
def synthetic_page():
    """Synthetic data generation page."""
    return render_current_dataset(page="synthetic")


@app.route("/compare", methods=["GET"])
def compare_page():
    """Synthetic-versus-original comparison page."""
    return render_current_dataset(page="compare")


@app.route("/history", methods=["GET"])
def history_page():
    """Saved analysis history page."""
    return render_current_dataset(page="history")



@app.route("/clear-history", methods=["POST"])
def clear_history_route():
    """Clear saved history and redirect to the dashboard."""
    deleted_count = clear_history()

    return redirect(
        url_for("index", history_cleared=deleted_count)
    )



@app.route("/download-synthetic/<token>", methods=["GET"])
def download_synthetic(token):
    """Download a generated CSV once."""
    csv_bytes = SYNTHETIC_EXPORTS.pop(token, None)
    expires_at = SYNTHETIC_EXPORT_EXPIRY.pop(token, None)

    if csv_bytes is None or (expires_at is not None and time.time() > expires_at):
        abort(
            404,
            description="This export has expired or was already used.",
        )

    return send_file(
        BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="datashadow_synthetic_data.csv",
    )


@app.route("/download-privacy-report/<token>", methods=["GET"])
def download_privacy_report(token):
    """Download a one-time CSV summary of empirical privacy experiments."""
    csv_bytes = REPORT_EXPORTS.pop(token, None)
    expires_at = REPORT_EXPORT_EXPIRY.pop(token, None)
    if csv_bytes is None or (expires_at is not None and time.time() > expires_at):
        abort(404, description="This report export has expired or was already used.")
    return send_file(
        BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="datashadow_privacy_evaluation.csv",
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_current_dataset(
        page="upload",
        error="File exceeds the 5 MB upload limit.",
    ), 413


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)