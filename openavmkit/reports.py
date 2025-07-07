import importlib.resources
import os
import warnings

import markdown
import pdfkit

from openavmkit.utilities.settings import get_model_group, get_valuation_date


class MarkdownReport:
    """
    A report generator that uses a Markdown template.

    Attributes
    ----------
    name : str
        Name of the report, corresponding to a Markdown template.
    template : str
        The raw Markdown template text.
    rendered : str
        The rendered Markdown text after variable substitution.
    variables : dict
        Dictionary of variables for substitution in the template.
    """

    name: str
    template: str
    rendered: str
    variables: dict

    def __init__(self, name: str) -> None:
        """
        Initialize the MarkdownReport by loading the Markdown template.

        Parameters
        ----------
        name : str
            Name of the report template (without file extension).
        """
        self.name = name
        with importlib.resources.open_text(
            "openavmkit.resources.reports", f"{name}.md", encoding="utf-8"
        ) as file:
            self.template = file.read()
        self.variables = {}
        self.rendered = ""

    def get_var(self, key: str):
        """
        Get the value of a variable.

        Parameters
        ----------
        key : str
            Variable key.

        Returns
        -------
        Any or None
            The value associated with `key`, or `None` if not set.
        """
        return self.variables.get(key)

    def set_var(self, key: str, value, fmt: str = None):
        """
        Set a variable value with optional formatting.

        Parameters
        ----------
        key : str
            Variable key.
        value : Any
            Value to be set.
        fmt : str, optional
            Format string to apply to `value`.

        Returns
        -------
        None
        """
        if value is None:
            formatted_value = "<NULL>"
        elif fmt is not None:
            formatted_value = format(value, fmt)
        else:
            formatted_value = str(value)
        self.variables[key] = formatted_value

    def render(self) -> str:
        """
        Render the report by substituting variables in the template.

        Returns
        -------
        str
            The rendered Markdown text with all variables replaced.
        """
        self.rendered = self.template
        for key, value in self.variables.items():
            self.rendered = self.rendered.replace(f"{{{{{key}}}}}", str(value))
        return self.rendered


def start_report(report_name: str, settings: dict, model_group: str) -> MarkdownReport:
    """
    Create and initialize a MarkdownReport with basic variables set.

    Parameters
    ----------
    report_name : str
        Name of the report template.
    settings : dict
        Settings dictionary.
    model_group : str
        Model group identifier.

    Returns
    -------
    MarkdownReport
        Initialized MarkdownReport object.
    """

    report = MarkdownReport(report_name)
    locality = settings.get("locality", {}).get("name")
    val_date = get_valuation_date(settings)
    val_date = val_date.strftime("%Y-%m-%d")

    model_group_obj = get_model_group(settings, model_group)
    model_group_name = model_group_obj.get("name", model_group)

    report.set_var("locality", locality)
    report.set_var("val_date", val_date)
    report.set_var("model_group", model_group_name)
    return report


def finish_report(
    report: MarkdownReport, outpath: str, css_file: str, settings: dict
) -> None:
    """
    Render the report and export it in Markdown, HTML, and PDF formats.

    Saves the rendered Markdown to disk and converts it to target formats using a
    specified CSS file.

    Parameters
    ----------
    report : MarkdownReport
        MarkdownReport object to be finished.
    outpath : str
        Output file path (without extension).
    css_file : str
        Name of the CSS file (without extension) to style the report.
    settings : dict
        Settings dictionary.
    """
    formats = settings.get("analysis", {}).get("report", {}).get("formats", None)
    if formats is None:
        formats = ["pdf", "md"]

    report_text = report.render()
    os.makedirs(outpath, exist_ok=True)
    with open(f"{outpath}.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    pdf_path = f"{outpath}.pdf"

    _markdown_to_pdf(report_text, pdf_path, formats=formats, css_file=css_file)

    if "md" not in formats:
        os.remove(f"{outpath}.md")


def _markdown_to_pdf(
    md_text: str, out_path: str, formats: list[str], css_file: str = None
) -> None:
    """
    Convert Markdown text to PDF (and optionally other formats) via an HTML intermediate.

    This function first converts the provided Markdown to HTML, saves the HTML to disk,
    and then converts the HTML to PDF and any additional specified formats.

    Parameters
    ----------
    md_text : str
        Markdown text to be converted.
    out_path : str
        Base output file path (without extension) where generated files will be saved.
    formats : list[str]
        List of formats to output (e.g., ["pdf", "md", "html"]).
    css_file : str, optional
        Optional CSS file stub for styling the HTML before conversion.
    """
    html_text = _markdown_to_html(md_text, css_file)
    html_path = out_path.replace(".pdf", ".html")
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_text)

    if "pdf" in formats:
        _html_to_pdf(html_text, out_path)

    if "html" not in formats:
        os.remove(html_path)


def _markdown_to_html(md_text, css_file_stub=None):
    """Convert Markdown text to a complete HTML document using a CSS file."""
    html_text = markdown.markdown(md_text, extensions=["extra"])

    css_path = _get_resource_path() + f"/reports/css/{css_file_stub}.css"

    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            css_text = css_file.read()
    else:
        css_text = ""

    css_base_path = _get_resource_path() + f"/reports/css/base.css"
    with open(css_base_path, "r", encoding="utf-8") as css_file:
        css_base_text = css_file.read()

    css_text = css_base_text + css_text
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
        {css_text}
        </style>
    </head>
    <body>
        {html_text}
    </body>
    </html>
    """
    return html_template


def _html_to_pdf(html_text, out_path):
    """Convert an HTML string to a PDF file using pdfkit."""
    try:
        pdfkit.from_string(html_text, out_path, options={"quiet": False})
    except OSError:
        warnings.warn(
            "Failed to generate PDF report. Is `wkhtmltopdf` installed? See the README for details."
        )


def _get_resource_path():
    """Get the absolute path to the resources directory."""
    this_files_path = os.path.abspath(__file__)
    this_files_dir = os.path.dirname(this_files_path)
    resources_path = os.path.join(this_files_dir, "resources")
    return resources_path
