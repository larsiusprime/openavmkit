import importlib.resources
import os

import markdown
import pdfkit

from openavmkit.utilities.settings import get_model_group, get_valuation_date


class MarkdownReport:
  name: str
  template: str
  rendered: str
  variables: dict

  def __init__(self, name):
    self.name = name
    with importlib.resources.open_text("openavmkit.resources.reports", f"{name}.md", encoding="utf-8") as file:
      self.template = file.read()
    self.variables = {}
    self.rendered = ""


  def get_var(self, key: str):
    return self.variables.get(key)


  def set_var(self, key: str, value, fmt: str = None):
    if value is None:
      formatted_value = "<NULL>"
    elif fmt is not None:
      formatted_value = format(value, fmt)
    else:
      formatted_value = str(value)
    self.variables[key] = formatted_value

  def render(self):
    self.rendered = self.template
    for key in self.variables:
      value = self.variables.get(key)
      self.rendered = self.rendered.replace("{{{" + key + "}}}", str(value))
    return self.rendered


def markdown_to_pdf(md_text, out_path, css_file=None):
  html_text = _markdown_to_html(md_text, css_file)
  html_path = out_path.replace(".pdf", ".html")
  with open(html_path, "w", encoding="utf-8") as html_file:
    html_file.write(html_text)
  _html_to_pdf(html_text, out_path)


def start_report(report_name: str, settings: dict, model_group: str):
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

def finish_report(report: MarkdownReport, outpath: str, css_file: str):
  report_text = report.render()
  with open(f"{outpath}.md", "w", encoding="utf-8") as f:
    f.write(report_text)
  pdf_path = f"{outpath}.pdf"
  markdown_to_pdf(report_text, pdf_path, css_file=css_file)


def _markdown_to_html(md_text, css_file_stub=None):
  # First, convert the markdown to HTML
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
  pdfkit.from_string(html_text, out_path, options={"quiet": False})


def _get_resource_path():
  this_files_path = os.path.abspath(__file__)
  this_files_dir = os.path.dirname(this_files_path)
  resources_path = os.path.join(this_files_dir, "resources")
  return resources_path

