import importlib.resources
import os

import markdown
import pdfkit


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


def _markdown_to_html(md_text, css_file_stub=None):
  # First, convert the markdown to HTML
  html_text = markdown.markdown(md_text, extensions=["extra"])

  css_path = _get_resource_path() + f"/reports/css/{css_file_stub}.css"
  with open(css_path, "r", encoding="utf-8") as css_file:
    css_text = css_file.read()

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