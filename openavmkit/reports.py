import importlib.resources

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
      self.rendered = self.rendered.replace("{{{"+key+"}}}", str(value))
    return self.rendered