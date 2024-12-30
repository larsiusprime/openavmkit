import pandas as pd
from xlsxwriter import Workbook
from xlsxwriter.format import Format
from xlsxwriter.worksheet import Worksheet


def write_to_excel(df: pd.DataFrame, path: str, metadata: dict):
  metadata = metadata.copy()
  with pd.ExcelWriter(path, engine='xlsxwriter') as writer:

    sheet_name = metadata.get("sheet_name", "Sheet1")

    # Write DataFrame to Excel
    df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Grab the workbook and worksheet objects
    workbook: Workbook = writer.book

    format_header: Format = workbook.add_format({
      "text_wrap": True,
      "bold": True
    })

    columns = metadata.get("columns", {}).get("formats", {})
    conditions = metadata.get("columns", {}).get("conditions", {})

    new_columns = {}
    for column in columns:
      format_obj = columns[column]
      wkb_format: Format = workbook.add_format(format_obj)
      new_columns[column] = wkb_format
    columns = new_columns

    # check if there is already a worksheet with the name 'name':
    if sheet_name not in writer.sheets:
      workbook.add_worksheet(sheet_name)

    # get the current worksheet:
    worksheet: Worksheet = writer.sheets[sheet_name]

    for col_num, value in enumerate(df.columns.values):
      worksheet.write(0, col_num, value, format_header)

    for column in columns:
      if column not in df:
        continue
      column_format = columns.get(column)
      col_idx = df.columns.get_loc(column)
      worksheet.set_column(col_idx, col_idx, None, column_format)

    for column in conditions:
      if column not in df:
        continue
      condition = conditions[column]
      condition["format"] = workbook.add_format(condition["format"])
      col_idx = df.columns.get_loc(column)
      worksheet.conditional_format(1, col_idx, len(df), col_idx, condition)


def _clean_formats(formats: dict):
  replaces = {
    "$": "_dollar_",
    "%": "_percent_"
  }
  new_keys = {}
  for key in formats:
    new_key = key
    for rep in replaces:
      if rep in key:
        new_key = key.replace(rep, replaces[rep])
    new_keys[new_key] = formats[key]
  return new_keys
