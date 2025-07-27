from pdf_utils import extract_outline_and_text
pdf_path = "example.pdf"
print("Extracting outline and text from:", pdf_path)
res = extract_outline_and_text(pdf_path)
print(res)