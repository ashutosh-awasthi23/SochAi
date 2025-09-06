import fitz   #PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    return text



if __name__ == "__main__":
    pdf_file = r"C:\Users\ashut\Downloads\Social_Studies_IV_Shreya_Pandey.pdf"
    extracted_text = extract_text_from_pdf(pdf_file)
    print ("======= Extracted Text Preview =========")
    print(extracted_text[:1000])