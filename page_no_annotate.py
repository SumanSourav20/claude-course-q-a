import argparse
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.colors import red
from reportlab.lib.pagesizes import letter
import io
import os

def add_page_numbers(input_pdf, output_pdf, font_size=10, x_offset=30, y_offset=30):
    """
    Add page numbers to the top left corner of each page in a PDF file.
    
    Args:
        input_pdf (str): Path to the input PDF file
        output_pdf (str): Path to save the output PDF file
        font_size (int, optional): Size of the page number font. Defaults to 10.
        x_offset (int, optional): Horizontal position from left edge in points. Defaults to 30.
        y_offset (int, optional): Distance from top edge in points. Defaults to 30.
    """
    print(f"Adding page numbers to {input_pdf}...")
    
    # Read the input PDF
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    
    # Process each page
    for page_num, page in enumerate(reader.pages):
        # Create a canvas to draw the page number on
        packet = io.BytesIO()
        # Use the actual page size rather than assuming letter
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)
        c = canvas.Canvas(packet, pagesize=(page_width, page_height))
        
        # Set the text properties
        c.setFont("Helvetica-Bold", font_size)
        c.setFillColor(red)
        
        # Calculate the position from the top left (note that PDF coordinates start from bottom left)
        text = f"page {page_num + 1}"
        c.drawString(x_offset, page_height - y_offset, text)
        
        c.save()
        
        # Move to the beginning of the BytesIO buffer
        packet.seek(0)
        
        # Create a new PDF with the page number
        overlay = PdfReader(packet)
        
        # Merge the original page with the page number overlay
        page.merge_page(overlay.pages[0])
        
        # Add the annotated page to the output PDF
        writer.add_page(page)
    
    # Write the output PDF
    with open(output_pdf, "wb") as output_file:
        writer.write(output_file)
    
    print(f"Successfully created annotated PDF: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Add page numbers to a PDF file")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument(
        "-o", "--output", 
        help="Path to save the output PDF file (default: input_pdf_numbered.pdf)"
    )
    parser.add_argument(
        "-s", "--size", type=int, default=18,
        help="Font size for page numbers (default: 10)"
    )
    parser.add_argument(
        "-x", type=int, default=30,
        help="Horizontal position from left edge in points (default: 30)"
    )
    parser.add_argument(
        "-y", type=int, default=30,
        help="Distance from top edge in points (default: 30)"
    )
    
    args = parser.parse_args()
    
    # If output file is not specified, create a default name
    if not args.output:
        base, ext = os.path.splitext(args.input_pdf)
        args.output = f"{base}_numbered{ext}"
    
    add_page_numbers(args.input_pdf, args.output, args.size, args.x, args.y)

if __name__ == "__main__":
    main()