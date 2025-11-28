from PIL import Image, ImageDraw
from PyPDF2 import PdfReader, PdfWriter
import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="merged_output.pdf", help="Output file path")
parser.add_argument("-i", "--input", default="./**", help="Input file path")
args = parser.parse_args()

def create_image_grid(images, title, title_size = 40, page_size=(900, 600)):  # A4 size in points
    """
    Arranges up to 4 images on a single PDF page in a 2x2 grid.
    Each image will be resized to fit a quarter of the page.
    """
    grid_image = Image.new("RGB", (page_size[0], page_size[1] + 2*title_size), "white")  # Background for the page
    draw = ImageDraw.Draw(grid_image)
    
    # Resize each image and place them on the grid
    draw.text((page_size[0] / 2, title_size), title, fill="black", anchor="mm", align="center")
    positions = [(0, 2*title_size), (page_size[0] // 2, 2*title_size), (0, page_size[1] // 2 + 2*title_size), (page_size[0] // 2, page_size[1] // 2 + 2*title_size)]
    cell_size = (page_size[0] // 2, page_size[1] // 2)
    
    for i, img in enumerate(images):
        img = img.convert("RGB")
        img = img.resize(cell_size, Image.ANTIALIAS)
        grid_image.paste(img, positions[i])
    
    return grid_image

def merge_images_pdfs(output_path, input_path):
    pdf_writer = PdfWriter()
    
    all_paths = glob.glob(input_path, recursive=True)
    input_files = []
    for file in all_paths:
        if ".png" in file or ".pdf" in file:
            input_files.append(file)
    
    plot_selection_names = [["p", r'tan(lambda)', "norm. multiplicity", "NCl [#]"], ["fTPCInnerParam", "fTgl", "fNormMultTPC", "fNormNClustersTPC"]]
    output_plots = [[],[]]
    for i, n in enumerate(plot_selection_names[1]):
        append_axis = [" [NN]", " [BB]"]
        for j, p in enumerate(["NetworkRatioNSigmaBins_", "BBRatioNSigmaBins_"]):
            plot = np.array(input_files)[[(("/" + n + "/" in file) and (p in file)) for file in input_files]]
            output_plots[0].append(plot_selection_names[0][i] + append_axis[j])
            output_plots[1].append(plot)
    
    sort_particles = ["Electrons", "Muons", "Pions", "Kaons", "Protons", "Deuteron", "Triton", "Helium3"]
    for category, plots in output_plots:
        category_plots = sorted(plots)
        
        images = []
        for plot in category_plots:
            if plot.endswith(".png"):
                image = Image.open(plot)
                images.append(image)
            elif plot.endswith(".pdf"):
                pdf_reader = PdfReader(plot)
                for page_num in range(len(pdf_reader.pages)):
                    pdf_page = pdf_reader.pages[page_num]
                    pdf_writer.add_page(pdf_page)
        
        if images:
            # Create image grid
            grid_image = create_image_grid(images, category)
            
            # Convert grid image to PDF page
            grid_image_pdf_path = "grid_page_{}.pdf".format(category)
            grid_image.save(grid_image_pdf_path, "PDF")
            pdf_reader = PdfReader(grid_image_pdf_path)
            pdf_writer.add_page(pdf_reader.pages[0])
            os.remove(grid_image_pdf_path)  # Clean up the temporary PDF file
    
    with open(output_path, "wb") as out_file:
        pdf_writer.write(out_file)

merge_images_pdfs(args.output, args.input)
