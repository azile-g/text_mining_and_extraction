#DEPENDENCIES
#The usual suspects
import pandas as pd
import glob
import re 
import os
import matplotlib.pyplot as plt
import time
import itertools
import os
#image handling 
import cv2 as cv
from PIL import Image
#pdf handling
import fitz
from pdfminer.layout import LAParams, LTTextBox, LTImage, LTFigure, LTLine, LTRect
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator

#GET PATHS AND ITER
#local path handling
def getiter_paths(doc_file): 
    reader_path = re.sub(r"\\", "/", str(os.getcwd())) + r"\\" + doc_file + r"\\"
    iter = glob.iglob(reader_path + "*.pdf")
    lst = glob.glob(reader_path + "*.pdf")
    name_lst = [os.path.basename(i) for i in lst]
    return lst, iter, name_lst

#METADATA COMPONENTS
def get_metadata(lst, name_lst): 
    documents = [fitz.open(i) for i in lst]
    pg_count = [d.page_count for d in documents]
    toc = [d.get_toc(simple = False) for d in documents]
    metadata = [d.metadata for d in documents]
    created_lst = [metadata[i]["creationDate"] for i in range(len(documents))]
    mod_lst = [metadata[i]["modDate"] for i in range(len(documents))]
    sys_lst = [metadata[i]["creator"] for i in range(len(documents))]
    format_lst = [metadata[i]["format"] for i in range(len(documents))]
    metadata_df = pd.DataFrame(data = {"doc_name": name_lst, "page_count": pg_count, "toc": toc, "created_dt": created_lst, "mod_dt": mod_lst, "system": sys_lst, "PDF_format": format_lst})
    return metadata_df

#WIP - to clean and parse toc data 
def get_toc(lst): 
        documents = [fitz.open(i) for i in lst]
        toc = [d.get_toc(simple = False) for d in documents]
        return toc

def get_links(path, aggre = False): 
    doc = fitz.open(path)
    pages = [doc[i]for i in range(len(doc))]
    raw_links = [i.get_links() for i in pages]
    page_dict = {f"page_{i}": {"kind": [], "destination": [], "origin": []} for i in range(len(doc))}
    for i in range(len(raw_links)): 
        page_links = raw_links[i]
        kind_lst = [page_links[j]["kind"] for j in range(len(page_links))]
        origin_lst = [page_links[j]["from"] for j in range(len(page_links))]
        page_dict[f"page_{i}"]["kind"] = kind_lst
        page_dict[f"page_{i}"]["origin"] = origin_lst
        dest_lst = []
        for index, k in enumerate(kind_lst): 
            if k == 0: 
                dest_lst.append("dummy_link")
            elif k == 1: 
                dest_lst.append({"internal_ref": page_links[index]["to"]})
            elif k == 2: 
                dest_lst.append(page_links[index]["uri"])
            elif k == 3: 
                dest_lst.append("executable")
            elif k == 4: 
                dest_lst.append("named_loc")
            elif k == 5: 
                dest_lst.append("external_PDF")
            else: 
                raise ValueError("k has to be between 0 and 5, FON Liz Cheng for help or look at PyMuDF documentation")
        page_dict[f"page_{i}"]["destination"] = dest_lst
        for page in page_dict.keys(): 
            page_dict[page] = pd.DataFrame(page_dict[page])
            page_dict[page]["p_i"] = page
    if aggre == False: 
        return page_dict
    elif aggre == True: 
        link_df = page_dict["page_0"]
        for i in range(1, len(page_dict.keys())): 
            link_df = pd.concat([link_df, page_dict[f"page_{i}"]])
        link_df = link_df.reset_index()
        return link_df
    else: 
        raise ValueError("Hello excuse me sir aggre is bool")

def get_fonts(path, aggre = False): 
    doc = fitz.open(path)
    pages = [doc[i]for i in range(len(doc))]
    raw_fonts = [i.get_fonts(full = True) for i in pages]
    page_dict = {f"page_{i}": {"font_name": [], "font_reference": [], "type": [], "encoder": []} for i in range(len(doc))}
    for i in range(len(raw_fonts)): 
        page_fonts = raw_fonts[i]
        name_lst = [page_fonts[j][3] for j in range(len(page_fonts)) if page_fonts[j][6] == 0]
        font_ref = [page_fonts[j][4] for j in range(len(page_fonts)) if page_fonts[j][6] == 0]
        font_type = [page_fonts[j][2] for j in range(len(page_fonts)) if page_fonts[j][6] == 0]
        font_encoding = [page_fonts[j][5] for j in range(len(page_fonts)) if page_fonts[j][6] == 0]
        page_dict[f"page_{i}"]["font_name"] = name_lst
        page_dict[f"page_{i}"]["font_reference"] = font_ref
        page_dict[f"page_{i}"]["type"] = font_type
        page_dict[f"page_{i}"]["encoder"] = font_encoding
        for page in page_dict.keys(): 
            page_dict[page] = pd.DataFrame(page_dict[page])
            page_dict[page]["p_i"] = page
    if aggre == False: 
        return page_dict
    elif aggre == True: 
        font_df = page_dict["page_0"]
        for i in range(1, len(page_dict.keys())): 
            font_df = pd.concat([font_df, page_dict[f"page_{i}"]])
        font_df = font_df.reset_index()
        return font_df
    else: 
        raise ValueError("Hello excuse me sir aggre is bool")

#IMAGE DATA COMPONENT
def get_img_mdata(path, aggre = False): 
    doc = fitz.open(path)
    pages = [doc[i]for i in range(len(doc))]
    raw_img_refs = [i.get_image_info(hashes = True, xrefs = True) for i in pages]
    #unique_img_refs = [i for i in raw_img_refs if digestval isin digestlst]
    page_dict = {f"page_{i}": {"img_ref": [], "coords": [], "width": [], "height": [], "cs_name": [], "xres": [], "yres": [], "bpc": [], "storage": []} for i in range(len(doc))}
    for i in range(len(raw_img_refs)): 
        page_img_refs = raw_img_refs[i]
        imgref_lst = [page_img_refs[j]["xref"] for j in range(len(page_img_refs))]
        coords_lst = [page_img_refs[j]["bbox"] for j in range(len(page_img_refs))]
        width_lst = [page_img_refs[j]["width"] for j in range(len(page_img_refs))]
        height_lst = [page_img_refs[j]["height"] for j in range(len(page_img_refs))]
        csname_lst = [page_img_refs[j]["cs-name"] for j in range(len(page_img_refs))]
        xres_lst = [page_img_refs[j]["xres"] for j in range(len(page_img_refs))]
        yres_lst = [page_img_refs[j]["yres"] for j in range(len(page_img_refs))]
        bpc_lst = [page_img_refs[j]["bpc"] for j in range(len(page_img_refs))]
        storage_lst = [page_img_refs[j]["size"] for j in range(len(page_img_refs))]
        page_dict[f"page_{i}"]["img_ref"] = imgref_lst
        page_dict[f"page_{i}"]["coords"] = coords_lst
        page_dict[f"page_{i}"]["width"] = width_lst
        page_dict[f"page_{i}"]["height"] = height_lst
        page_dict[f"page_{i}"]["cs_name"] = csname_lst
        page_dict[f"page_{i}"]["xres"] = xres_lst
        page_dict[f"page_{i}"]["yres"] = yres_lst
        page_dict[f"page_{i}"]["bpc"] = bpc_lst
        page_dict[f"page_{i}"]["storage"] = storage_lst
        for page in page_dict.keys(): 
            page_dict[page] = pd.DataFrame(page_dict[page])
            page_dict[page]["p_i"] = page
    if aggre == False: 
        return page_dict
    elif aggre == True: 
        mimg_df = page_dict["page_0"]
        for i in range(1, len(page_dict.keys())): 
            mimg_df = pd.concat([mimg_df, page_dict[f"page_{i}"]])
        mimg_df = mimg_df.reset_index()
        return mimg_df, raw_img_refs
    else: 
        raise ValueError("Hello excuse me sir aggre is bool")

#TEXT DATA COMPONENT
def get_raw_text(filepath, aggre = False): 
    doc = fitz.open(filepath)
    pages = [doc[i] for i in range(len(doc))]
    page_dict = {f"page_{i}": ialue.get_text().split("\n") for i, ialue in enumerate(pages)}
    for j in page_dict.keys(): 
        lst = " ".join(page_dict[j])
        page_dict[j] = lst
    return page_dict

def parse_doc_text(filepath, linemargin = 0.35): 
    docfile = open(filepath, "rb")
    rsrcmgr = PDFResourceManager()
    device = PDFPageAggregator(rsrcmgr, laparams=LAParams(line_margin = linemargin))
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(docfile)
    doc = {}
    i = 0
    for page in pages:
        interpreter.process_page(page)
        layout = device.get_result()
        x0_lst = []
        x1_lst = []
        y0_lst = []
        y1_lst = []
        txt_lst = []
        for lobj in layout:
            if isinstance(lobj, LTTextBox):
                x0, x1, y0, y1, text = lobj.bbox[0], lobj.bbox[1], lobj.bbox[2], lobj.bbox[3], lobj.get_text()
                x0_lst.append(x0)
                x1_lst.append(x1)
                y0_lst.append(y0)
                y1_lst.append(y1)
                txt_lst.append(text)
        doc[f"page_{i}"] = pd.DataFrame({"text": txt_lst, "x0": x0_lst, "x1": x1_lst, "y0": y0_lst, "y1": y1_lst})
        doc[f"page_{i}"]["tl_coords"] = list(zip(doc[f"page_{i}"]["x0"].to_list(), doc[f"page_{i}"]["y1"].to_list()))
        doc[f"page_{i}"]["br_coords"] = list(zip(doc[f"page_{i}"]["y0"].to_list(), doc[f"page_{i}"]["x1"].to_list()))
        doc[f"page_{i}"]["bl_coords"] = list(zip(doc[f"page_{i}"]["x0"].to_list(), doc[f"page_{i}"]["x1"].to_list()))
        doc[f"page_{i}"]["tr_coords"] = list(zip(doc[f"page_{i}"]["y0"].to_list(), doc[f"page_{i}"]["y1"].to_list()))
        #doc[f"page_{i}"] = doc[f"page_{i}"].sort_values(by = ["y1"], ascending = False) 
        doc[f"page_{i}"]["text"] = list(map(str.strip, doc[f"page_{i}"]["text"].to_list()))
        i = i+1
    return doc

def coord_plot(dataframe): 
    tl = dataframe["tl_coords"].to_list()
    tlx, tly = list(zip(*tl))
    br = dataframe["br_coords"].to_list()
    brx, bry = list(zip(*br))
    bl = dataframe["bl_coords"].to_list()
    blx, bly = list(zip(*bl))
    tr = dataframe["tr_coords"].to_list()
    trx, tr_y = list(zip(*tr))
    plt.rcParams["figure.figsize"] = [10, 18]
    plt.scatter(tlx, tly)
    plt.scatter(brx, bry)
    plt.scatter(blx, bly)
    plt.scatter(trx, tr_y)
    return

def parse_docs(iter, linemargin = 0.35): 
    #multiprocessing to process the directory of pdfs
    docfile = open(iter, "rb")
    rsrcmgr = PDFResourceManager()
    device = PDFPageAggregator(rsrcmgr, laparams=LAParams(line_margin = linemargin))
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(docfile)
    doc = {}
    i = 1
    for page in pages:
        interpreter.process_page(page)
        layout = device.get_result()
        x0_lst = []
        x1_lst = []
        y0_lst = []
        y1_lst = []
        txt_lst = []
        for lobj in layout:
            if isinstance(lobj, LTTextBox):
                x0, x1, y0, y1, text = lobj.bbox[0], lobj.bbox[1], lobj.bbox[2], lobj.bbox[3], lobj.get_text()
                x0_lst.append(x0)
                x1_lst.append(x1)
                y0_lst.append(y0)
                y1_lst.append(y1)
                txt_lst.append(text)
        doc[f"page_{i}"] = {"text": txt_lst, "x0": x0_lst, "x1": x1_lst, "y0": y0_lst, "y1": y1_lst}
        i = i+1
    return doc

def test(one):
    if one == "test":
        print("hello world")