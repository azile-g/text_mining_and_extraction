#the usual suspects
import pandas as pd
import fitz

#def get_pages(filepaths): 
    #documents = [fitz.Document(doc) for doc in ]
    
#find table based on known bounds
def find_table(page, title_str, bot_str): 
    table_name = str(title_str)
    top_bound = page.search_for(table_name)
    if not top_bound: 
        raise ValueError("Table title not found. Please readjust widget accordingly. All else fails, FON Liz Cheng, thanks.")
    top_rect = top_bound[0]
    ymin = top_rect.y0
    bot_bound = page.search_for(bot_str)
    if not bot_str: 
        print("Table bottom bounds not found. Take bottom of the page.")
        ymax = 9999
    else: 
        bot_rect = bot_bound[0]
        ymax = bot_rect.y0
    if not ymin < ymax: 
        raise ValueError("Table bottom bound exceeds top bound. Please readjust widget accordingly. All else fails, FON Liz Cheng, thanks.")
    table = fitz.Rect([0, ymin, 9999, ymax])
    return table 

#coordinate-based parsing
def parse_table(page, bbox): 
    raw_text = page.get_text("words", clip = bbox)
    text_df = pd.DataFrame(data = raw_text)
    rows_group = text_df.groupby(by = [5])
    table_dict = {}
    for i, row_obj in rows_group: 
        row_df = pd.DataFrame(row_obj)
        temp = get_row(row_df) 
        table_dict[f"row_{i}"] = temp
    return table_dict

def get_row(row_df): 
    cell_df = row_df.groupby(by = [6])
    row = []
    for i, ialue in cell_df: 
        cell_val_df = pd.DataFrame(data = ialue)
        cell_val = ""
        for j in cell_val_df[4].to_list(): 
            cell_val = cell_val + " " + j
        row.append(cell_val)
    return row
    
