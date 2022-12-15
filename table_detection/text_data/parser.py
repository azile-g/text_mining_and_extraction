import fitz
import json
import pandas as pd

class text_table_parser: 

    def find_table(page, title_str, bot_str, top_idx = 0, bot_idx = 0): 
        table_name = str(title_str)
        top_bound = page.search_for(table_name)
        if not top_bound: 
            raise ValueError("Table title not found. Please readjust widget accordingly. All else fails, FON Liz Cheng, thanks.")
        top_rect = top_bound[top_idx]
        ymin = top_rect.y0
        bot_bound = page.search_for(bot_str)
        if not bot_bound: 
            print("Table bottom bounds not found. Take bottom of the page.")
            ymax = 9999
        else: 
            print(bot_bound)
            bot_rect = bot_bound[bot_idx]
            ymax = bot_rect.y1
        if not ymin < ymax: 
            raise ValueError("Table bottom bound exceeds top bound. Please readjust widget accordingly. All else fails, FON Liz Cheng, thanks.")
        table = fitz.Rect([0, ymin, 9999, ymax])
        return table 

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

    def parse_table(page, bbox, merged_rows = False): 
        raw_text = page.get_text("words", clip = bbox)
        text_df = pd.DataFrame(data = raw_text)
        rows_group = text_df.groupby(by = [5])
        table_dict = {}
        if merged_rows == False:
            for i, row_obj in rows_group: 
                row_df = pd.DataFrame(row_obj)
                temp = text_table_parser.get_row(row_df) 
                table_dict[f"row_{i}"] = temp
        elif merged_rows == True: 
            for i, row_obj in rows_group: 
                row_df = pd.DataFrame(row_obj)
                display(row_df)
                temp = text_table_parser.get_row(row_df) 
                temp_coords = min(row_df[0].tolist())
                print(temp_coords)
                table_dict[f"row_{i}"] = temp
        return table_dict, text_df

class doc_type_init: 

    def __init__(self, target_keys, *kwrds): 
        self.keys = target_keys
        self.kwds = kwrds
    
    def create_target_template(self): 
        template_dict = {"target": {}, "kwds": {"front": [], "back": []}}
        for i in self.keys: 
            if type(i) == list: 
                if type(i[0]) == int: 
                    template_dict["target"][i[1]] = [{j: [] for j in i[2:]} for k in range(i[0])]
                else: 
                    template_dict["target"][i[0]] = {j: [] for j in i[1:]}
            else: 
                template_dict["target"][i] = []
        for i, ialue in enumerate(self.kwds): 
            if i/2 == i//2:
                template_dict["kwds"]["front"].append(ialue)
            else: 
                template_dict["kwds"]["back"].append(ialue)
        return template_dict
