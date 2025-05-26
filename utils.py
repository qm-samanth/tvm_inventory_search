import csv
import re

# --- Constants ---
ALLOWED_PARAMS = {
    'year', 'make', 'model', 'trim', 'color', 'vehicletypes', 'transmissions',
    'featuresubcategories', 'paymentmax', 'paymentmin', 'type',
    'mileagemin', 'mileagemax'
}

SUPPORTED_TYPES = [
    'convertible', 'coupe', 'suv', 'sedan', 'truck', 'van', 'wagon', 'hatchback', 'mpv'
]

MODEL_TO_MAKE_DATA = {} # Global dict to store loaded CSV data

MILEAGE_VALUE_STRIP_KEYWORDS = [
    "under ", "less than ", "at most ", "maximum ", " up to ", "below ",
    "over ", "starting at ", "more than ", "at least ", "minimum "
]

# --- Helper Functions ---
def is_effectively_none_or_absent(param_val):
    if param_val is None:
        return True
    if isinstance(param_val, str) and param_val.strip().lower() in ["", "null", "none"]:
        return True
    return False

def normalize_price_for_comparison(val_str):
    if is_effectively_none_or_absent(val_str):
        return None
    s = str(val_str).lower().strip().replace("$", "").replace(",", "")
    if s in ["null", "none"]: return None

    multiplier = 1
    if s.endswith('k'): 
        multiplier = 1000; s = s[:-1].strip()
    elif s.endswith('m'): 
        multiplier = 1000000; s = s[:-1].strip()
    
    if not s or s in ["null", "none"]: return None
    
    try:
        return float(s) * multiplier
    except ValueError:
        print(f"[DEBUG] normalize_price_for_comparison: ValueError converting '{s}' to float.")
        return None

def strip_mileage_keywords_from_value(val_str_input):
    s = str(val_str_input).lower().strip() # Work with lowercase
    for keyword in MILEAGE_VALUE_STRIP_KEYWORDS:
        if s.startswith(keyword):
            s = s[len(keyword):].strip()
            break
    return s

def normalize_mileage_for_internal_comparison(val_str_input):
    if is_effectively_none_or_absent(val_str_input):
        return None
    
    s = str(val_str_input)
    s = strip_mileage_keywords_from_value(s)
    s = s.lower().strip().replace(",", "")

    if s.endswith('k'):
        s = s[:-1].strip()
        try: return float(s) * 1000 if s else None
        except ValueError: return None
    elif "miles".casefold() in s: 
        s = s.replace("miles","").strip()

    if not s or s in ["null", "none"]: return None
    try:
        return float(s)
    except ValueError:
        print(f"[DEBUG] normalize_mileage_for_internal_comparison: ValueError converting '{s}' (original: '{val_str_input}') to float.")
        return None

def normalize_value_for_collision_check(val_str_input):
    if is_effectively_none_or_absent(val_str_input):
        return None
    
    s = str(val_str_input)
    s_temp_mileage_norm = strip_mileage_keywords_from_value(s)
    s_temp_mileage_norm = s_temp_mileage_norm.lower().replace(",", "").replace("miles", "").strip()
    s_temp_price_norm = s.lower().replace("$","").replace(",","").strip()

    if len(s_temp_mileage_norm) < len(s_temp_price_norm) and s_temp_mileage_norm.replace('.','',1).isdigit():
        s_for_num = s_temp_mileage_norm
    else:
        s_for_num = s_temp_price_norm
    
    multiplier = 1
    if s_for_num.endswith('k'):
        multiplier = 1000; s_for_num = s_for_num[:-1].strip()
    elif s_for_num.endswith('m'):
        multiplier = 1000000; s_for_num = s_for_num[:-1].strip()

    if not s_for_num or not s_for_num.replace('.','',1).isdigit():
        num_val = normalize_mileage_for_internal_comparison(str(val_str_input))
        return num_val

    try:
        return float(s_for_num) * multiplier
    except ValueError:
        print(f"[DEBUG] normalize_value_for_collision_check: ValueError converting '{s_for_num}' (original: '{val_str_input}')")
        return None

def load_model_make_mapping_from_csv():
    """Load model to make mapping from CSV file."""
    # This function now directly modifies the MODEL_TO_MAKE_DATA in this module.
    global MODEL_TO_MAKE_DATA
    mapping_file_path = "model_make_map.csv" 
    temp_model_to_make_data = {}
    try:
        with open(mapping_file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for rows in reader:
                if len(rows) == 2:
                    make_from_csv = rows[0].strip().lower()
                    model_from_csv = rows[1].strip().lower()
                    if model_from_csv and make_from_csv:
                        temp_model_to_make_data[model_from_csv] = make_from_csv
        MODEL_TO_MAKE_DATA.clear()
        MODEL_TO_MAKE_DATA.update(temp_model_to_make_data)
        print(f"[INFO] Successfully loaded {len(MODEL_TO_MAKE_DATA)} model-make mappings from {mapping_file_path} into utils.MODEL_TO_MAKE_DATA")
    except FileNotFoundError:
        print(f"[ERROR] {mapping_file_path} not found. Model-make correction will not work.")
    except Exception as e:
        print(f"[ERROR] Failed to load {mapping_file_path}: {e}")

