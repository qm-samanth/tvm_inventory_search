import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlencode
import json
import re

# Imports from our new modules
from llm_interface import get_llm_params_from_query # Changed to absolute import
from utils import (
    is_effectively_none_or_absent,
    normalize_price_for_comparison,
    strip_mileage_keywords_from_value,
    normalize_mileage_for_internal_comparison,
    normalize_value_for_collision_check,
    load_model_make_mapping_from_csv,
    MODEL_TO_MAKE_DATA, # Import the dictionary itself
    ALLOWED_PARAMS,
    SUPPORTED_TYPES,
    VALID_QUERY_PARAM_TYPES # Add this import
) # Changed to absolute import

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Load model to make mapping on application startup."""
    load_model_make_mapping_from_csv() # Call the function from utils

# Allow CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def extract_params(user_query):
    # Step 1: Get initial params from LLM
    params = get_llm_params_from_query(user_query) # Use the new function from llm_interface
    if not params: # If params is empty due to an error in _get_llm_params
        print("[DEBUG] get_llm_params_from_query returned empty. Aborting extract_params.")
        return {}

    print("[DEBUG] Initial params from LLM:", params)
    user_query_lower = user_query.lower()
    print("[DEBUG] User query (lower):", user_query_lower)

    # --- Handle "certified" as "cpo" for type ---
    if "certified" in user_query_lower and "cpo" not in user_query_lower:
        # If LLM didn\'t pick up "type" or picked up something else, 
        # and "certified" is in query, force type to "cpo".
        # This also handles if LLM put "certified" in params['type'].
        llm_type = params.get("type")
        if isinstance(llm_type, str) and "certified" in llm_type.lower():
            print(f"[DEBUG] LLM type was '{llm_type}'. Query contains 'certified'. Setting type to 'cpo'.")
            params["type"] = "cpo"
        elif is_effectively_none_or_absent(llm_type):
            print(f"[DEBUG] Query contains 'certified' and LLM type is absent. Setting type to 'cpo'.")
            params["type"] = "cpo"
        # If LLM picked up another valid type, and "certified" is also in query, 
        # we might need a more nuanced rule, but for now, "certified" implies "type=cpo".
        # This could override a more specific type if LLM found one, e.g. "certified sedan".
        # Current logic will make it "cpo". If "vehicletypes" is "sedan", that will remain.

    # --- Start of Make/Model Correction from LOADED_MODEL_MAKE_MAPPING ---
    llm_make_val = params.get("make")
    llm_model_val = params.get("model")

    # Diagnostic prints
    print(f"[DIAGNOSTIC] llm_make_val before correction: '{llm_make_val}', llm_model_val before correction: '{llm_model_val}'")
    if not MODEL_TO_MAKE_DATA: # Use MODEL_TO_MAKE_DATA from utils.py
        print("[DIAGNOSTIC] utils.MODEL_TO_MAKE_DATA is empty or not loaded. Skipping make/model correction.")
    # print(f"[DIAGNOSTIC] utils.MODEL_TO_MAKE_DATA has {len(MODEL_TO_MAKE_DATA)} entries.") # Optional: for verbosity

    if MODEL_TO_MAKE_DATA and isinstance(llm_make_val, str): # Only proceed if mapping exists and llm_make_val is a string
        llm_make_lower = llm_make_val.lower()
        
        if llm_make_lower in MODEL_TO_MAKE_DATA: # Use MODEL_TO_MAKE_DATA from utils.py
            true_make_from_csv = MODEL_TO_MAKE_DATA[llm_make_lower] # Use MODEL_TO_MAKE_DATA from utils.py
            # potential_model_name_if_llm_misplaced is llm_make_val itself, as it\'s the value LLM put in the \'make\' field.
            potential_model_name_if_llm_misplaced = llm_make_val 

            print(f"[DIAGNOSTIC] Candidate for make correction: llm_make_val='{llm_make_val}', llm_make_lower='{llm_make_lower}', maps to true_make_from_csv='{true_make_from_csv}'")

            # Condition to apply correction:
            # 1. The true make (from CSV) for what LLM called 'make' is different from what LLM called 'make'
            #    (e.g., LLM make: "civic", CSV says "civic" is model of "honda". So, "honda" != "civic")
            # OR
            # 2. The LLM's model field is empty (and llm_make_val was found in CSV, so it might be a model that needs to populate the model field)
            #    (e.g., LLM make: "civic", model: None OR LLM make: "honda", model: None if "honda" is a key in MODEL_TO_MAKE_DATA)
            if true_make_from_csv.lower() != llm_make_lower or is_effectively_none_or_absent(llm_model_val):
                print(f"[DEBUG] Make Correction Triggered. Original LLM make: '{llm_make_val}', LLM model: '{llm_model_val}'. True make for '{llm_make_lower}' from CSV: '{true_make_from_csv}'")
                
                params["make"] = true_make_from_csv # Correct the make

                # Now, adjust the model field based on this correction.
                # potential_model_name_if_llm_misplaced holds the value LLM originally put in 'make'.
                
                # Case 1: LLM's make was a make (e.g., "honda"), its true make from CSV is itself ("honda"), 
                #         and LLM's model field was empty.
                # Action: Make is correct. Model should remain absent (not become the make name).
                if true_make_from_csv.lower() == llm_make_lower and is_effectively_none_or_absent(llm_model_val):
                    print(f"[DEBUG] Make Correction: llm_make_val ('{llm_make_val}') is a make. LLM model was absent. Model field remains absent/as is.")
                    # If llm_model_val was originally None, "", "null" etc., ensure 'model' key is removed if it exists with such a value.
                    if "model" in params and is_effectively_none_or_absent(params.get("model")):
                        params.pop("model")
                
                # Case 2: LLM's model field was empty, and llm_make_val was a model name (e.g. "civic", true make "honda").
                # Action: Set model to this model name.
                elif is_effectively_none_or_absent(llm_model_val):
                    params["model"] = potential_model_name_if_llm_misplaced
                    print(f"[DEBUG] Make Correction: Set model to '{params['model']}' (from original LLM make value: '{potential_model_name_if_llm_misplaced}'). LLM model was absent.")
                
                # Case 3: LLM's model field had a value.
                # Action: Prepend potential_model_name_if_llm_misplaced if it's not already part of the LLM's model string.
                elif isinstance(llm_model_val, str):
                    if potential_model_name_if_llm_misplaced.lower() not in llm_model_val.lower():
                        params["model"] = f"{potential_model_name_if_llm_misplaced} {llm_model_val}".strip()
                        print(f"[DEBUG] Make Correction: Combined '{potential_model_name_if_llm_misplaced}' with LLM model '{llm_model_val}'. New model: '{params['model']}'")
                    else:
                        # LLM's model value already contains the model name.
                        # e.g., llm_make="civic" (true_make="honda"), llm_model="civic si". Corrected make="honda". Model remains "civic si".
                        print(f"[DEBUG] Make Correction: LLM model ('{llm_model_val}') already contains the potential model name ('{potential_model_name_if_llm_misplaced}'). Model remains '{llm_model_val}'.")
                # else: llm_model_val was not a string and not absent (e.g., a number). Leave model as is in this sub-block.
                
            else:
                # This 'else' is hit if:
                # llm_make_lower was in MODEL_TO_MAKE_DATA (e.g. "honda" -> "honda" in CSV) AND
                # true_make_from_csv.lower() == llm_make_lower (e.g. "honda" == "honda") AND
                # llm_model_val was NOT absent (e.g. llm_model_val was "pilot")
                # This is the "Brand new Honda Pilot" case. LLM is likely correct. No correction needed from this block.
                print(f"[DEBUG] Make/Model appears correct or not fitting primary correction criteria. LLM make: '{llm_make_val}', LLM model: '{llm_model_val}'. No changes made by this correction block.")
        else:
            # llm_make_val (lowercase) was not found as a key in MODEL_TO_MAKE_DATA.
            print(f"[DIAGNOSTIC] llm_make_val ('{llm_make_val}') not found as a key in utils.MODEL_TO_MAKE_DATA. No make/model correction based on it.")
    elif not MODEL_TO_MAKE_DATA: # Use MODEL_TO_MAKE_DATA from utils.py
        pass # Diagnostic already printed if MODEL_TO_MAKE_DATA is empty
    elif not isinstance(llm_make_val, str):
        # llm_make_val was not a string (e.g., None, or some other type from LLM).
        print(f"[DIAGNOSTIC] llm_make_val ('{llm_make_val}') is not a string. Skipping make/model correction.")

    print(f"[DIAGNOSTIC] Params after make/model correction attempt: make='{params.get('make')}', model='{params.get('model')}'")

    # --- Start of Non-CSV Model Name Removal ---
    current_model_val = params.get("model")
    if isinstance(current_model_val, str):
        model_val_lower = current_model_val.lower() # LLM's model, lowercased

        found_in_csv = False
        # Check 1: Original form from LLM (lowercased)
        if model_val_lower in MODEL_TO_MAKE_DATA:
            found_in_csv = True
            print(f"[DEBUG] Model '{current_model_val}' (as '{model_val_lower}') found directly in MODEL_TO_MAKE_DATA.")
        else:
            # Check 2: Convert LLM's model's hyphens to spaces and check
            model_as_spaced = model_val_lower.replace('-', ' ')
            if model_as_spaced != model_val_lower and model_as_spaced in MODEL_TO_MAKE_DATA:
                found_in_csv = True
                print(f"[DEBUG] Model '{current_model_val}' (normalized to '{model_as_spaced}') found in MODEL_TO_MAKE_DATA.")
            else:
                # Check 3: Convert LLM's model's spaces to hyphens and check
                model_as_hyphenated = model_val_lower.replace(' ', '-')
                if model_as_hyphenated != model_val_lower and model_as_hyphenated in MODEL_TO_MAKE_DATA:
                    found_in_csv = True
                    print(f"[DEBUG] Model '{current_model_val}' (normalized to '{model_as_hyphenated}') found in MODEL_TO_MAKE_DATA.")
        
        if not found_in_csv:
            print(f"[DEBUG] Model was '{current_model_val}'. Not found in MODEL_TO_MAKE_DATA after checking variations. Removing model parameter.")
            params.pop("model", None)
        else:
            # If the model is valid, ensure it's stored in a consistently hyphenated format in params 
            # for consistency before build_inventory_url, which also hyphenates.
            params['model'] = model_val_lower.replace(' ', '-')
            print(f"[DEBUG] Model '{current_model_val}' is valid. Storing/ensuring as hyphenated: '{params['model']}'.")
    # --- End of Non-CSV Model Name Removal ---

    # --- Start of Make-Model Consistency Check ---
    # This block ensures that if a valid model is present, the make parameter aligns with MODEL_TO_MAKE_DATA.
    # It runs AFTER the "Non-CSV Model Name Removal" which validates and normalizes params['model'].
    final_model_val = params.get("model")
    final_make_val = params.get("make")

    if isinstance(final_model_val, str) and isinstance(final_make_val, str) and MODEL_TO_MAKE_DATA:
        # model_val should already be lowercased and hyphenated by the "Non-CSV Model Name Removal" block.
        # Keys in MODEL_TO_MAKE_DATA are lowercased.
        model_key_for_lookup = final_model_val # Already normalized (lower, hyphenated)
        
        if model_key_for_lookup in MODEL_TO_MAKE_DATA:
            true_make_for_model_from_csv = MODEL_TO_MAKE_DATA[model_key_for_lookup]
            current_make_in_params_lower = final_make_val.lower()

            if current_make_in_params_lower != true_make_for_model_from_csv:
                print(f"[DEBUG] Make-Model Inconsistency Correction: Initial make was '{final_make_val}', model was '{final_model_val}'. "
                      f"Model '{model_key_for_lookup}' is associated with make '{true_make_for_model_from_csv}' in CSV. "
                      f"Updating make from '{final_make_val}' to '{true_make_for_model_from_csv}'.")
                params["make"] = true_make_for_model_from_csv
            else:
                print(f"[DEBUG] Make-Model Consistency Check: Make '{final_make_val}' for model '{final_model_val}' is consistent with CSV data ('{true_make_for_model_from_csv}'). No change.")
        else:
            # This case implies that params["model"] exists but is not a key in MODEL_TO_MAKE_DATA.
            # This should ideally be prevented by the "Non-CSV Model Name Removal" block,
            # which should have removed such a model. This log is a safeguard.
            print(f"[WARN] Make-Model Consistency Check: Model '{model_key_for_lookup}' (from params) not found as a key in MODEL_TO_MAKE_DATA. "
                  f"Cannot verify/correct make consistency for make '{final_make_val}'. This might indicate an issue with prior model validation steps.")
    
    print(f"[DIAGNOSTIC] Params after Make-Model Consistency Check: make='{params.get('make')}', model='{params.get('model')}'")
    # --- End of Make-Model Consistency Check ---

    # --- Start of Final Make Validation ---
    # This block ensures that the 'make' parameter, after all corrections, is a known make.
    current_make_val = params.get("make")
    if isinstance(current_make_val, str) and MODEL_TO_MAKE_DATA:
        # Get all unique makes from the CSV data (values in MODEL_TO_MAKE_DATA)
        # Ensure they are all lowercased for consistent comparison.
        known_makes_from_csv = {make_name.lower() for make_name in MODEL_TO_MAKE_DATA.values()}
        
        if current_make_val.lower() not in known_makes_from_csv:
            print(f"[DEBUG] Final Make Validation: Make value '{current_make_val}' is not a recognized make found in MODEL_TO_MAKE_DATA values. Removing 'make' parameter.")
            params.pop("make", None)
        else:
            print(f"[DEBUG] Final Make Validation: Make value '{current_make_val}' is a recognized make. No change.")
    elif isinstance(current_make_val, str) and not MODEL_TO_MAKE_DATA:
        print("[WARN] Final Make Validation: MODEL_TO_MAKE_DATA is not loaded. Cannot validate make parameter.")
    # --- End of Final Make Validation ---

    # --- Start of Price Logic Refinement ---

    upper_bound_keywords = ["under ", "less than ", "at most ", "maximum ", " up to ", "below "] # ADDED "below "
    query_has_upper_bound_keyword = any(keyword in user_query_lower for keyword in upper_bound_keywords)

    lower_bound_keywords = ["over ", "starting at ", "more than ", "at least ", "minimum "]
    query_has_lower_bound_keyword = any(keyword in user_query_lower for keyword in lower_bound_keywords)

    llm_paymentmin_raw_val = params.get("paymentmin")
    llm_paymentmax_raw_val = params.get("paymentmax")

    norm_llm_min_for_swap_check = normalize_price_for_comparison(llm_paymentmin_raw_val)
    norm_llm_max_for_swap_check = normalize_price_for_comparison(llm_paymentmax_raw_val)

    # 1. Correction for "under X" type queries (query has upper bound, no lower bound)
    if query_has_upper_bound_keyword and not query_has_lower_bound_keyword:
        # If paymentmin (from LLM) has a valid number, and paymentmax (from LLM) does not,
        # or paymentmax is zero when paymentmin is not (which would be odd for "under X").
        if norm_llm_min_for_swap_check is not None and \
           (norm_llm_max_for_swap_check is None or \
            (norm_llm_max_for_swap_check == 0 and norm_llm_min_for_swap_check != 0)):
            print(f"[DEBUG] Query implies 'under X'. LLM's paymentmin ('{llm_paymentmin_raw_val}' -> {norm_llm_min_for_swap_check}) seems to be the correct value, "
                  f"and paymentmax ('{llm_paymentmax_raw_val}' -> {norm_llm_max_for_swap_check}) is not valid or is an unexpected zero. "
                  f"Setting paymentmax to paymentmin's value and removing paymentmin.")
            params["paymentmax"] = llm_paymentmin_raw_val # Use the raw value, normalization happens later for all params
            if "paymentmin" in params: params.pop("paymentmin")

    # 2. Correction for "over X" type queries (query has lower bound, no upper bound)
    elif query_has_lower_bound_keyword and not query_has_upper_bound_keyword:
        # If paymentmax (from LLM) has a valid number, and paymentmin (from LLM) does not,
        # or paymentmin is zero when paymentmax is not (odd for "over X").
        if norm_llm_max_for_swap_check is not None and \
           (norm_llm_min_for_swap_check is None or \
            (norm_llm_min_for_swap_check == 0 and norm_llm_max_for_swap_check != 0)):
            print(f"[DEBUG] Query implies 'over X'. LLM's paymentmax ('{llm_paymentmax_raw_val}' -> {norm_llm_max_for_swap_check}) seems to be the correct value, "
                  f"and paymentmin ('{llm_paymentmin_raw_val}' -> {norm_llm_min_for_swap_check}) is not valid or is an unexpected zero. "
                  f"Setting paymentmin to paymentmax's value and removing paymentmax.")
            params["paymentmin"] = llm_paymentmax_raw_val # Use the raw value
            if "paymentmax" in params: params.pop("paymentmax")

    # 3. Handle cases where LLM might have set paymentmin == paymentmax for "under X" queries,
    #    or paymentmin > paymentmax. This runs *after* potential swaps above.
    #    Re-fetch current min/max from params dict as they might have changed.
    current_paymentmin_val = params.get("paymentmin")
    current_paymentmax_val = params.get("paymentmax")

    if not is_effectively_none_or_absent(current_paymentmin_val) and not is_effectively_none_or_absent(current_paymentmax_val):
        if query_has_upper_bound_keyword and not query_has_lower_bound_keyword: # Specifically for "under X"
            norm_min = normalize_price_for_comparison(current_paymentmin_val)
            norm_max = normalize_price_for_comparison(current_paymentmax_val)

            if norm_min is not None and norm_min == norm_max and norm_min != 0:
                print(f"[DEBUG] Query implies 'under X' and paymentmin ('{current_paymentmin_val}') == paymentmax ('{current_paymentmax_val}'). Unsetting paymentmin to allow defaulting to 0.")
                params.pop("paymentmin")
            elif norm_min is not None and norm_max is not None and norm_min > norm_max:
                print(f"[DEBUG] For 'under X' query, paymentmin ('{current_paymentmin_val}') > paymentmax ('{current_paymentmax_val}'). This is illogical. Removing paymentmin.")
                params.pop("paymentmin")
        else: # General check for min > max (not specific to "under X", could be LLM error on a range)
            norm_min = normalize_price_for_comparison(current_paymentmin_val)
            norm_max = normalize_price_for_comparison(current_paymentmax_val)
            if norm_min is not None and norm_max is not None and norm_min > norm_max:
                print(f"[DEBUG] paymentmin ('{current_paymentmin_val}') > paymentmax ('{current_paymentmax_val}'). This is generally illogical. Removing paymentmin.")
                params.pop("paymentmin")
    
    # --- End of Price Logic Refinement ---

    # --- Start of Mileage Logic Refinement ---
    # This section aims to correctly interpret LLM's mileage outputs.

    upper_bound_mileage_keywords = ["under ", "less than ", "at most ", "maximum ", " up to ", "below "]
    lower_bound_mileage_keywords = ["over ", "starting at ", "more than ", "at least ", "minimum "]

    # Revised mileage keyword detection
    contains_mileage_term = "mileage" in user_query_lower or "miles" in user_query_lower
    
    query_has_upper_bound_mileage_keyword = contains_mileage_term and any(ukw in user_query_lower for ukw in upper_bound_mileage_keywords)
    query_has_lower_bound_mileage_keyword = contains_mileage_term and any(lkw in user_query_lower for lkw in lower_bound_mileage_keywords)

    print(f"[DEBUG] Mileage Keyword Flags: contains_mileage_term={contains_mileage_term}, query_has_upper_bound_mileage_keyword={query_has_upper_bound_mileage_keyword}, query_has_lower_bound_mileage_keyword={query_has_lower_bound_mileage_keyword}")

    llm_mileagemin = params.get("mileagemin")
    llm_mileagemax = params.get("mileagemax")

    if query_has_upper_bound_mileage_keyword and not query_has_lower_bound_mileage_keyword:
        if not is_effectively_none_or_absent(llm_mileagemin) and is_effectively_none_or_absent(llm_mileagemax):
            print(f"[DEBUG] Query implies 'under X miles'. LLM provided mileagemin ('{llm_mileagemin}') but mileagemax is absent/null. Swapping mileagemin to mileagemax.")
            params["mileagemax"] = params.pop("mileagemin")
            if "mileagemin" in params: params.pop("mileagemin") # Ensure it's gone

    elif query_has_lower_bound_mileage_keyword and not query_has_upper_bound_mileage_keyword:
         if not is_effectively_none_or_absent(llm_mileagemax) and is_effectively_none_or_absent(llm_mileagemin):
            print(f"[DEBUG] Query implies 'over X miles'. LLM provided mileagemax ('{llm_mileagemax}') but mileagemin is absent/null. Swapping mileagemax to mileagemin.")
            params["mileagemin"] = params.pop("mileagemax")
            if "mileagemax" in params: params.pop("mileagemax") # Ensure it's gone
    
    # Re-fetch after potential swaps
    current_mileagemin_str = params.get("mileagemin")
    current_mileagemax_str = params.get("mileagemax")

    norm_mileagemin_for_logic = normalize_mileage_for_internal_comparison(current_mileagemin_str)
    norm_mileagemax_for_logic = normalize_mileage_for_internal_comparison(current_mileagemax_str)

    if norm_mileagemin_for_logic is not None and norm_mileagemax_for_logic is not None:
        if query_has_upper_bound_mileage_keyword and not query_has_lower_bound_mileage_keyword: # "under X miles"
            if norm_mileagemin_for_logic == norm_mileagemax_for_logic and norm_mileagemin_for_logic != 0:
                print(f"[DEBUG] Query implies 'under X miles' and mileagemin ('{current_mileagemin_str}') == mileagemax ('{current_mileagemax_str}'). Unsetting mileagemin.")
                params.pop("mileagemin")
            elif norm_mileagemin_for_logic > norm_mileagemax_for_logic:
                print(f"[DEBUG] For 'under X miles' query, mileagemin ('{current_mileagemin_str}') > mileagemax ('{current_mileagemax_str}'). Removing mileagemin.")
                params.pop("mileagemin")
        elif norm_mileagemin_for_logic > norm_mileagemax_for_logic: # General case if min > max
            print(f"[DEBUG] mileagemin ('{current_mileagemin_str}') > mileagemax ('{current_mileagemax_str}'). Removing mileagemin.")
            params.pop("mileagemin")

    # Default mileagemin to 0 if mileagemax is present and mileagemin is effectively absent
    if not is_effectively_none_or_absent(params.get("mileagemax")) and is_effectively_none_or_absent(params.get("mileagemin")):
        params["mileagemin"] = "0" # Set as string "0", will be normalized to int later
        print(f"[DEBUG] Defaulted mileagemin to 0 as mileagemax is present and mileagemin was effectively absent.")

    # --- End of Mileage Logic Refinement ---

    # --- Start of Re-routing and Collision Checks --- 

    price_keywords_pattern = r"(\$|\€|\£|\¥|\₹|dollar|euro|pound|yen|rupee|aud|price|cost|budget|payment)"
    query_has_strong_price_keywords = bool(re.search(price_keywords_pattern, user_query_lower))

    # query_has_upper_bound_mileage_keyword and query_has_lower_bound_mileage_keyword are already defined
    # We also need a general check for any mileage keyword if those aren't set
    generic_mileage_keywords_pattern = r"(mile|mileage|km|kilometer)"
    query_has_generic_mileage_keywords = bool(re.search(generic_mileage_keywords_pattern, user_query_lower))
    query_context_is_mileage = query_has_upper_bound_mileage_keyword or query_has_lower_bound_mileage_keyword or query_has_generic_mileage_keywords

    # --- NEW: Re-routing logic for LLM misplacing mileage numbers into price fields ---
    if query_context_is_mileage and not query_has_strong_price_keywords:
        # Check for upper-bound mileage context (e.g., "below X miles", "under X miles")
        if query_has_upper_bound_mileage_keyword:
            if not is_effectively_none_or_absent(params.get("paymentmin")) and is_effectively_none_or_absent(params.get("mileagemax")):
                val_to_move = params.pop("paymentmin")
                params["mileagemax"] = val_to_move
                print(f"[DEBUG] Re-routing (mileage context): Moved LLM value from paymentmin ('{val_to_move}') to mileagemax.")
            elif not is_effectively_none_or_absent(params.get("paymentmax")) and is_effectively_none_or_absent(params.get("mileagemax")):
                # This case is less likely if LLM puts "below X" into paymentmax, but included for completeness
                val_to_move = params.pop("paymentmax")
                params["mileagemax"] = val_to_move
                print(f"[DEBUG] Re-routing (mileage context): Moved LLM value from paymentmax ('{val_to_move}') to mileagemax.")
        
        # Check for lower-bound mileage context (e.g., "over X miles", "at least X miles")
        elif query_has_lower_bound_mileage_keyword:
            if not is_effectively_none_or_absent(params.get("paymentmax")) and is_effectively_none_or_absent(params.get("mileagemin")):
                val_to_move = params.pop("paymentmax")
                params["mileagemin"] = val_to_move
                print(f"[DEBUG] Re-routing (mileage context): Moved LLM value from paymentmax ('{val_to_move}') to mileagemin.")
            elif not is_effectively_none_or_absent(params.get("paymentmin")) and is_effectively_none_or_absent(params.get("mileagemin")):
                # This case is less likely if LLM puts "over X" into paymentmin, but included for completeness
                val_to_move = params.pop("paymentmin")
                params["mileagemin"] = val_to_move
                print(f"[DEBUG] Re-routing (mileage context): Moved LLM value from paymentmin ('{val_to_move}') to mileagemin.")

    # --- Existing Cross-Contamination/Collision Check (Numeric comparison) ---

    # Get current raw values that might be in params
    raw_paymentmin = params.get("paymentmin")
    raw_paymentmax = params.get("paymentmax")
    raw_mileagemin = params.get("mileagemin")
    raw_mileagemax = params.get("mileagemax")

    comp_paymentmin = normalize_value_for_collision_check(raw_paymentmin)
    comp_paymentmax = normalize_value_for_collision_check(raw_paymentmax)
    comp_mileagemin = normalize_value_for_collision_check(raw_mileagemin)
    comp_mileagemax = normalize_value_for_collision_check(raw_mileagemax)

    # Collision Check 1: paymentmax == mileagemax
    if comp_paymentmax is not None and comp_paymentmax == comp_mileagemax:
        if query_context_is_mileage and not query_has_strong_price_keywords:
            print(f"[DEBUG] Collision: paymentmax ({raw_paymentmax} -> {comp_paymentmax}) == mileagemax ({raw_mileagemax} -> {comp_mileagemax}). Query context is mileage. Popping paymentmax.")
            params.pop("paymentmax", None)
            # If paymentmin was 0 and likely defaulted due to this paymentmax, it should also be considered.
            # However, the main paymentmin defaulting runs after this, so if paymentmax is gone, paymentmin won't be defaulted from it.
            # If paymentmin was also a collision (see below), it will be handled.

    # Collision Check 2: paymentmin == mileagemin
    if comp_paymentmin is not None and comp_paymentmin == comp_mileagemin:
        if query_context_is_mileage and not query_has_strong_price_keywords:
            print(f"[DEBUG] Collision: paymentmin ({raw_paymentmin} -> {comp_paymentmin}) == mileagemin ({raw_mileagemin} -> {comp_mileagemin}). Query context is mileage. Popping paymentmin.")
            params.pop("paymentmin", None)
    
    # Collision Check 3: paymentmax == mileagemin (e.g. query "over 75k miles", LLM puts 75k in paymentmax and mileagemin)
    if comp_paymentmax is not None and comp_paymentmax == comp_mileagemin:
        if query_context_is_mileage and not query_has_strong_price_keywords:
            print(f"[DEBUG] Collision: paymentmax ({raw_paymentmax} -> {comp_paymentmax}) == mileagemin ({raw_mileagemin} -> {comp_mileagemin}). Query context is mileage. Popping paymentmax.")
            params.pop("paymentmax", None)

    # Collision Check 4: paymentmin == mileagemax (e.g. query "under 75k miles", LLM puts 75k in paymentmin and mileagemax)
    if comp_paymentmin is not None and comp_paymentmin == comp_mileagemax:
        if query_context_is_mileage and not query_has_strong_price_keywords:
            print(f"[DEBUG] Collision: paymentmin ({raw_paymentmin} -> {comp_paymentmin}) == mileagemax ({raw_mileagemax} -> {comp_mileagemax}). Query context is mileage. Popping paymentmin.")
            params.pop("paymentmin", None)

    # --- End of Cross-Contamination/Collision Check ---

    # Default paymentmin to 0 if paymentmax is present (and not effectively absent) 
    # and paymentmin is effectively absent (or was unset by prior logic).
    if not is_effectively_none_or_absent(params.get('paymentmax')) and is_effectively_none_or_absent(params.get('paymentmin')):
        params['paymentmin'] = 0 
        print(f"[DEBUG] Defaulted paymentmin to 0 as paymentmax is present and paymentmin was effectively absent.")

    # Default mileagemin to 0 if mileagemax is present and mileagemin is effectively absent
    if not is_effectively_none_or_absent(params.get("mileagemax")) and is_effectively_none_or_absent(params.get("mileagemin")):
        params["mileagemin"] = "0" # Set as string "0", will be normalized to int later
        print(f"[DEBUG] Defaulted mileagemin to 0 as mileagemax is present and mileagemin was effectively absent.")

    # Step 5.1: Normalize 'paymentmin' and 'paymentmax' values
    for pay_key in ["paymentmin", "paymentmax"]:
        if pay_key in params:
            val = params[pay_key]
            if val is None: # Explicitly handle None from LLM
                print(f"[DEBUG] Removing {pay_key} because LLM provided null/None.")
                params.pop(pay_key)
                continue

            val_str = str(val).strip().lower() # Convert to string, strip whitespace, lowercase

            if not val_str: # If, after stripping, it's an empty string
                print(f"[DEBUG] Removing {pay_key} because value became empty string after stripping: original ('{val}')")
                params.pop(pay_key)
                continue
            
            # Remove currency symbols and commas
            val_str = val_str.replace("$", "").replace(",", "")

            multiplier = 1
            if val_str.endswith('k'):
                multiplier = 1000
                val_str = val_str[:-1].strip() 
            elif val_str.endswith('m'):
                multiplier = 1000000
                val_str = val_str[:-1].strip()
            
            # If after k/m stripping, it's empty again (e.g. "k" by itself)
            if not val_str: # Check if val_str became empty after stripping k/m
                 print(f"[DEBUG] Removing {pay_key} because value became empty after k/m processing: original ('{val}')")
                 params.pop(pay_key)
                 continue
            
            try:
                num_val = float(val_str) * multiplier
                # Ensure that 0 is preserved as int, not float, if it started as such or becomes whole number
                if num_val == 0:
                    params[pay_key] = 0
                elif num_val.is_integer():
                    params[pay_key] = int(num_val)
                else:
                    params[pay_key] = num_val
                print(f"[DEBUG] Normalized {pay_key} (from '{val}') to {params[pay_key]}")
            except (ValueError, TypeError) as e:
                print(f"[DEBUG] Removing {pay_key} (original value '{val}') due to normalization error: {e}")
                if pay_key in params: params.pop(pay_key) # Ensure removal on error
    
    # NEW Step for Mileage Normalization (e.g., Step 5.2)
    for m_key in ["mileagemin", "mileagemax"]:
        if m_key in params:
            val = params[m_key]
            if is_effectively_none_or_absent(val):
                print(f"[DEBUG] Removing {m_key} because value is effectively none/absent: original ('{val}')")
                params.pop(m_key)
                continue

            val_str = str(val) # Start with original string form
            # Strip prefixes like "less than " from the value itself
            val_str = strip_mileage_keywords_from_value(val_str)
            
            # Standard cleaning
            val_str = val_str.lower().strip().replace(",", "").replace("miles", "").strip()

            multiplier = 1
            if val_str.endswith('k'):
                multiplier = 1000
                val_str = val_str[:-1].strip()
            
            if not val_str or val_str in ["null", "none"]: # Check after all stripping
                print(f"[DEBUG] Removing {m_key} because value became empty or 'null'/'none' after cleaning: original ('{val}')")
                params.pop(m_key)
                continue
            
            try:
                num_val = float(val_str) * multiplier 
                if num_val.is_integer():
                    params[m_key] = int(num_val)
                else: 
                    print(f"[DEBUG] Mileage {m_key} ('{val}') resulted in non-integer {num_val} after normalization. Removing.")
                    params.pop(m_key) 
                print(f"[DEBUG] Normalized {m_key} (from '{val}') to {params.get(m_key)}")
            except (ValueError, TypeError) as e:
                print(f"[DEBUG] Removing {m_key} (original value '{val}') due to mileage normalization error: {e}")
                if m_key in params: params.pop(m_key)
    
    # Step 4: Type detection and normalization (used, cpo, new)
    preowned_pattern = r"pre[-\s]?owned|preowned|used"
    certified_preowned_pattern = r"certified[ -]?(pre[-\s]?owned|preowned)"
    preowned_match = re.search(preowned_pattern, user_query_lower, re.IGNORECASE)
    certified_preowned_match = re.search(certified_preowned_pattern, user_query_lower, re.IGNORECASE)
    
    query_words = re.findall(r'\b\w+\b', user_query_lower)
    cpo_keyword_match = 'cpo' in query_words
    new_keyword_match = 'new' in query_words

    print(f"[DEBUG] User query lower: '{user_query_lower}'")
    print(f"[DEBUG] Query words for type check: {query_words}")
    print(f"[DEBUG] Preowned/used pattern match in user query: {preowned_match}")
    print(f"[DEBUG] Certified pre-owned pattern match in user query: {certified_preowned_match}")
    print(f"[DEBUG] CPO keyword match in user query: {cpo_keyword_match}")
    print(f"[DEBUG] NEW keyword match in user query: {new_keyword_match}")

    explicit_type_from_query = None
    if certified_preowned_match or cpo_keyword_match:
        explicit_type_from_query = 'cpo'
        print(f"[DEBUG] Type interpreted as 'cpo' from query text.")
    elif preowned_match:
        explicit_type_from_query = 'used'
        print(f"[DEBUG] Type interpreted as 'used' from query text.")
    elif new_keyword_match:
        explicit_type_from_query = 'new'
        print(f"[DEBUG] Type interpreted as 'new' from query text.")

    # Get the type LLM provided, if any
    llm_provided_type = params.get("type")
    if isinstance(llm_provided_type, str):
        llm_provided_type = llm_provided_type.lower().strip()
        if llm_provided_type not in VALID_QUERY_PARAM_TYPES:
            # Corrected f-string: Use different quotes for the outer f-string and inner string literals,
            # or escape the inner quotes if they must be the same.
            # Here, we ensure the f-string uses single quotes for its value part if the outer uses double, or vice-versa.
            # Simpler: just use the variable `llm_provided_type` which already holds the value of params.get("type")
            print(f"[DEBUG] LLM provided type '{llm_provided_type}' (original: '{params.get('type')}') is not in VALID_QUERY_PARAM_TYPES. Discarding LLM type.")
            llm_provided_type = None # Discard invalid type from LLM
    else:
        llm_provided_type = None # LLM type was not a string or absent

    # Decision logic for final type:
    # Priority 1: Explicit type found in the user query.
    if explicit_type_from_query:
        params["type"] = explicit_type_from_query
        print(f"[DEBUG] Final type set from explicit query mention: {params['type']}")
    # Priority 2: No explicit type in query, so remove any type LLM might have suggested.
    else:
        if "type" in params: # If LLM suggested a type and it's still in params
            print(f"[DEBUG] No explicit type in query. Removing type '{params.get('type')}' that might have been suggested by LLM.")
            params.pop("type")
        else:
            print(f"[DEBUG] No explicit type in query and no type suggested by LLM. 'type' parameter remains absent.")

    # Step 4.1: Clean trim if it duplicates the detected type
    llm_trim_val = params.get('trim')
    current_type = params.get('type')

    if llm_trim_val and isinstance(llm_trim_val, str) and current_type:
        llm_trim_lower = llm_trim_val.lower()
        
        type_related_trim_keywords = {
            'cpo': ["certified pre-owned", "cpo", "certified preowned"],
            'used': ["used", "pre-owned", "preowned"],
            'new': ["new"]
        }

        if current_type in type_related_trim_keywords:
            if llm_trim_lower in type_related_trim_keywords[current_type]:
                print(f"[DEBUG] Removing trim (\'{llm_trim_val}\') as it duplicates the detected type (\'{current_type}\').")
                params.pop('trim', None) # Use None to avoid KeyError if trim was already removed

    # Step 6: Validate and filter vehicletypes
    # Only populate vehicletypes if a supported type is EXPLICITLY mentioned in the query.
    if 'vehicletypes' in params: # Check if LLM even provided it as a starting point
        llm_provided_vehicletype = params.get('vehicletypes') # Store for logging
        print(f"[DEBUG] LLM initially provided vehicletypes: {llm_provided_vehicletype}")
        params.pop('vehicletypes') # Remove it first, we will re-add ONLY if explicit

    explicitly_mentioned_supported_type = None
    for vt in SUPPORTED_TYPES: # Use SUPPORTED_TYPES from utils.py
        # Search for whole word matches of supported types in the user query
        if re.search(rf'\b{re.escape(vt)}\b', user_query_lower):
            explicitly_mentioned_supported_type = vt
            print(f"[DEBUG] Found explicitly mentioned supported type in query: '{vt}'")
            break # Take the first one found
    
    if explicitly_mentioned_supported_type:
        params['vehicletypes'] = explicitly_mentioned_supported_type
        print(f"[DEBUG] Setting vehicletypes to explicitly mentioned supported type: '{params['vehicletypes']}'")
    else:
        # If 'vehicletypes' was removed and no explicit type was found, it remains absent.
        # If LLM provided something but it wasn't explicit, it's now gone.
        print(f"[DEBUG] No explicitly mentioned supported type found in query. 'vehicletypes' will not be included unless it was already handled (e.g. model moved to vehicletypes).")
        # Ensure it's really gone if it wasn't set by explicit mention
        if 'vehicletypes' in params and params['vehicletypes'] != explicitly_mentioned_supported_type:
             # This case should ideally not be hit if logic is correct, but as a safeguard:
             print(f"[DEBUG] Redundant check: Removing vehicletypes as it was not set by explicit mention.")
             params.pop('vehicletypes')

    # Step 6.1: Prevent model from being the same as vehicletype
    current_model = params.get('model')
    current_vehicletype = params.get('vehicletypes')

    if current_model and current_vehicletype and isinstance(current_model, str) and \
       current_model.lower() == current_vehicletype.lower():
        print(f"[DEBUG] Model ('{current_model}') is the same as vehicletypes ('{current_vehicletype}'). Removing model.")
        params.pop('model')

    # Step 7: Year logic and inferring type=used
    current_year = datetime.datetime.now().year
    if 'year' in params:
        try:
            year_val_str = str(params['year']).strip()
            if not year_val_str: # Handle empty string for year
                print("[DEBUG] Year value is empty, removing.")
                params.pop('year')
            else:
                year_val = int(year_val_str)
                params['year'] = year_val # Ensure year is stored as int
                print(f"[DEBUG] Year in params: {year_val}, current year: {current_year}")
                # If year is old, no type is set, and query doesn't explicitly say "new"
                # As per user request, we no longer automatically set type=used here.
                # Type should only come from explicit user query terms or LLM (handled in Step 4).
                if year_val < current_year and not params.get('type') and 'new' not in query_words: # query_words was defined in Step 4
                    print(f"[DEBUG] Condition met for potential old year type inference (year: {year_val} < {current_year}, no type, 'new' not in query), but NOT setting type=used as per new rule.")
                    # params['type'] = 'used' # <--- This line is now removed/commented
        except ValueError:
            print(f"[DEBUG] Year '{params.get('year')}' is not a valid integer. Removing year.")
            if 'year' in params: params.pop('year')
        except Exception as e:
            print(f"[DEBUG] Exception processing year: {e}. Removing year.")
            if 'year' in params: params.pop('year')

    # Final cleanup of params: remove any keys with None, empty string/list/dict values
    # Also remove keys not in ALLOWED_PARAMS (though build_inventory_url also does this)
    final_params = {}
    for k, v in params.items():
        if k in ALLOWED_PARAMS and v is not None and v != '' and v != [] and v != {}: # Use ALLOWED_PARAMS from utils.py
            # Specific check for string "null" or "none" which might come from LLM
            if isinstance(v, str) and v.strip().lower() in ['null', 'none']:
                print(f"[DEBUG] Removing param '{k}' with value '{v}'")
                continue
            final_params[k] = v
    
    print("[DEBUG] Final processed params before returning:", final_params)
    return final_params

def build_inventory_url(base_url, params):
    GENERIC_TERMS = {"car", "cars", "vehicle", "vehicles"}
    filtered = {}
    for k, v in params.items():
        k_lower = k.lower()
        if k_lower not in ALLOWED_PARAMS: # Use ALLOWED_PARAMS from utils.py
            continue  # Skip any param not in the allowed list
        # Skip None, empty, or string \'null\'/\'none\' (case-insensitive)
        if v is None or v == "" or v == [] or v == {}:
            continue
        if isinstance(v, str) and v.strip().lower() in {"null", "none"}:
            continue
        if isinstance(v, dict):
            continue
        # Exclude generic values for any field
        def is_generic(val):
            if isinstance(val, str):
                return val.strip().lower() in GENERIC_TERMS or val.strip().lower() in {"null", "none"}
            if isinstance(val, list):
                return any(is_generic(item) for item in val)
            return False
        if is_generic(v):
            continue
        if isinstance(v, list):
            v = [item for item in v if item and not is_generic(item)]
            if len(v) == 1:
                v = v[0]
            elif len(v) > 1:
                v = ",".join(str(item) for item in v)
            else:
                continue
        
        # Convert current value to string and lowercase
        val_str_lower = str(v).lower()

        # If the key is 'model', replace spaces with hyphens
        if k_lower == 'model':
            val_str_lower = val_str_lower.replace(' ', '-')
        
        # Skip paymentmax if its value is 0
        if k_lower == 'paymentmax' and val_str_lower == '0':
            print(f"[DEBUG] build_inventory_url: Skipping '{k_lower}' because its value is 0.")
            continue
            
        filtered[k_lower] = val_str_lower
    return f"{base_url}?{urlencode(filtered)}"

@app.post("/api/search")
async def search(request: QueryRequest):
    params = extract_params(request.query)
    url = build_inventory_url("https://www.paragonhonda.com/inventory", params)
    return {"url": url, "params": params}
