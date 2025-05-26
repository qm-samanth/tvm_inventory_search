import datetime
ALLOWED_PARAMS = {
    'year', 'make', 'model', 'trim', 'color', 'vehicletypes', 'transmissions',
    'featuresubcategories', 'paymentmax', 'paymentmin', 'type',
    'mileagemin', 'mileagemax' # Added mileage params
}
SUPPORTED_TYPES = [
    'convertible', 'coupe', 'suv', 'sedan', 'truck', 'van', 'wagon', 'hatchback', 'mpv'
]
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from urllib.parse import urlencode
import json
import re

app = FastAPI()

# Allow CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

llm = OllamaLLM(model="llama3.2")

prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "Extract the following fields from this vehicle search query, but ONLY include a field if it is explicitly mentioned in the query: "
        "year, make, model, trim, color, vehicle type, transmission, features, mileage, and type (used/new/certified). " # Added mileage
        "Also extract price information: if a price range like 'between X and Y' or 'X to Y' is given, populate 'paymentmin' with X and 'paymentmax' with Y. "
        "If only one price is mentioned (e.g., 'under X', 'around X', 'less than X', 'at most X'), populate 'paymentmax' with X. "
        "If the query says 'over X', 'starting at X', 'more than X', 'at least X', populate 'paymentmin' with X. "
        "Also extract mileage information: if a mileage range like 'between X and Y miles' or 'X to Y miles' is given, populate 'mileagemin' with X and 'mileagemax' with Y. " # Mileage instruction
        "If only one mileage is mentioned (e.g., 'under X miles', 'less than X miles', 'at most X miles'), populate 'mileagemax' with X. " # Mileage instruction
        "If the query says 'over X miles', 'starting at X miles', 'more than X miles', 'at least X miles', populate 'mileagemin' with X. " # Mileage instruction
        "Supported vehicle types are: convertible, coupe, suv, sedan, truck, van, wagon, hatchback, mpv. "
        "If the query uses generic terms like 'car', 'cars', 'vehicle', or 'vehicles', do not include any value for 'vehicletypes' unless a specific supported type is also mentioned. "
        "Do NOT guess or fill in any values that are not present in the query. "
        "Return a JSON object with only the keys that were mentioned in the query, using lowercase for all keys and values. "
        "Use these keys in the JSON: 'year', 'make', 'model', 'trim', 'color', 'vehicletypes' (for vehicle type), "
        "'transmissions' (for transmission), 'featuresubcategories' (for features), 'type', 'paymentmin', 'paymentmax', "
        "'mileagemin', 'mileagemax'. " # Added mileage keys
        "Query: {query}"
    ),
)

def extract_params(user_query):
    # Step 1: Get initial params from LLM
    formatted_prompt = prompt.format(query=user_query)
    response = llm.invoke(formatted_prompt)
    try:
        # Attempt to parse the entire response as JSON
        params = json.loads(response)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from a string that might contain other text
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                params = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"[DEBUG] Failed to parse extracted JSON: {match.group(0)}")
                return {} # Return empty if LLM output is not parsable JSON
        else:
            print(f"[DEBUG] No JSON object found in LLM response: {response}")
            return {} # Return empty if no JSON object is found
    except Exception as e:
        print(f"[DEBUG] An unexpected error occurred during LLM response processing: {e}")
        return {}

    print("[DEBUG] Initial params from LLM:", params)
    user_query_lower = user_query.lower()
    print("[DEBUG] User query (lower):", user_query_lower)

    # --- Start of Price Logic Refinement ---

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
        # Check for null-like strings again after initial cleaning, though is_effectively_none_or_absent should catch most.
        if s in ["null", "none"]: return None

        multiplier = 1
        if s.endswith('k'): 
            multiplier = 1000; s = s[:-1].strip()
        elif s.endswith('m'): 
            multiplier = 1000000; s = s[:-1].strip()
        
        if not s or s in ["null", "none"]: return None # Check if s became empty or null-like after k/m processing
        
        try:
            return float(s) * multiplier
        except ValueError:
            print(f"[DEBUG] normalize_price_for_comparison: ValueError converting '{s}' to float.")
            return None

    upper_bound_keywords = ["under ", "less than ", "at most ", "maximum ", " up to "]
    query_has_upper_bound_keyword = any(keyword in user_query_lower for keyword in upper_bound_keywords)

    lower_bound_keywords = ["over ", "starting at ", "more than ", "at least ", "minimum "]
    query_has_lower_bound_keyword = any(keyword in user_query_lower for keyword in lower_bound_keywords)

    llm_paymentmin = params.get("paymentmin")
    llm_paymentmax = params.get("paymentmax")

    # 1. Correction: If query implies "under X" (upper bound) but LLM provides only paymentmin (or paymentmax is effectively absent)
    if query_has_upper_bound_keyword and not query_has_lower_bound_keyword:
        if not is_effectively_none_or_absent(llm_paymentmin) and is_effectively_none_or_absent(llm_paymentmax):
            print(f"[DEBUG] Query implies 'under X'. LLM provided paymentmin ('{llm_paymentmin}') but paymentmax is effectively absent/null ('{llm_paymentmax}'). Swapping paymentmin to paymentmax.")
            params["paymentmax"] = params.pop("paymentmin")
            llm_paymentmax = params.get("paymentmax") # Update local var
            llm_paymentmin = None # paymentmin was popped
            if "paymentmin" in params: params.pop("paymentmin") # Ensure it is gone

    # 2. Correction: If query implies "over X" (lower bound) but LLM provides only paymentmax (or paymentmin is effectively absent)
    elif query_has_lower_bound_keyword and not query_has_upper_bound_keyword:
         if not is_effectively_none_or_absent(llm_paymentmax) and is_effectively_none_or_absent(llm_paymentmin):
            print(f"[DEBUG] Query implies 'over X'. LLM provided paymentmax ('{llm_paymentmax}') but paymentmin is effectively absent/null ('{llm_paymentmin}'). Swapping paymentmax to paymentmin.")
            params["paymentmin"] = params.pop("paymentmax")
            llm_paymentmin = params.get("paymentmin") # Update local var
            llm_paymentmax = None # paymentmax was popped
            if "paymentmax" in params: params.pop("paymentmax") # Ensure it is gone

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

    # Keywords for stripping from values if LLM includes them
    mileage_value_strip_keywords = [
        "under ", "less than ", "at most ", "maximum ", " up to ", # Upper bounds
        "over ", "starting at ", "more than ", "at least ", "minimum "  # Lower bounds
    ]

    def strip_mileage_keywords_from_value(val_str_input):
        s = val_str_input.lower().strip() # Work with lowercase
        for keyword in mileage_value_strip_keywords:
            if s.startswith(keyword):
                s = s[len(keyword):].strip()
                break # Remove only one instance of a prefix
        return s

    def normalize_mileage_for_internal_comparison(val_str_input):
        if is_effectively_none_or_absent(val_str_input):
            return None
        
        s = str(val_str_input) # Don't lowercase here, strip_mileage_keywords_from_value will
        s = strip_mileage_keywords_from_value(s) # Strip prefixes like "less than "
        s = s.lower().strip().replace(",", "") # Now lowercase and clean commas

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

    upper_bound_mileage_keywords = ["under ", "less than ", "at most ", "maximum ", " up to "] # Assuming "miles" is handled by LLM or normalization
    query_has_upper_bound_mileage_keyword = any(keyword + "miles" in user_query_lower or keyword[:-1] + " miles" in user_query_lower or keyword + "mileage" in user_query_lower for keyword in upper_bound_mileage_keywords) or any(user_query_lower.startswith(keyword + "miles") or user_query_lower.startswith(keyword[:-1] + " miles") for keyword in upper_bound_mileage_keywords)


    lower_bound_mileage_keywords = ["over ", "starting at ", "more than ", "at least ", "minimum "]
    query_has_lower_bound_mileage_keyword = any(keyword + "miles" in user_query_lower or keyword[:-1] + " miles" in user_query_lower or keyword + "mileage" in user_query_lower for keyword in lower_bound_mileage_keywords) or any(user_query_lower.startswith(keyword + "miles") or user_query_lower.startswith(keyword[:-1] + " miles") for keyword in lower_bound_mileage_keywords)


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

    # --- Start of Cross-Contamination/Collision Check (NEW) ---
    # This runs after initial price & mileage keyword logic, but before defaulting min to 0 and final numeric normalization.

    price_keywords_pattern = r"(\$|\€|\£|\¥|\₹|dollar|euro|pound|yen|rupee|aud|price|cost|budget|payment)"
    query_has_strong_price_keywords = bool(re.search(price_keywords_pattern, user_query_lower))

    # query_has_upper_bound_mileage_keyword and query_has_lower_bound_mileage_keyword are already defined
    # We also need a general check for any mileage keyword if those aren't set
    generic_mileage_keywords_pattern = r"(mile|mileage|km|kilometer)"
    query_has_generic_mileage_keywords = bool(re.search(generic_mileage_keywords_pattern, user_query_lower))
    query_context_is_mileage = query_has_upper_bound_mileage_keyword or query_has_lower_bound_mileage_keyword or query_has_generic_mileage_keywords

    def normalize_value_for_collision_check(val_str_input):
        if is_effectively_none_or_absent(val_str_input):
            return None
        
        s = str(val_str_input) # Original value
        # Try to strip prefixes that mileage logic's normalize_mileage_for_internal_comparison would strip
        # This is to make the comparison fair if one value has it and other doesn't pre-normalization
        s_temp_mileage_norm = strip_mileage_keywords_from_value(s) # strip "less than" etc.
        s_temp_mileage_norm = s_temp_mileage_norm.lower().replace(",", "").replace("miles", "").strip()
        
        # Also try to strip price related symbols for a more raw number comparison
        s_temp_price_norm = s.lower().replace("$","").replace(",","").strip()

        # If mileage normalization changed it significantly (e.g. removed "less than X miles") use that, else use price norm
        # This is a heuristic. The goal is to get to the core number.
        if len(s_temp_mileage_norm) < len(s_temp_price_norm) and s_temp_mileage_norm.replace('.','',1).isdigit():
            s_for_num = s_temp_mileage_norm
        else:
            s_for_num = s_temp_price_norm
        
        # Handle 'k' or 'm' if present at the end of s_for_num
        multiplier = 1
        if s_for_num.endswith('k'):
            multiplier = 1000; s_for_num = s_for_num[:-1].strip()
        elif s_for_num.endswith('m'):
            multiplier = 1000000; s_for_num = s_for_num[:-1].strip()

        if not s_for_num or not s_for_num.replace('.','',1).isdigit(): # check if it can be a number
            # Fallback if the above heuristic didn't yield a clean number string
            # Try the mileage normalizer directly as it's more comprehensive for stripping text
            num_val = normalize_mileage_for_internal_comparison(str(val_str_input)) # This returns a float or None
            return num_val

        try:
            return float(s_for_num) * multiplier
        except ValueError:
            print(f"[DEBUG] normalize_value_for_collision_check: ValueError converting '{s_for_num}' (original: '{val_str_input}')")
            return None

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
    
    # Extract all whole words from the query for accurate keyword matching
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

    if explicit_type_from_query:
        params['type'] = explicit_type_from_query
        print(f"[DEBUG] Final type set from explicit query mention: {params['type']}")
    else:
        # If no explicit type (new, used, cpo) was found in the user's query text,
        # remove any 'type' the LLM might have provided. The prompt instructs
        # the LLM to only include explicitly mentioned fields. If the LLM still provides it,
        # we override that here for the 'type' field to ensure adherence to the rule.
        # The year-based logic in Step 7 can later infer 'used' if applicable.
        if 'type' in params:
            llm_provided_type_val = params.get('type')
            print(f"[DEBUG] Removing LLM-provided 'type': '{llm_provided_type_val}' because no explicit type (new/used/cpo) was found in the user query text.")
            params.pop('type')
        else:
            print(f"[DEBUG] No explicit type in query, and LLM did not provide 'type'. 'type' remains unset before year logic.")

    # Step 6: Validate and filter vehicletypes
    # Only populate vehicletypes if a supported type is EXPLICITLY mentioned in the query.
    if 'vehicletypes' in params: # Check if LLM even provided it as a starting point
        llm_provided_vehicletype = params.get('vehicletypes') # Store for logging
        print(f"[DEBUG] LLM initially provided vehicletypes: {llm_provided_vehicletype}")
        params.pop('vehicletypes') # Remove it first, we will re-add ONLY if explicit

    explicitly_mentioned_supported_type = None
    for vt in SUPPORTED_TYPES:
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
        if k in ALLOWED_PARAMS and v is not None and v != '' and v != [] and v != {}:
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
        if k_lower not in ALLOWED_PARAMS:
            continue  # Skip any param not in the allowed list
        # Skip None, empty, or string 'null'/'none' (case-insensitive)
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
        filtered[k_lower] = str(v).lower()
    return f"{base_url}?{urlencode(filtered)}"

@app.post("/api/search")
async def search(request: QueryRequest):
    params = extract_params(request.query)
    url = build_inventory_url("https://www.paragonhonda.com/inventory", params)
    return {"url": url, "params": params}
