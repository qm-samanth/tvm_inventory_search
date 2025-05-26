import datetime
ALLOWED_PARAMS = {
    'year', 'make', 'model', 'trim', 'color', 'vehicletypes', 'transmissions',
    'featuresubcategories', 'paymentmax', 'paymentmin', 'type'
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
        "year, make, model, trim, color, vehicle type, transmission, features, and type (used/new/certified). "
        "Also extract price information: if a price range like 'between X and Y' or 'X to Y' is given, populate 'paymentmin' with X and 'paymentmax' with Y. "
        "If only one price is mentioned (e.g., 'under X', 'around X', 'less than X', 'at most X'), populate 'paymentmax' with X. "
        "If the query says 'over X', 'starting at X', 'more than X', 'at least X', populate 'paymentmin' with X. "
        "Supported vehicle types are: convertible, coupe, suv, sedan, truck, van, wagon, hatchback, mpv. "
        "If the query uses generic terms like 'car', 'cars', 'vehicle', or 'vehicles', do not include any value for 'vehicletypes' unless a specific supported type is also mentioned. "
        "Do NOT guess or fill in any values that are not present in the query. "
        "Return a JSON object with only the keys that were mentioned in the query, using lowercase for all keys and values. "
        "Use these keys in the JSON: 'year', 'make', 'model', 'trim', 'color', 'vehicletypes' (for vehicle type), "
        "'transmissions' (for transmission), 'featuresubcategories' (for features), 'type', 'paymentmin', 'paymentmax'. "
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

    # Step 2: Correct LLM confusion: if model is a supported vehicle type, move to vehicletypes
    if 'model' in params and params['model']:
        model_val = str(params['model']).strip().lower()
        if model_val in SUPPORTED_TYPES:
            print(f"[DEBUG] Moving model '{model_val}' to vehicletypes due to LLM confusion.")
            if 'vehicletypes' not in params or not params['vehicletypes']: # Only overwrite if not already set or empty
                params['vehicletypes'] = model_val
            params.pop('model')

    # Step 5: Price field handling. LLM should provide 'paymentmin' and 'paymentmax' directly from prompt.
    # Minimal fallbacks for older LLM behavior or misinterpretations:
    if 'maximum price' in params and is_effectively_none_or_absent(params.get('paymentmax')): 
        val = params.pop('maximum price')
        if not is_effectively_none_or_absent(val):
            params['paymentmax'] = val
            print(f"[DEBUG] Fallback: Mapped 'maximum price' to 'paymentmax'")

    if 'price' in params: 
        price_val = params.get('price')
        if not is_effectively_none_or_absent(price_val):
            # Only use 'price' if both paymentmin and paymentmax are still effectively absent after refinements
            if is_effectively_none_or_absent(params.get('paymentmin')) and is_effectively_none_or_absent(params.get('paymentmax')):
                if query_has_lower_bound_keyword and not query_has_upper_bound_keyword:
                    params['paymentmin'] = price_val
                    print(f"[DEBUG] Fallback: Assuming 'price' field ('{price_val}') is for 'paymentmin' due to lower-bound keywords.")
                else: 
                    params['paymentmax'] = price_val
                    print(f"[DEBUG] Fallback: Assuming 'price' field ('{price_val}') is for 'paymentmax'.")
        # Always pop 'price' after considering it
        if 'price' in params: params.pop('price')

    # Default paymentmin to 0 if paymentmax is present (and not effectively absent) 
    # and paymentmin is effectively absent (or was unset by prior logic).
    if not is_effectively_none_or_absent(params.get('paymentmax')) and is_effectively_none_or_absent(params.get('paymentmin')):
        params['paymentmin'] = 0 
        print(f"[DEBUG] Defaulted paymentmin to 0 as paymentmax is present and paymentmin was effectively absent.")

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
