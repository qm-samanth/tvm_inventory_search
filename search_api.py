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
    if 'maximum price' in params and 'paymentmax' not in params: # If LLM used old key
        params['paymentmax'] = params.pop('maximum price')
        print(f"[DEBUG] Fallback: Mapped 'maximum price' to 'paymentmax'")

    if 'price' in params: # If LLM used generic 'price' key
        # This is a basic fallback. If 'price' exists and paymentmin/max are not set by LLM,
        # we assume 'price' might be a single max value.
        # Complex range parsing from 'price' is best handled by LLM via the new prompt.
        if 'paymentmax' not in params and 'paymentmin' not in params:
            params['paymentmax'] = params['price'] # Value will be cleaned in Step 5.1
            print(f"[DEBUG] Fallback: Assuming 'price' field ('{params.get('paymentmax')}') is for 'paymentmax'")
        # Remove 'price' as it's either used or superseded by direct paymentmin/max
        if 'price' in params: # check again as it might have been popped if used for paymentmax
             params.pop('price')

    # Default paymentmin to 0 if paymentmax is present and paymentmin was not set by LLM/fallback
    if 'paymentmax' in params and params.get('paymentmin') is None: # Check for None, as 0 is a valid value
        params['paymentmin'] = 0 # Default to 0
        print(f"[DEBUG] Defaulted paymentmin to 0 as paymentmax is present and paymentmin was not set.")

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
    
    # Step 4: Type detection and normalization (used, cpo, new) - (Moved Step 3 logic here, re-numbering for clarity)
    preowned_pattern = r"pre[-\s]?owned|preowned|used"
    certified_preowned_pattern = r"certified[ -]?(pre[-\s]?owned|preowned)"
    preowned_match = re.search(preowned_pattern, user_query_lower, re.IGNORECASE)
    certified_preowned_match = re.search(certified_preowned_pattern, user_query_lower, re.IGNORECASE)
    cpo_keyword_match = 'cpo' in user_query_lower.split() # Check for 'cpo' as a whole word

    print(f"[DEBUG] User query lower: '{user_query_lower}'")
    print(f"[DEBUG] Preowned/used pattern match in user query: {preowned_match}")
    print(f"[DEBUG] Certified pre-owned pattern match in user query: {certified_preowned_match}")
    print(f"[DEBUG] CPO keyword match in user query: {cpo_keyword_match}")

    explicit_type_from_query = None
    if certified_preowned_match or cpo_keyword_match:
        explicit_type_from_query = 'cpo'
        print(f"[DEBUG] Type set to '{explicit_type_from_query}' from query (certified/cpo).")
    elif preowned_match:
        explicit_type_from_query = 'used'
        print(f"[DEBUG] Type set to '{explicit_type_from_query}' from query (preowned/used).")
    elif 'new' in user_query_lower.split(): # Check for 'new' as a whole word
        explicit_type_from_query = 'new'
        print(f"[DEBUG] Type set to '{explicit_type_from_query}' from query ('new').")

    llm_type_val = str(params.get('type', '')).strip().lower()
    print(f"[DEBUG] LLM output for type: '{llm_type_val}'")

    if explicit_type_from_query:
        params['type'] = explicit_type_from_query
        print(f"[DEBUG] Final type set from query: {params['type']}")
    elif llm_type_val:
        # Normalize LLM output if it wasn't overridden by query
        if re.fullmatch(certified_preowned_pattern, llm_type_val, re.IGNORECASE) or llm_type_val == 'cpo':
            params['type'] = 'cpo'
            print(f"[DEBUG] Final type set from LLM (normalized to cpo): {params['type']}")
        elif re.fullmatch(preowned_pattern, llm_type_val, re.IGNORECASE):
            params['type'] = 'used'
            print(f"[DEBUG] Final type set from LLM (normalized to used): {params['type']}")
        elif llm_type_val == 'new':
            params['type'] = 'new'
            print(f"[DEBUG] Final type set from LLM (new): {params['type']}")
        else:
            # If LLM type is not recognized and not set by query, remove it
            if 'type' in params:
                print(f"[DEBUG] Removing unrecognized LLM type: {params['type']}")
                params.pop('type')
    elif 'type' in params: # If no explicit query type and no valid LLM type, remove
        print(f"[DEBUG] Removing type from params as it was not explicitly in query nor a valid LLM output.")
        params.pop('type')

    # Step 6: Validate and filter vehicletypes
    if 'vehicletypes' in params:
        vt_value_from_llm = params['vehicletypes']
        print(f"[DEBUG] LLM vehicletypes: {vt_value_from_llm}")
        explicitly_mentioned_supported_type = None

        # Check if any supported type is mentioned as a whole word in the query
        for vt in SUPPORTED_TYPES:
            if re.search(rf'\b{re.escape(vt)}\b', user_query_lower):
                explicitly_mentioned_supported_type = vt
                print(f"[DEBUG] Found explicitly mentioned supported type in query: '{vt}'")
                break # Take the first one found for simplicity, or decide on a strategy for multiple
        
        if explicitly_mentioned_supported_type:
            # If a supported type is explicitly mentioned in the query, use that.
            # This overrides whatever the LLM might have put in vehicletypes if it was generic or incorrect.
            params['vehicletypes'] = explicitly_mentioned_supported_type
            print(f"[DEBUG] Setting vehicletypes to explicitly mentioned type: '{params['vehicletypes']}'")
        elif vt_value_from_llm:
            # If no specific supported type was in the query, then validate LLM's output
            current_vt_values = []
            if isinstance(vt_value_from_llm, str):
                current_vt_values = [v.strip().lower() for v in vt_value_from_llm.split(',')]
            elif isinstance(vt_value_from_llm, list):
                current_vt_values = [str(v).strip().lower() for v in vt_value_from_llm]

            supported_and_mentioned_in_llm = [v for v in current_vt_values if v in SUPPORTED_TYPES]

            if supported_and_mentioned_in_llm:
                # If LLM provided valid types, use them (prefer single, then comma-separated)
                if len(supported_and_mentioned_in_llm) == 1:
                    params['vehicletypes'] = supported_and_mentioned_in_llm[0]
                else:
                    params['vehicletypes'] = ",".join(supported_and_mentioned_in_llm)
                print(f"[DEBUG] Validated LLM vehicletypes: {params['vehicletypes']}")
            else:
                # If LLM output is not a supported type or empty after filtering, remove it.
                params.pop('vehicletypes')
                print(f"[DEBUG] Removing vehicletypes: LLM output '{vt_value_from_llm}' not in supported list or not explicitly mentioned.")
        else:
            # If 'vehicletypes' was present but empty from LLM and nothing explicit in query, remove.
            params.pop('vehicletypes')
            print("[DEBUG] Removing empty vehicletypes from params.")

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
                if year_val < current_year and not params.get('type') and 'new' not in user_query_lower.split():
                    print("[DEBUG] Setting type to 'used' due to year < current year and no type/new mentioned.")
                    params['type'] = 'used'
        except ValueError:
            print(f"[DEBUG] Year '{params['year']}' is not a valid integer. Removing year.")
            params.pop('year')
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
