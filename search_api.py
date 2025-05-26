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
        "year, make, model, trim, color, vehicle type, transmission, features, maximum price, and type (used/new/certified). "
        "Supported vehicle types are: convertible, coupe, suv, sedan, truck, van, wagon, hatchback, mpv. "
        "If the query uses generic terms like 'car', 'cars', or 'vehicle', do not include any value for 'vehicletypes'. "
        "Do NOT guess or fill in any values that are not present in the query. "
        "Return a JSON object with only the keys that were mentioned in the query, using lowercase for all keys and values. "
        "For vehicle type, use the key 'vehicletypes' in the JSON. "
        "For transmission, use the key 'transmissions' in the JSON. "
        "For features, use the key 'featuresubcategories' in the JSON. "
        "Query: {query}"
    ),
)

def extract_params(user_query):
   

    formatted_prompt = prompt.format(query=user_query)
    response = llm.invoke(formatted_prompt)
    try:
        params = json.loads(response)
    except Exception:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            params = json.loads(match.group(0))
        else:
            # If parsing fails, return empty params or raise
            return {}

    print("[DEBUG] Extracted params from LLM:", params)
    user_query_lower = user_query.lower()
    print("[DEBUG] User query (lower):", user_query_lower)

    # Robustly detect any form of 'pre-owned' or 'used' in user query (case-insensitive, dash/space/none)
    preowned_pattern = r"pre[-\s]?owned|preowned|used"
    certified_preowned_pattern = r"certified[ -]?(pre[-\s]?owned|preowned)"
    preowned_match = re.search(preowned_pattern, user_query_lower, re.IGNORECASE)
    certified_preowned_match = re.search(certified_preowned_pattern, user_query_lower, re.IGNORECASE)
    print("[DEBUG] Preowned/used pattern match in user query:", preowned_match)
    print("[DEBUG] Certified pre-owned pattern match in user query:", certified_preowned_match)
    if certified_preowned_match or 'cpo' in user_query_lower:
        print("[DEBUG] Setting type to 'cpo' due to certified pre-owned/cpo in user query.")
        params['type'] = 'cpo'
    elif preowned_match:
        print("[DEBUG] Setting type to 'used' due to user query match.")
        params['type'] = 'used'
    elif 'type' in params and params['type']:
        type_val = str(params['type']).strip().lower()
        print("[DEBUG] LLM output type value:", type_val)
        type_match = re.fullmatch(preowned_pattern, type_val, re.IGNORECASE)
        certified_type_match = re.fullmatch(certified_preowned_pattern, type_val, re.IGNORECASE)
        print("[DEBUG] Preowned/used pattern match in LLM output type:", type_match)
        print("[DEBUG] Certified pre-owned pattern match in LLM output type:", certified_type_match)
        if certified_type_match or type_val == 'cpo':
            print("[DEBUG] Setting type to 'cpo' due to LLM output type match.")
            params['type'] = 'cpo'
        elif type_match:
            print("[DEBUG] Setting type to 'used' due to LLM output type match.")
            params['type'] = 'used'
        elif 'certified' in user_query_lower:
            print("[DEBUG] Setting type to 'cpo' due to 'certified' in user query.")
            params['type'] = 'cpo'
        elif 'new' in user_query_lower:
            print("[DEBUG] Setting type to 'new' due to 'new' in user query.")
            params['type'] = 'new'
        elif 'used' in user_query_lower:
            print("[DEBUG] Setting type to 'used' due to 'used' in user query.")
            params['type'] = 'used'
        else:
            print("[DEBUG] Removing type from params due to no match.")
            params.pop('type')

    # Map maximum price to paymentmax and set paymentmin=0 if only max is present
    if 'maximum price' in params:
        params['paymentmax'] = params['maximum price']
        params['paymentmin'] = 0
        params.pop('maximum price')
    elif 'price' in params:
        params['paymentmax'] = params['price']
        params['paymentmin'] = 0
        params.pop('price')

    # Only include vehicletypes if a supported type is explicitly mentioned as a whole word in the user query
    if 'vehicletypes' in params:
        vt_value = params['vehicletypes']
        mentioned = False
        for vt in SUPPORTED_TYPES:
            vt_match = re.search(rf'\b{re.escape(vt)}\b', user_query_lower)
            print(f"[DEBUG] Checking vehicle type '{vt}': match={vt_match}")
            if vt_match:
                mentioned = True
                break
        if not mentioned:
            print("[DEBUG] Removing vehicletypes from params: not explicitly mentioned in user query.")
            params.pop('vehicletypes')
        else:
            # If it's a list, check all values; if string, check directly
            if isinstance(vt_value, list):
                vt_value = [v for v in vt_value if v in SUPPORTED_TYPES]
                print(f"[DEBUG] Filtered vehicletypes list: {vt_value}")
                if vt_value:
                    params['vehicletypes'] = vt_value[0] if len(vt_value) == 1 else ",".join(vt_value)
                else:
                    print("[DEBUG] Removing vehicletypes from params: no supported types in list.")
                    params.pop('vehicletypes')
            elif vt_value not in SUPPORTED_TYPES:
                print(f"[DEBUG] Removing vehicletypes from params: '{vt_value}' not supported.")
                params.pop('vehicletypes')

    # If year is present, less than current year, and no type, set type=used
    current_year = datetime.datetime.now().year
    if 'year' in params:
        try:
            year_val = int(str(params['year']).strip())
            print(f"[DEBUG] Year in params: {year_val}, current year: {current_year}")
            if year_val < current_year and 'type' not in params:
                print("[DEBUG] Setting type to 'used' due to year < current year and no type present.")
                params['type'] = 'used'
        except Exception as e:
            print(f"[DEBUG] Exception parsing year: {e}")
    # (Removed duplicate/stray return params outside function)
    return params

def build_inventory_url(base_url, params):
    GENERIC_TERMS = {"car", "cars", "vehicle", "vehicles"}
    filtered = {}
    for k, v in params.items():
        k_lower = k.lower()
        if k_lower not in ALLOWED_PARAMS:
            continue  # Skip any param not in the allowed list
        if v is None or v == "" or v == [] or v == {}:
            continue
        if isinstance(v, dict):
            continue
        # Exclude generic values for any field
        def is_generic(val):
            if isinstance(val, str):
                return val.strip().lower() in GENERIC_TERMS
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
