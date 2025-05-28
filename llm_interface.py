import re
import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# LLM Configuration
llm = OllamaLLM(model="llama3.2")

# Print which LLM model we're using
print(f"[DEBUG] Initialized LLM with model: {llm.model}")

# --- LLM Cache ---
# Simple in-memory cache. For a production system, consider a more robust
# caching solution like Redis or a library like cachetools with TTL.
llm_response_cache = {}
# --- End LLM Cache ---

prompt = PromptTemplate(
    input_variables=["query"],
    template="""Extract the following fields from this vehicle search query, but ONLY include a field if it is explicitly mentioned in the query:
year, make, model, trim, color, vehicle type, transmission, features, mileage, type, and drivetrains.

FIELD-SPECIFIC INSTRUCTIONS:

1. TYPE FIELD:
   - If query mentions 'certified' or 'cpo': value should be 'cpo'
   - If query mentions 'used' or 'pre-owned': value should be 'used'
   - If query mentions 'new': value should be 'new'

2. TRANSMISSIONS FIELD:
   🚫 DO NOT include 'transmissions' unless user specifically mentions transmission type
   - Allowed values: "manual", "automatic", or "cvt"
   - Map alternative terms: 'stick shift' → manual, 'auto' → automatic
   - Special rule: If user mentions 'cvt' OR 'automatic', return "cvt,automatic" (both values)
   - Only return "manual" by itself when manual transmission is specifically mentioned
   - Ensure CVT goes in 'transmissions' field, NOT 'vehicletypes'

3. DRIVETRAINS FIELD:
   🚫 DO NOT include 'drivetrains' unless user specifically mentions drivetrain type
   - Allowed values: "4wd", "awd", "2wd", "fwd", or "rwd"
   - Map alternative terms:
     • 'four-wheel drive' → 4wd
     • 'all-wheel drive' → awd
     • 'front-wheel drive' → fwd
     • 'rear-wheel drive' → rwd
     • 'two-wheel drive' → 2wd
   - All values should be lowercase

4. FEATURES FIELD:
   - For multi-word features, join with underscores
   - Examples: "adaptive_cruise_control", "parking_sensors"

5. VEHICLE TYPES FIELD:
   - ONLY allowed values: convertible, coupe, suv, sedan, truck, van, wagon, hatchback, mpv
   - Do NOT put transmission types (cvt, manual, automatic) in vehicletypes
   - Generic terms like 'car', 'cars', 'vehicle', 'vehicles' should NOT be included
   - Only include if a specific supported type is mentioned

PRICE AND MILEAGE EXTRACTION:

🚨 CRITICAL: DOLLAR AMOUNTS ($) ARE ALWAYS PRICE, NOT MILEAGE!

Price Information (contains $, money terms, or payment context):
- Dollar signs ($): "$30000" → paymentmax: 30000
- Money terms: "30k budget", "budget of 25000" → payment fields
- Payment context: "monthly payment", "financing", "lease"
- Range format ('between $X and $Y'): populate 'paymentmin' with X and 'paymentmax' with Y
- Upper limit ('under $30000', 'below $25k'): populate 'paymentmax' with number only
- Lower limit ('over $20000', 'starting at $15k'): populate 'paymentmin' with number only

Mileage Information (contains miles/mileage terms, NOT dollar signs):
- Must explicitly mention "miles", "mileage", "km", or "kilometers"
- Examples: "under 50000 miles", "low mileage", "100k miles"
- Range format ('between 30000 and 60000 miles'): populate 'mileagemin' and 'mileagemax'
- Upper limit ('under 50000 miles', 'less than 100k miles'): populate 'mileagemax' with number only
- Lower limit ('over 20000 miles', 'more than 50k miles'): populate 'mileagemin' with number only

EXAMPLES:
❌ WRONG: "under $30000" → mileagemax: 30000
✅ CORRECT: "under $30000" → paymentmax: 30000
❌ WRONG: "50000 miles" → paymentmax: 50000  
✅ CORRECT: "50000 miles" → mileagemax: 50000

IMPORTANT RULES:
🚫 CRITICAL: Do NOT include 'transmissions' field unless user specifically mentions transmission (manual, automatic, CVT, etc.)
🚫 CRITICAL: Do NOT include 'drivetrains' field unless user specifically mentions drivetrain (AWD, FWD, 4WD, etc.)
🚫 CRITICAL: Do NOT include 'vehicletypes' field unless user specifically mentions vehicle type (sedan, SUV, truck, etc.)
- Do NOT guess or fill in any values that are not present in the query
- Return ONLY a single, valid JSON object
- NO comments, explanations, or non-JSON text within or around the JSON
- Use these keys ONLY when explicitly mentioned: 'year', 'make', 'model', 'trim', 'color', 'vehicletypes', 'transmissions', 'featuresubcategories', 'type', 'paymentmin', 'paymentmax', 'mileagemin', 'mileagemax', 'drivetrains'
- All keys and string values in the JSON should be lowercase

Query: {query}""",
)

def get_llm_params_from_query(user_query: str) -> dict:
    """Helper function to call LLM and parse its JSON response, with caching."""
    
    llm_response_cache.clear() # TEMPORARY: Clears cache on every call for testing

    # Check cache first
    # Given the line above, this cache check will effectively always be a cache miss now,
    # but the structure is kept for when the clear() line is removed.
    if user_query in llm_response_cache:
        print(f"[DEBUG] Returning cached LLM response for query: {user_query}")
        return llm_response_cache[user_query]

    print(f"[DEBUG] Querying LLM model '{llm.model}' (not cached or cache cleared): {user_query}")
    try:
        formatted_prompt = prompt.format(query=user_query)
        response_text = llm.invoke(formatted_prompt)
    except Exception as e:
        print(f"[DEBUG] Error during LLM invocation: {e}")
        return {}

    try:
        params = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r'{.*}', response_text, re.DOTALL)
        if match:
            try:
                params = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"[DEBUG] Failed to parse extracted JSON: {match.group(0)}")
                return {}
        else:
            print(f"[DEBUG] No JSON object found in LLM response: {response_text}")
            return {}
    except Exception as e:
        print(f"[DEBUG] An unexpected error occurred during LLM response processing: {e}")
        return {}
    
    # Store in cache
    llm_response_cache[user_query] = params
    return params

