import re
import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# LLM Configuration
llm = OllamaLLM(model="llama3.2")

# --- LLM Cache ---
# Simple in-memory cache. For a production system, consider a more robust
# caching solution like Redis or a library like cachetools with TTL.
llm_response_cache = {}
# --- End LLM Cache ---

prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "Extract the following fields from this vehicle search query, but ONLY include a field if it is explicitly mentioned in the query: "
        "year, make, model, trim, color, vehicle type, transmission, features, mileage, and type. "
        "For the 'type' field: if the query mentions 'certified' or 'cpo', the value for 'type' should be 'cpo'. If the query mentions 'used' or 'pre-owned', the value for 'type' should be 'used'. If the query mentions 'new', the value for 'type' should be 'new'. "
        "For the 'transmissions' field, if mentioned, the value should be one of: \"manual\", \"automatic\", or \"cvt\". If the query implies one of these but uses different wording (e.g., 'stick shift' for manual, 'auto' for automatic), map it to the correct value. Ensure that terms like \"cvt\" are placed in the 'transmissions' field and NOT in 'vehicletypes'. "
        "Also extract price information: if a price range like 'between X and Y' or 'X to Y' is given, populate 'paymentmin' with X and 'paymentmax' with Y. "
        "If only one price is mentioned (e.g., 'under X', 'around X', 'less than X', 'at most X', 'below X'), populate 'paymentmax' with X. "
        "If the query says 'over X', 'starting at X', 'more than X', 'at least X'), populate 'paymentmin' with X. "
        "Also extract mileage information: if a mileage range like 'between X and Y miles' or 'X to Y miles' is given, populate 'mileagemin' with X and 'mileagemax' with Y. "
        "If only one mileage is mentioned (e.g., 'under X miles', 'less than X miles', 'at most X miles', 'below X miles'), populate 'mileagemax' with X. "
        "If the query says 'over X miles', 'starting at X miles', 'more than X miles', 'at least X miles'), populate 'mileagemin' with X. "
        "Supported vehicle types for the 'vehicletypes' field are ONLY: convertible, coupe, suv, sedan, truck, van, wagon, hatchback, mpv. Do NOT put transmission types (e.g., cvt, manual, automatic) or any other terms in the 'vehicletypes' field. "
        "If the query uses generic terms like 'car', 'cars', 'vehicle', or 'vehicles', do not include any value for 'vehicletypes' unless a specific supported type from the list above is also mentioned. "
        "Do NOT guess or fill in any values that are not present in the query. "
        "Return ONLY a single, valid JSON object. The JSON object must be pure JSON and must NOT contain any comments, explanations, or any other non-JSON text within it or around it. "
        "Use these keys in the JSON: 'year', 'make', 'model', 'trim', 'color', 'vehicletypes' (for vehicle type), "
        "'transmissions' (for transmission), 'featuresubcategories' (for features), 'type', 'paymentmin', 'paymentmax', "
        "'mileagemin', 'mileagemax'. All keys and string values in the JSON should be lowercase. "
        "Query: {query}"
    ),
)

def get_llm_params_from_query(user_query: str) -> dict:
    """Helper function to call LLM and parse its JSON response, with caching."""
    
    # Check cache first
    if user_query in llm_response_cache:
        print(f"[DEBUG] Returning cached LLM response for query: {user_query}")
        return llm_response_cache[user_query]

    print(f"[DEBUG] Querying LLM (not cached): {user_query}")
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

