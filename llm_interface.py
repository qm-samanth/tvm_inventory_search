import re
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

def initialize_llm_with_model(model_name: str):
    """Initialize the LLM with a specific model name."""
    if model_name.startswith("gemini"):
        print(f"[DEBUG] Initializing {model_name}...")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key="AIzaSyBLUvd17J8wC8dcGLnIYue5jEZfyfMmsrs"
        )
    elif model_name.startswith("llama"):
        print(f"[DEBUG] Initializing {model_name}...")
        return OllamaLLM(model=model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: gemini-1.5-flash, llama3.2")

# --- LLM Cache ---
# Simple in-memory cache. For a production system, consider a more robust
# caching solution like Redis or a library like cachetools with TTL.
llm_response_cache = {}
# --- End LLM Cache ---

prompt = PromptTemplate(
    input_variables=["query"],
    template="""Extract the following fields from this vehicle search query, but ONLY include a field if it is explicitly mentioned in the query:
year, make, model, trim, color, vehicle type, transmission, features, mileagemin, mileagemax, paymentmin, paymentmax, type, and drivetrains.

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

4. FEATURESUBCATEGORIES FIELD:
   🚫 DO NOT include 'featuresubcategories' unless user specifically mentions features
   - For multi-word features, join with underscores
   - Examples: "apple_carplay", "android_auto", "adaptive_cruise_control", "parking_sensors"

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
- 🚫 CRITICAL: ONLY include mileagemin OR mileagemax when EXPLICITLY mentioned, NOT BOTH unless it's a range
- Range format ('between 30000 and 60000 miles'): populate BOTH 'mileagemin' and 'mileagemax'
- Upper limit ONLY ('under 50000 miles', 'below 100k miles'): populate 'mileagemax' ONLY, do NOT include mileagemin
- Lower limit ONLY ('over 20000 miles', 'more than 50k miles'): populate 'mileagemin' ONLY, do NOT include mileagemax
- 🚫 DO NOT guess or infer mileage ranges when only one bound is specified

EXAMPLES:
❌ WRONG: "under $30000" → mileagemax: 30000
✅ CORRECT: "under $30000" → paymentmax: 30000
❌ WRONG: "50000 miles" → paymentmax: 50000  
✅ CORRECT: "50000 miles" → mileagemax: 50000
❌ WRONG: "below 75000 miles" → mileagemin: 50000, mileagemax: 75000
✅ CORRECT: "below 75000 miles" → mileagemax: 75000 (ONLY)

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

def get_llm_params_from_query(user_query: str, model_name: str = None) -> dict:
    """Helper function to call LLM and parse its JSON response, with caching.
    
    Args:
        user_query: The user's search query
        model_name: Optional model to use. Defaults to "gemini-1.5-flash" if None.
                   Supported values: "gemini-1.5-flash", "llama3.2"
    """
    
    # Use specific model if provided, otherwise use default
    if model_name:
        try:
            current_llm = initialize_llm_with_model(model_name)
            provider = "gemini" if model_name.startswith("gemini") else "ollama"
            print(f"[DEBUG] Using specific model: {model_name}")
        except Exception as e:
            print(f"[DEBUG] Error initializing model {model_name}: {e}")
            # Fallback to default model
            model_name = "gemini-1.5-flash"
            current_llm = initialize_llm_with_model(model_name)
            provider = "gemini"
            print(f"[DEBUG] Falling back to default model: {model_name}")
    else:
        # Default to Gemini if no model specified
        model_name = "gemini-1.5-flash"
        current_llm = initialize_llm_with_model(model_name)
        provider = "gemini"
        print(f"[DEBUG] Using default model: {model_name}")
    
    llm_response_cache.clear() # TEMPORARY: Clears cache on every call for testing

    # Check cache first
    cache_key = f"{model_name}:{user_query}"
    if cache_key in llm_response_cache:
        print(f"[DEBUG] Returning cached LLM response for query: {user_query}")
        return llm_response_cache[cache_key]

    print(f"[DEBUG] Querying LLM with provider '{provider}' (not cached or cache cleared): {user_query}")
    try:
        formatted_prompt = prompt.format(query=user_query)
        response = current_llm.invoke(formatted_prompt)
        
        # Handle different response types based on provider
        if provider == "gemini":
            # ChatGoogleGenerativeAI returns a message object, extract the content
            response_text = response.content if hasattr(response, 'content') else str(response)
        elif provider == "ollama":
            # OllamaLLM returns a string directly
            response_text = response
        else:
            response_text = str(response)
            
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
    llm_response_cache[cache_key] = params
    return params

