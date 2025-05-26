
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import json
import re

# Initialize the Ollama LLM (make sure your Llama 3.2 model is running in Ollama)
llm = OllamaLLM(model="llama3.2")

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "Extract the following fields from this vehicle search query: "
        "year, make, model, color, body style, and maximum price. "
        "Return as a JSON object with keys: year, make, model, color, bodystyle, paymentmax.\n"
        "Query: {query}"
    ),
)

def extract_params(user_query):
    formatted_prompt = prompt.format(query=user_query)
    response = llm.invoke(formatted_prompt)
    # Try to extract JSON from the response
    try:
        return json.loads(response)
    except Exception:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("Could not parse JSON from LLM response.")

from urllib.parse import urlencode

def build_inventory_url(base_url, params):
    # Only include params with non-None values
    filtered = {k: v for k, v in params.items() if v is not None}
    return f"{base_url}?{urlencode(filtered)}"

if __name__ == "__main__":
    user_query = "Show me blue SUVs from 2020 under $30000"
    params = extract_params(user_query)
    print("Extracted params:", params)
    url = build_inventory_url("https://domain.com", params)
    print("Inventory URL:", url)
