# test_client.py
import requests
import json

# url = "http://localhost:8000/hackrx/run"
url = "https://hackrx-api.onrender.com/hackrx/run"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer 0a49a7d9c4a349a842f507e197933b12d6a776863c0dc5518de11e8747fdd506",
    "Content-Type": "application/json"
}

# --- THIS PAYLOAD IS NOW CORRECT ---
# "documents" is a single string to match your server code.
payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

print("Sending final request to the server...")

try:
    # Make the POST request with a longer timeout
    response = requests.post(url, headers=headers, json=payload, timeout=300) # 5 minute timeout
    
    print(f"\nStatus Code: {response.status_code}")
    print("\n--- FINAL SERVER RESPONSE ---")
    
    # Pretty print the final JSON output
    try:
        response_json = response.json()
        
        # Check if the response contains the 'answers' key and they are strings
        if 'answers' in response_json and all(isinstance(i, str) for i in response_json['answers']):
            answers_list = [json.loads(ans_str) for ans_str in response_json.get('answers', [])]
            final_output = {"answers": answers_list}
            print(json.dumps(final_output, indent=4))
        else:
            # If it's a validation error or other non-standard response
            print(json.dumps(response_json, indent=4))

    except json.JSONDecodeError:
        print("Could not parse JSON response:")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"\n--- REQUEST FAILED ---")
    print(f"An error occurred: {e}")
    print("Please make sure your Uvicorn server is running.")