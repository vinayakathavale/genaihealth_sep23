import requests
from fastapi import FastAPI
import uvicorn
from app.models import MsgInput, MsgOutput
# Define the API endpoint
api_endpoint = 'https://graph.facebook.com/v17.0/144367702085285/messages'

# Define the Bearer token
bearer_token = 'EAAJFM3ZBzpZAQBO5NEh7tNGDb8DcIGQhPz9ME3qsN1O2oOigXJZCPq64qr5MIjZCjZCGGnn4xGjlCg42wItZC0D6QcZCkLF3Qj6JNV61bijW1jxoufZB5UgUn2DTkm5aaFga4CcaZArhWq89mM4etJYQZAIrUL1eGgHwR0FbfMZBWuHYBTRZBtkxZBlaoqbj1nvPEOgcGH2G2pNsll8ssiVCCdyEZD'

# Define the message data in JSON format
# message_data = {
#     "messaging_product": "whatsapp",
#     "to": "447471331661",  # Replace with the recipient's phone number
#     "type": "template",
#     "template": {
#         "name": "hello_world",
#         "language": {
#             "code": "en_US"
#         }
#     }
# }


message_data = {
    "messaging_product": "whatsapp",
    "to": "447471331661",  # Replace with the recipient's phone number
    "type": "text",  # Specify the message type as "text"
    "text": { "body": "Hello, it's me 3", "preview_url": False},  # Replace with your custom message
}

# Define the headers, including the Bearer token and Content-Type
headers = {
    'Authorization': f'Bearer {bearer_token}',
    'Content-Type': 'application/json',
}


app = FastAPI()


# def send_message():
#     try:
#         response = requests.post(api_endpoint, json=message_data, headers=headers)

#         # Check if the request was successful (HTTP status code 200)
#         if response.status_code == 200:
#             print('Message sent successfully!')
#             print('Response:', response.json())

#         else:
#             print(f'Request failed with status code: {response.status_code}')
#             print('Response content:', response.text)

#     except requests.exceptions.RequestException as e:
#         print(f'Request failed: {str(e)}')
#     except Exception as e:
#         print(f'An error occurred: {str(e)}')


@app.post("/receive_message")
def read_root(request: MsgInput):
    print(f"received {request.text}")
    return MsgOutput(text=request.text)



if __name__=="__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)