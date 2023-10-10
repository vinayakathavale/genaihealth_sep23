import requests, uvicorn, json
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.middleware.cors import CORSMiddleware

import ml_stuff

from pydantic import BaseModel


import openai
openai.api_key = ""


print(openai.api_key)

vectorstore = ml_stuff.preprocess_index_db()


bot_app = FastAPI()


origins = [
    "*",
]

bot_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def formatOutput(Body, data):
    
    if 'error' not in data:
        data_output = json.dumps(data['nutritions'])
        data_output = data_output.replace(" ", "") #remove whitespaces
    else:
        data_output = json.dumps(data)
        data_output = data_output.replace(":", "\n") #create paragraphs

    data_output = data_output.replace('"', '') #get only the nutrition information
    data_output = data_output.replace(",", "\n") #create paragraphs
    data_output = data_output.replace("{","").replace("}", "") #remove brackets
    
    data_output = Body +"\n \n" + data_output #add the header

    return data_output




def getInfoFruit(fruit_name):
    fruit_name = fruit_name.lower()
    url = 'https://fruityvice.com/api/fruit/{}'.format(fruit_name)
    resp = requests.get(url)
    if resp.status_code != 200:
        return {"error": "not a fruit"}
    data = resp.json()
    
    return data 

@bot_app.post("/bot")
async def chat(Body: str = Form(...)):

    data = getInfoFruit(Body)
    output = formatOutput(Body, data)
    response = MessagingResponse()
    msg = response.message(output)

    return Response(content=str(response), media_type="application/xml")


@bot_app.post("/pre_surgery_qa")
async def pre_surgery_qa(Body: str = Form(...)):

    if Body.lower() == "hi":
        output = """Hello Mrs Smith, your hip operation is in 4 weeks.\n
Here is a reminder to do your pre-op exercise: https://www.youtube.com/watch?v=B19AsoXg59c&ab_channel=SWLEOCElectiveOrthopaedicSurgeryInformation \n
Do you have any questions about your operation?"""
    else:
        output = ml_stuff.chat_with_user(msg_=Body, vectorstore=vectorstore)[0]
    response = MessagingResponse()
    msg = response.message(output)
    return Response(content=str(response), media_type="application/xml")


@bot_app.post("/all_seen_chunks")
async def all_seen_chunks():
    output = ml_stuff.return_seen_chunks()
    return output



if __name__ == "__main__":
    uvicorn.run("main:bot_app", host="0.0.0.0", port=5000, reload=True)