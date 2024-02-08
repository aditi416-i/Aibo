import chardet
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS as vectordb
from langchain.embeddings.openai import OpenAIEmbeddings
import json
from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv
from flask_cors import CORS
from transformers import pipeline
import time
from twilio.rest import Client


app = Flask(__name__)
CORS(app)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

users = []
# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Read the file using the correct encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


encoding = detect_encoding("sext_base.txt")
with open("sext_base.txt", "r", encoding=encoding) as f:
    text = f.read()

# Write the text back to a new file, ensuring it's in UTF-8 encoding
with open("sext_base_utf8.txt", "w", encoding="utf-8") as f:
    f.write(text)

raw_documents = TextLoader('sext_base_utf8.txt').load()
text_splitter = CharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, separator="He")
documents = text_splitter.split_documents(raw_documents)

db = vectordb.from_documents(documents, OpenAIEmbeddings())

history = []
history_size = 51  # keep this odd


def retrieve_info(query):
    similar_info = db.similarity_search(query, k=4)
    contents = [doc.page_content for doc in similar_info]
    return contents


prompt = "You are Aibo the panda, a cute and friendly mental health support chatbot having a conversation with a human. \nIf you are asked a question that requires external information or if you don't know the answer, refer to the info given after the question. If the info is irrlevant, then ignore it. Give human like responses.\nUse the context from the entire conversation."

history.append({"role": "system", "content": prompt})


# @app.route('/chatbot', methods=['POST'])
def chatbot():
    while True:
        # data = request.json
        # question = data['message']
        question = input("enter message: ")
        question = "He: " + question
        info = retrieve_info(question)
        context = " ".join(info)
        history.append({"role": "user", "content":  question +
                       "\nIf you don't know the answer, refer to the following text from the information database for any relevant context/information:\n"+context})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=history)
        history.pop()
        history.append({"role": "user", "content": question})
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        while (len(history) > history_size):
            history.pop(1)
            history.pop(1)
        print("\n" + "Aibo: " + reply + "\n")
        # print(history)
        # return jsonify({"Answer": reply})


@app.route('/mood', methods=['POST'])
def mood():
    data = request.get_json()
    text = data.get('entry')

    classes = ["Optimistic", "Pessimistic"]
    Model = "cross-encoder/nli-roberta-base"
    Task = "zero-shot-classification"
    sentiment_analysis = pipeline(model=Model, task=Task, from_pt=True)
    dict = sentiment_analysis(
        sequences=text, candidate_labels=classes, multi_label=False)
    arr = dict["scores"]
    index = arr.index(max(arr))
    index1 = dict["labels"].index("Optimistic")
    result = dict["labels"][index]  # class
    score = 10*(dict["scores"][index1])  # 1-10
    timee = time.ctime()

    print("result: " + result)
    print("score: " + str(score))
    # print("time: " + timee)

    # resolving path
    public_folder = "../frontend/public"
    file_path = os.path.join(public_folder, "tracker.json")
    try:
        # Read existing data from the JSON file
        with open(os.path.join(public_folder, "doctorinfo.json"), "r") as file:
            temp = json.load(file)
        temp[timee] = score  # appending to json

        # Write the updated data back to the JSON file
        with open(os.path.join(public_folder, "doctorinfo.json"), 'w') as file:
            json.dump(temp, file, indent=4)
    except:
        pass

    try:
        # Read existing data from the JSON file
        with open(file_path, "r") as file:
            existing_data = json.load(file)
        existing_data[timee] = score  # appending to json

        # Write the updated data back to the JSON file
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
        return jsonify(message='Data appended to JSON file'), 200
    except Exception as e:
        return jsonify(message='Error appending to JSON file', error=str(e)), 500


@app.route('/read_tracker', methods=['GET'])
def read_tracker():
    # resolving path
    public_folder = "../frontend/public"
    file_path = os.path.join(public_folder, "tracker.json")
    try:
        with open(file_path, 'r') as file:
            trackerData = json.load(file)
        # do operations on this current json data

        dates = list(trackerData.keys())
        scores = list(trackerData.values())

        if (len(dates) < 2):
            return jsonify(trackerData)

        last_date = dates[-1]
        second_last_date = dates[-2]

        if last_date[:3] == second_last_date[:3]:
            dates.pop()
            last_score = scores.pop()
            second_last_score = scores.pop()
            average = (last_score + second_last_score) / 2
            scores.append(average)

            while len(dates) > 7:
                dates.pop(0)
                scores.pop(0)

        # Convert lists to a dictionary
        new_data = {}
        for i in range(len(dates)):
            new_data[dates[i]] = scores[i]

        # print("new data: " + jsonify(new_data))

        # Write the updated data to a JSON file
        with open(file_path, "w") as file:
            json.dump(new_data, file, indent=4)
        return jsonify(new_data)

    except Exception as e:
        return jsonify({'error': 'Error reading JSON file'}), 500

# def send_daily_reminders(users):
#     message = "Did you take your medication today?"
#     for user in users:
#         send_whatsapp_message(user['phone_number'], message)

# # schedule daily reminders
# # schedule.every().day.at("16:37").do(send_daily_reminders)

# def send_whatsapp_message(phone_number, message):
#     try:
#         print("hi about to send message")
#         client.messages.create(
#             to=f"whatsapp:+91{phone_number}",
#             from_="whatsapp:+14155238886",
#             body=message
#         )
#     except Exception as e:
#         print(f"Error sending WhatsApp message: {str(e)}")


@app.route('/register', methods=['POST'])
def register_user():
    print(users)
    data = request.get_json()
    username = data.get('username')
    phone_number = data.get('phone_number')

    if not username or not phone_number:
        return jsonify({'message': 'Both username and phone number are required'}), 400
    if any(user['phone_number'] == phone_number for user in users):
        return jsonify({'error': 'Phone number is already registered'}), 400

    users.append({'username': username, 'phone_number': phone_number})
    print(users)
    # send whatsapp messages to all the registered user
    message = "Did you take your medication today?"
    for user in users:
        try:
            print("hi about to send message")
            print(user)
            receiver = "whatsapp:+91" + user['phone_number']
            print(user["phone_number"])
            client.messages.create(
                to=receiver,
                from_="whatsapp:+14155238886",
                body=message
            )
        except Exception as e:
            print(f"Error sending WhatsApp message: {str(e)}")
    return jsonify({'message': "User registered successfully"}), 201


if __name__ == '__main__':
    # app.run(debug=True)
    # while True:
    chatbot()

# while True:
#     schedule.run_pending()
#     time.sleep(1)
