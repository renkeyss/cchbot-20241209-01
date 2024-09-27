# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import openai
import os
import sys
import aiohttp
from datetime import datetime, timedelta
from fastapi import Request, FastAPI, HTTPException
from linebot import (
    AsyncLineBotApi, WebhookParser
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # read local .env file

# Dictionary to store user message counts and reset times
user_message_counts = {}

# User daily limit
USER_DAILY_LIMIT = 5

# Vector Store for Endocrinology Scientists and Assistant details
VECTOR_STORE_ID = 'vs_G4UCAxMLaXFL4WcwwtUjcJqg'
ASSISTANT_ID = 'asst_ShZXAJwKlokkj9rNhRi2f6pG'
ASSISTANT_NAME = 'CCHDM'

def reset_user_count(user_id):
    user_message_counts[user_id] = {
        'count': 0,
        'reset_time': datetime.now() + timedelta(days=1)
    }

# Initialize OpenAI API

def call_openai_chat_api(user_message, is_classification=False):
    openai.api_key = os.getenv('OPENAI_API_KEY', None)
    
    if is_classification:
        # Use a special prompt for classification
        prompt = (
            "Classify the following message as relevant or non-relevant "
            "to medical, endocrinology, medications, medical quality, or patient safety:\n\n"
            f"{user_message}"
        )
        messages = [
            {"role": "system", "content": f"You are a helpful assistant named {ASSISTANT_NAME}."},
            {"role": "user", "content": prompt},
        ]
    else:
        assistant_instructions = f"Assistant: {ASSISTANT_NAME}, ID: {ASSISTANT_ID}"
        prompt = f"{assistant_instructions}\n\nUser: {user_message}\nAssistant:"
        messages = [
            {"role": "system", "content": f"You are a helpful assistant named {ASSISTANT_NAME}."},
            {"role": "user", "content": prompt},
        ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message['content']

def call_vector_search_api(query):
    openai.api_key = os.getenv('OPENAI_API_KEY', None)
    
    response = openai.Engine(id=VECTOR_STORE_ID).search(
        documents=[query]
    )

    return response['data'][0]['text']

# Get channel_secret and channel_access_token from your environment variables
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

# Initialize LINE Bot Messaging API
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Introduction message
introduction_message = (
    f"我是彰化基督教醫院 內分泌科小助理 {ASSISTANT_NAME}，"
    "您有任何關於：糖尿病、高血壓及內分泌的相關問題都可以問我。"
)

@app.post("/callback")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessage):
            continue

        user_id = event.source.user_id

        # Check if user_ids's count is to be reset
        if user_id in user_message_counts:
            if datetime.now() >= user_message_counts[user_id]['reset_time']:
                reset_user_count(user_id)
        else:
            reset_user_count(user_id)

        # Check if user exceeded daily limit
        if user_message_counts[user_id]['count'] >= USER_DAILY_LIMIT:
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="您今天的用量已經超過，請明天再詢問。")
            )
            continue

        user_message = event.message.text

        # Check if the user is asking for an introduction
        if "介紹" in user_message or "你是誰" in user_message:
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=introduction_message)
            )
            continue

        # Classify the message
        classification_response = call_openai_chat_api(user_message, is_classification=True)

        # Check if the classification is not relevant
        if "non-relevant" in classification_response.lower():
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="您的問題已經超出我的功能，我無法進行回覆，請重新提出您的問題。")
            )
            continue

        # Use vector store for endocrinology-related queries
        vector_search_result = call_vector_search_api(user_message)

        result = vector_search_result or "對不起，我無法找到相關的資訊。"

        # Increment user's message count
        user_message_counts[user_id]['count'] += 1

        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result)
        )

    return 'OK'
