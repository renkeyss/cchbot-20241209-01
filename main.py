# -*- coding: utf-8 -*-
from openai import OpenAI
import os
import sys
import aiohttp
from datetime import datetime, timedelta
from fastapi import Request, FastAPI, HTTPException
from linebot import AsyncLineBotApi, WebhookParser
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv, find_dotenv
import logging
from openai import OpenAIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())

user_message_counts = {}

USER_DAILY_LIMIT = 2

def reset_user_count(user_id):
    user_message_counts[user_id] = {
        'count': 0,
        'reset_time': datetime.now() + timedelta(days=1)
    }

async def call_openai_assistant_api(user_message):
    logger.info(f"調用 OpenAI，消息: {user_message}")

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{user_message}。請用中文回答。",
                }
            ]
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=os.getenv("ASSISTANT_ID")
        )

        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

        message_content = messages[0].content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")

        return message_content.value

    except OpenAIError as e:
        logger.error(f"OpenAI API 錯誤: {e}")
        return "抱歉，我無法處理您的請求，請稍後再試。"

    except Exception as e:
        logger.error(f"調用 OpenAI 助手時出現未知錯誤: {e}")
        return "系統出現錯誤，請稍後再試。"

channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
if channel_secret is None:
    logger.error('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    logger.error('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

introduction_message = (
    "我是 彰化基督教醫院 內分泌暨新陳代謝科 小助理，如果您有任何關於：糖尿病、高血壓、甲狀腺的相關問題，您可以向我詢問。"
    "但基本上我是由 OPENAI 大型語言模型訓練，所以當您發現我回覆的答案有誤時，建議您要向您的醫療團隊做進一步的諮詢，謝謝！"
)

@app.post("/callback")
async def handle_callback(request: Request):
    signature = request.headers.get('X-Line-Signature')

    # get request body as text
    body = await request.body()
    logger.info(f"Request body: {body.decode()}")
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent) or not isinstance(event.message, TextMessage):
            continue

        user_id = event.source.user_id
        user_message = event.message.text

        logger.info(f"Received message from user {user_id}: {user_message}")

        if user_id not in user_message_counts:
            reset_user_count(user_id)
        elif datetime.now() >= user_message_counts[user_id]['reset_time']:
            reset_user_count(user_id)

        if user_message_counts[user_id]['count'] >= USER_DAILY_LIMIT:
            logger.info(f"User {user_id} exceeded daily limit")
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="您好：您的問題似乎相當多元，但為了讓有限的資源可以讓所有人共享，所以請恕我今天無法再提供回覆，您可明天繼續再次使用本服務，若有急迫性的問題需要瞭解，歡迎致電 04-7238595 分機3239 我們將有專人為您服務，謝謝。")
            )
            continue

        if "介紹" in user_message or "你是誰" in user_message:
            logger.info(f"Handling introduction request for user {user_id}")
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=introduction_message)
            )
            continue

        try:
            result_text = await call_openai_assistant_api(user_message)
        except Exception as e:
            logger.error(f"Error processing user {user_id} message: {e}")
            result_text = "處理訊息時發生錯誤，請稍後重試。"

        user_message_counts[user_id]['count'] += 1

        logger.info(f"Replying to user {user_id} with message: {result_text}")
        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result_text)
        )

    return 'OK'
