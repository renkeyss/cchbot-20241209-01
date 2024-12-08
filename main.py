# -*- coding: utf-8 -*-

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
import logging
import numpy as np

# 設置日誌紀錄
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 讀取環境變數
_ = load_dotenv(find_dotenv())

# Dictionary to store user message counts and reset times
user_message_counts = {}

# Dictionary to store conversation history per user
user_conversations = {}

# User daily limit
USER_DAILY_LIMIT = 15

# Maximum conversation history length
MAX_CONVERSATION_LENGTH = 8

# Threshold for topic similarity (越低表示越相似)
SIMILARITY_THRESHOLD = 0.5

def reset_user_count(user_id):
    user_message_counts[user_id] = {
        'count': 0,
        'reset_time': datetime.now() + timedelta(days=1)
    }

# 計算文本向量的餘弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 呼叫 OpenAI Chat API
async def call_openai_chat_api(conversation_history):
    openai.api_key = os.getenv('OPENAI_API_KEY')  # 確保使用環境變數中正確的 API key

    try:
        response = await openai.ChatCompletion.acreate(
            model="ft:gpt-3.5-turbo-1106:personal:20241105:AFCelO98",  # 使用OpenAI的模型
            messages=conversation_history
        )
        assistant_response = response.choices[0]['message']['content']
        logger.info(f"Response from OpenAI assistant: {assistant_response}")
        return assistant_response
    except Exception as e:
        logger.error(f"Error calling OpenAI assistant: {e}")
        return "抱歉，系統出現錯誤，請稍後再試。"

# 獲取文本的嵌入向量
async def get_text_embedding(text):
    try:
        response = await openai.Embedding.acreate(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        logger.error(f"Error getting text embedding: {e}")
        return None

# 獲取 channel_secret 和 channel_access_token
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
if channel_secret is None:
    logger.error('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    logger.error('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

# Initialize LINE Bot Messaging API
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Introduction message
introduction_message = (
    "我是 彰化基督教醫院 內分泌暨新陳代謝科 小助理，如果您有任何關於：糖尿病、高血壓、甲狀腺的相關問題，您可以向我詢問。"
    "但基本上我是由 OPENAI 大型語言模型訓練，所以當您發現我回覆的答案有誤時，建議您要向您的醫療團隊做進一步的諮詢，謝謝！"
)

@app.post("/callback")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

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
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessage):
            continue

        user_id = event.source.user_id
        user_message = event.message.text

        logger.info(f"Received message from user {user_id}: {user_message}")

        # 檢查訊息計數是否需要重置
        if user_id in user_message_counts:
            if datetime.now() >= user_message_counts[user_id]['reset_time']:
                reset_user_count(user_id)
        else:
            reset_user_count(user_id)

        # 檢查用戶是否超過每日限制
        if user_message_counts[user_id]['count'] >= USER_DAILY_LIMIT:
            logger.info(f"User {user_id} exceeded daily limit")
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="您好：您的問題似乎相當多元，但為了讓有限的資源可以讓所有人共享，所以請恕我今天無法再提供回覆，您可明天繼續再次使用本服務，若有急迫性的問題需要瞭解，歡迎致電 04-7238595 分機3239 我們將有專人為您服務，謝謝。")
            )
            continue

        # 處理特殊請求（如介紹）
        if "你是誰" in user_message or "你是誰" in user_message:
            await line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=introduction_message)
            )
            continue

        # 获取用户的对话历史，如果没有则初始化
        conversation_history = user_conversations.get(user_id, [])

        # 添加系统消息（如果是新的对话）
        if not conversation_history:
            conversation_history.append({"role": "system", "content": "你是一位專業的內分泌科的專家，回覆問題要有醫療且專業的口吻，並且都要使用繁體中文回覆。"})

        # 判斷是否離題
        is_off_topic = False
        if len(conversation_history) > 1:
            # 獲取最新一條用戶消息的嵌入向量
            latest_user_message_embedding = await get_text_embedding(user_message)
            # 獲取上一輪用戶消息的嵌入向量
            previous_user_message = None
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    previous_user_message = msg['content']
                    break
            if previous_user_message:
                previous_user_message_embedding = await get_text_embedding(previous_user_message)
                # 計算相似度
                if latest_user_message_embedding and previous_user_message_embedding:
                    similarity = cosine_similarity(latest_user_message_embedding, previous_user_message_embedding)
                    logger.info(f"Similarity between messages: {similarity}")
                    if similarity < SIMILARITY_THRESHOLD:
                        is_off_topic = True
                        logger.info("Detected off-topic message. Resetting conversation history.")

        if is_off_topic:
            # 重置對話歷史，只保留系統消息和當前用戶消息
            conversation_history = [{"role": "system", "content": "你是一位專業的內分泌科的專家，回覆問題要有醫療且專業的口吻，並且都要使用繁體中文回覆。"}]
            conversation_history.append({"role": "user", "content": user_message})
        else:
            # 添加用户消息到对话历史
            conversation_history.append({"role": "user", "content": user_message})

        # 限制对话历史的长度
        if len(conversation_history) > MAX_CONVERSATION_LENGTH:
            # 保留系統消息和最近的對話
            system_message = conversation_history[0]
            conversation_history = [system_message] + conversation_history[-(MAX_CONVERSATION_LENGTH - 1):]

        # 呼叫 OpenAI 助手
        result_text = await call_openai_chat_api(conversation_history)

        # 添加助手的回复到对话历史
        conversation_history.append({"role": "assistant", "content": result_text})

        # 更新用户的对话历史
        user_conversations[user_id] = conversation_history

        # 更新用戶訊息計數
        user_message_counts[user_id]['count'] += 1

        # 回應用戶訊息
        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result_text)
        )

    return 'OK'
