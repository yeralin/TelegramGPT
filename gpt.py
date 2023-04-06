from enum import Enum
from typing import Dict, Tuple, List, Union

import discord
import requests

import tiktoken


class BotGPTException(Exception):
    pass


class GPTModel(Enum):
    """
    Enum class for different GPT models with their corresponding name, token limit, and emoji name
    """
    CHAT_GPT = ('gpt-3.5-turbo', 4096, '3️⃣')
    GPT_4 = ('gpt-4', 8192, '4️⃣')

    def __init__(self, version: str, token_limit: int, emoji_name: str):
        self.version = version
        self.token_limit = token_limit
        self.emoji = emoji_name


def match_model_by_emoji(emoji: str) -> Union[GPTModel, None]:
    """
    Returns a GPTModel enum value that matches the specified emoji.

    Args:
        emoji (str): The emoji string to match.

    Returns:
        GPTModel or None: The GPTModel enum value that matches the specified emoji,
            or None if no match is found.
    """
    for model in GPTModel:
        if emoji == model.emoji:
            return model
    return None


async def calculate_tokens(msg: Dict[str, str], model: str = 'gpt-3.5-turbo') -> int:
    """
    Calculates the number of tokens required to process a message.

    Args:
    - msg: A dictionary containing the message to process.
    - model: A string representing the OpenAI model to use.

    Returns:
    An integer representing the number of tokens required to process the message.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')
    if model == 'gpt-3.5-turbo':
        num_tokens = 4
        for key, value in msg.items():
            num_tokens += len(encoding.encode(value))
            if key == 'name':
                num_tokens += -1
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(
            f'calculate_tokens() is not presently implemented for model {model}')


async def construct_gpt_payload(thread: discord.Thread) -> Tuple[List[Dict[str, str]], GPTModel]:
    """
    Constructs OpenAI GPT API message payload from the given channel.

    Args:
    - channel: A discord.ChannelType representing the channel to fetch messages from.

    Returns:
    A list of messages in the GPT API format.
    """
    messages = []
    tokens = 0
    # Fetch system message (the first message in the thread)
    starter_message = thread.starter_message
    if not starter_message:  # starter_message is not cached
        starter_message = await thread.parent.fetch_message(thread.id)

    # Extract chosen GPT model from the initial message
    model_emoji = starter_message.reactions[0].emoji
    model = match_model_by_emoji(model_emoji)
    if model is None:
        raise BotGPTException(f'Model was not found by {model_emoji}')
    # Construct GPT's system message
    system_message = {'role': 'system', 'content': starter_message.content}
    tokens += await calculate_tokens(system_message)
    messages.append(system_message)
    # Fetches history in reverse order
    async for msg in thread.history():
        if msg.type != discord.MessageType.default:
            continue
        content = msg.content
        for attachment in msg.attachments:
            response = requests.get(attachment.url)
            content += response.text
        entry = {
            'role': 'assistant' if msg.author.bot else 'user', 'content': content
        }
        tokens += await calculate_tokens(entry)
        if tokens > model.token_limit:
            break
        messages.insert(1, entry)
    return messages, model