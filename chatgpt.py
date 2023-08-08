import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# response = openai.ChatCompletion.create(
#   model="gpt-4",
#   messages=[
#         {"role": "system", "content":
#         """
# You are Quiz Bot, world's best quiz answerer specializing on macroeconomics. Your answers are based on "Macroeconomics for Business and Policy Professionals" book by Nicholas Vincent and Pierre Yared.
#         """},
#         {"role": "user", "content": """
# A government decides to “cool off” an overheated economy by reducing spending on infrastructure and research and development by $100 Billion without changing the level of taxation. Which of the following is true?

# a) Employment in the construction industry will rise
# b) In the long run the economy may grow more slowly
# c) Real interest rates are likely to rise
# d) Private investment is likely to fall

# Explain answer
# """},
#     ]
# )

# response = openai.ChatCompletion.create(
#   model="gpt-4",
#   messages=[
#         {"role": "system", "content":
#         """
# You are Quiz Bot, world's best quiz answerer specializing on capital markets. Your answers are based on "Investments" book  by Bodie, Kane, and Marcus and "Solutions Manual for Investments" book by Nicholas Racculia.
#         """},
#         {"role": "user", "content": """
# Consider two S&P 500 index funds. Fund X has an expense ratio of 0.60%, while Fund Y has an expense ratio of 0.10%. Since its launch in March 2009, Fund X has returned 12% per year. Since its launch in March 2000, Fund Y has returned 4% per year. Which fund is a better investment for investors today?
#
# a. Fund X
# b. Fund Y
# c. Investors should be indifferent to the two funds
# d. The choice of fund depends on what you expect to happen to the S&P 500 in the future
#
# Explain the answer
# """},
#     ]
# )

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
        {"role": "system", "content":
        """
You are world's best python programming bit.
        """},
        {"role": "user", "content": """
Here is my code for a telegram bot to communicate with OpenAI GPT API. Please convert it to a discord bot.

``` message_entry.py
from typing import Dict
import tiktoken


class MessageEntry:

    def __init__(self, role: 'Role', content: str,
        model: str = 'gpt-3.5-turbo'):
        self.msg: Dict[str, str] = {'role': role.value, 'content': content}
        self.tokens: int = self._calculate_tokens(model)

    def _calculate_tokens(self, model: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        if model == 'gpt-3.5-turbo':
            num_tokens = 4
            for key, value in self.msg.items():
                num_tokens += len(encoding.encode(value))
                if key == 'name':
                    num_tokens += -1
            num_tokens += 2
            return num_tokens
        else:
            raise NotImplementedError(
                f'calculate_tokens() is not presently implemented for model {model}')
```

```token_limiter.py
from enum import Enum
from typing import List, Dict
from collections import deque

import copy

from .message_entry import MessageEntry


class Role(Enum):
    ASSISTANT = 'assistant'
    USER = 'user'
    SYSTEM = 'system'


class TokenizedMessageLimiterException(Exception):
    pass


class TokenizedMessageLimiter:
    def __init__(self, limit: int = 4096):
        self.limit: int = limit
        self.message_history: deque[MessageEntry] = deque()
        self.token_count: int = 0
        # Initialize default system message
        self.system_message: MessageEntry = MessageEntry(
            Role.SYSTEM,
            'You are a helpful assistant that answers messages accurately and concisely')
        self.limit -= self.system_message.tokens

    def set_system_message(self, new_system_msg: str) -> None:
        if not new_system_msg:
            raise TokenizedMessageLimiterException(
                'System message cannot be empty')
        self.limit += self.system_message.tokens
        self.system_message = MessageEntry(Role.SYSTEM, new_system_msg)
        self.limit -= self.system_message.tokens

    def add_message(self, msg: str, role: Role) -> None:
        entry = MessageEntry(role, msg)
        self.message_history.append(entry)
        self.token_count += entry.tokens
        if self.token_count > self.limit:
            _ = self._prune_messages()

    def clean_messages(self):
        _ = self._prune_messages(0)
        self.set_system_message(self.system_message.msg['content'])

    def serialize_messages(self) -> List[str]:
        messages = [self.system_message.msg]
        messages.extend([entry.msg for entry in self.message_history])
        return messages

    def _prune_messages(self, limit: int = None) -> List[MessageEntry]:
        if limit is None:
            limit = self.limit
        pruned_messages: List[MessageEntry] = []
        while self.token_count > self.limit:
            removed_entry = self.message_history.popleft()
            self.token_count -= removed_entry.tokens
            pruned_messages.append(removed_entry)
        return pruned_messages

    def __deepcopy__(self, memo: Dict) -> 'TokenizedMessageLimiter':
        memo[id(self)] = self
        new_obj = TokenizedMessageLimiter(limit=self.limit)
        new_obj.message_history = copy.deepcopy(self.message_history, memo)
        new_obj.token_count = self.token_count
        new_obj.system_message = copy.deepcopy(self.system_message, memo)
        return new_obj
```

```bot.py
from dotenv import load_dotenv; load_dotenv()  # Load all variables first
import logging
import os
import openai
import constants
from util import typing_action
from telegram import Update
from telegram.ext import (
    filters,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    PicklePersistence,
)

from token_limiter.token_limiter import TokenizedMessageLimiter, Role

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: CallbackContext) -> None:
    if 'tokenized_message_limiter' not in context.user_data:
        context.user_data[
            'tokenized_message_limiter'] = TokenizedMessageLimiter()

    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=constants.WELCOME_MESSAGE)


async def system(update: Update, context: CallbackContext) -> None:
    if 'tokenized_message_limiter' not in context.user_data:
        context.user_data[
            'tokenized_message_limiter'] = TokenizedMessageLimiter()
    system_message = update.message.text.replace('/system', '').strip()
    tokenized_message_limiter = context.user_data['tokenized_message_limiter']
    if system_message:
        tokenized_message_limiter.set_system_message(system_message)
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=f'Successfully set system message to "{system_message}"')
    else:
        system_message = tokenized_message_limiter.system_message.msg['content']
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=f'Your current system message is "{system_message}"')


async def clean(update: Update, context: CallbackContext) -> None:
    if 'tokenized_message_limiter' not in context.user_data:
        context.user_data[
            'tokenized_message_limiter'] = TokenizedMessageLimiter()
    tokenized_message_limiter = context.user_data['tokenized_message_limiter']
    tokenized_message_limiter.clean_messages()
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=constants.CLEAN_MESSAGE)


async def message(update: Update, context: CallbackContext) -> None:
    typing_task = context.application.create_task(
        typing_action(context, update.effective_chat.id))
    if 'tokenized_message_limiter' not in context.user_data:
        context.user_data[
            'tokenized_message_limiter'] = TokenizedMessageLimiter()
    user_text = update.message.text
    tokenized_message_limiter = context.user_data['tokenized_message_limiter']
    tokenized_message_limiter.add_message(user_text, Role.USER)
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=tokenized_message_limiter.serialize_messages()
    )
    assistant_response = response['choices'][0]['message']['content']
    tokenized_message_limiter.add_message(assistant_response,
                                          Role.ASSISTANT)
    typing_task.cancel()
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=assistant_response)


async def rejection(update: Update, context: CallbackContext) -> None:
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=constants.REJECTION_MESSAGE)


def main() -> None:
    # Configure optional user filter
    user_id = os.getenv("USER_ID")
    user_filter = filters.User(int(user_id)) if user_id else filters.ALL
    # Configure bot application
    persistence = PicklePersistence(filepath='./local_storage')
    application = ApplicationBuilder().token(
        os.getenv("TELEGRAM_TOKEN")).persistence(persistence).build()
    # Configure main handlers
    start_handler = CommandHandler('start', start, user_filter)
    system_handler = CommandHandler('system', system, user_filter)
    clean_handler = CommandHandler(['clean', 'clear'], clean, user_filter)
    message_handler = MessageHandler(
        user_filter & filters.TEXT & (~filters.COMMAND), message)
    rejection_handler = MessageHandler(~user_filter, rejection)
    application.add_handlers(
        [start_handler, system_handler, clean_handler, message_handler, rejection_handler])
    application.run_polling()


if __name__ == '__main__':
    main()
```

"""},
    ]
)


print("Total tokens: " + str(response['usage']['total_tokens']))
print(response['choices'][0]['message']['content'])
