#!/usr/bin/env python

"""
Simplified Chat Service using the OpenAI API for generating responses.
"""

import argparse
import json
import os
import readline  # For history and editing support in the terminal
from datetime import datetime
from os import path
from sys import stdout
from typing import NamedTuple, Optional

from openai import OpenAI

client = OpenAI()


class ChatCompletionParams(NamedTuple):
    """
    Parameters for OpenAI's chat completion API.
    """
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict = {}
    user: str = ""


class ChatConfig(NamedTuple):
    """
    Configuration for the chat service.
    """
    context: str = ""
    params: ChatCompletionParams = ChatCompletionParams()


class ChatService:
    """
    A chat service that interacts with OpenAI API to generate responses.
    """

    def __init__(self, config: ChatConfig):
        self.context = config.context
        self.params = config.params
        self.history = ChatHistory(context=self.context)

    def start(self) -> None:
        """Start the chat session."""
        print('Type "/help" to see available commands.\n')

        try:
            while True:
                user_input = input(">>> ").strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    self.handle_command(user_input)
                else:
                    self.handle_chat(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!\n")

    def handle_command(self, command: str) -> None:
        """Handle various user commands."""
        commands = {
            "/help": self.display_help,
            "/log": self.display_log,
            "/save": self.save_log,
            "/clear": self.clear_log,
            "/forget": self.forget_last_message,
            "/context": self.display_context,
            "/exit": exit,
        }
        action = commands.get(command)
        if action:
            action()
        else:
            print(f"Unknown command: {command}")

    def display_help(self) -> None:
        """Display available commands."""
        print(
            """
/help    - View available commands
/exit    - Exit the program
/log     - View the conversation log
/save    - Save the conversation log to a file
/clear   - Clear the conversation log
/forget  - Remove the last message from the log
/context - Display the current chat context
"""
        )

    def display_log(self) -> None:
        """Display the conversation log."""
        log = self.history.get_log()
        print(f"\n{log if log else 'No messages yet.'}\n")

    def save_log(self) -> None:
        """Save the conversation log to a file."""
        file_path = self.history.save_log()
        print(f"\nLog saved to: {file_path}\n")

    def clear_log(self) -> None:
        """Clear the conversation log."""
        self.history.clear_log()
        print("\nConversation log cleared.\n")

    def forget_last_message(self) -> None:
        """Forget the last message in the conversation."""
        self.history.remove_last_message()
        print("\nLast message removed.\n")

    def display_context(self) -> None:
        """Display the current context."""
        print(f"\n{self.context if self.context else 'No context set.'}\n")

    def handle_chat(self, user_input: str) -> None:
        """Handle user chat input and generate response."""
        messages = self.create_prompt_messages(user_input)
        self.generate_response(messages)

    def create_prompt_messages(self, user_input: str) -> list:
        """Create a prompt for the AI to complete."""
        messages = []
        if self.context:
            messages.append({"role": "system", "content": self.context})
        messages.extend(self.history.get_messages())
        user_message = {"role": "user", "content": user_input}
        messages.append(user_message)
        self.history.add_message(user_message)
        return messages

    def generate_response(self, messages: list) -> None:
        """Generate and display response from the AI."""
        stream = client.chat.completions.create(stream=True, messages=messages, **self.params._asdict())

        response_text = ""
        stdout.write("\n")
        for obj in stream:
            delta = obj.choices[0].delta
            if delta.content:
                content = delta.content
                if response_text or content.strip():
                    stdout.write(content)
                    stdout.flush()
                    response_text += content

        stdout.write("\n\n")
        self.history.add_message({"role": "assistant", "content": response_text.strip()})


class ChatHistory:
    """
    Manages the history of the conversation.
    """

    def __init__(self, context: str = ""):
        self.context = context
        self.messages = []

    def add_message(self, message: dict) -> None:
        self.messages.append(message)

    def get_messages(self) -> list:
        return self.messages

    def get_last_message(self) -> Optional[dict]:
        return self.messages[-1] if self.messages else None

    def remove_last_message(self) -> None:
        if self.messages:
            self.messages.pop()

    def get_log(self) -> str:
        return "\n\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.messages)

    def save_log(self) -> str:
        home_dir = path.expanduser("~")
        log_dir = path.join(home_dir, ".ai", "log")
        os.makedirs(log_dir, exist_ok=True)
        file_name = datetime.now().strftime("%Y%m%d%H%M%S") + ".jsonl"
        file_path = path.join(log_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            log_entries = [json.dumps({"role": "system", "content": self.context})] if self.context else []
            log_entries += [json.dumps(msg, ensure_ascii=False) for msg in self.messages]
            f.write("\n".join(log_entries))

        return file_path

    def clear_log(self) -> None:
        self.messages.clear()


def read_args() -> argparse.Namespace:
    """Read command-line arguments and create chat completion parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="gpt-4", help="Model to use for chat completion")
    parser.add_argument("-M", "--max_tokens", type=int, default=1000, help="Maximum number of tokens for completion")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Temperature setting for randomness")
    return parser.parse_args()


def create_chat_config() -> ChatConfig:
    """Create a chat configuration from command-line arguments."""
    context = load_context()
    args = read_args()
    params = ChatCompletionParams(**vars(args))
    return ChatConfig(context=context, params=params)


def load_context() -> str:
    """Load context for the chat from the context file."""
    context_file = path.join(path.expanduser("~"), ".ai", "context.txt")
    context = f"Current time: {datetime.now().strftime('%a, %b %d %Y %I:%M %p')}\n\n"

    if path.exists(context_file):
        with open(context_file, "r", encoding="utf-8") as f:
            context += f.read().strip()

    return context


def main():
    """Main function to start the chat service."""
    config = create_chat_config()
    chat_service = ChatService(config=config)
    chat_service.start()


if __name__ == "__main__":
    main()
