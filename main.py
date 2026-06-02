import discord
import os
import urllib.request
import urllib.error
import json
import simdjson
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")

def _ollama_base() -> str:
    return OLLAMA_BASE_URL.removesuffix("/v1").removesuffix("/")

def _preflight():
    tags_url = f"{_ollama_base()}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError:
        raise SystemExit(
            "ERROR: Ollama is not running.\n"
            "Start it with: ollama serve"
        )

    pulled = {m["name"].split(":")[0] for m in data.get("models", [])}

    if OLLAMA_MODEL.split(":")[0] not in pulled:
        raise SystemExit(
            f"ERROR: Model '{OLLAMA_MODEL}' is not pulled.\n"
            f"Run: ollama pull {OLLAMA_MODEL}"
        )

    print(f"Ollama OK — main: '{OLLAMA_MODEL}'")

_preflight()

ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

_system_prompt_path = os.path.join(os.path.dirname(__file__), "system.txt")
try:
    with open(_system_prompt_path, encoding="utf-8") as _f:
        SYSTEM_PROMPT = _f.read().strip()
    if not SYSTEM_PROMPT:
        raise ValueError("system.txt is empty")
    SYSTEM_PROMPT += (
        "\n\nYou are chatting in a Discord server. Write like a human would: "
        "no punctuation, all lowercase, and reply with one short single-line message only."
    )
except (OSError, ValueError) as e:
    raise SystemExit(
        f"ERROR: Could not load system.txt ({e})\n"
        "Please redownload it from the GitHub repository."
    ) from e

# --- Conversation history ---

_CONV_FILE = os.path.join(os.path.dirname(__file__), "conversations.json")
_parser = simdjson.Parser()

def _load_conversations() -> dict[str, list]:
    if not os.path.exists(_CONV_FILE):
        return {}
    with open(_CONV_FILE, "rb") as f:
        raw = f.read()
    if not raw.strip():
        return {}
    return _parser.parse(raw, recursive=True)

def _save_conversations(convs: dict[str, list]) -> None:
    with open(_CONV_FILE, "w", encoding="utf-8") as f:
        json.dump(convs, f)

_conversations: dict[str, list] = _load_conversations()

# ---

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def generate_response(channel_id: str, user_content: str) -> str:
    history = _conversations.setdefault(channel_id, [])
    history.append({"role": "user", "content": user_content})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    response = ollama.chat.completions.create(model=OLLAMA_MODEL, messages=messages)
    reply = response.choices[0].message.content

    # Collapse to a single line so the bot always replies with one message
    reply = " ".join(reply.split())

    history.append({"role": "assistant", "content": reply})
    _save_conversations(_conversations)

    return reply


@client.event
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id})")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    channel_id = str(message.channel.id)

    is_mention = client.user in message.mentions
    is_reply_to_bot = (
        message.reference is not None
        and message.reference.resolved is not None
        and isinstance(message.reference.resolved, discord.Message)
        and message.reference.resolved.author == client.user
    )
    if not (is_mention or is_reply_to_bot):
        return

    # Strip bot mention from content
    content = message.content
    for mention in message.mentions:
        content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
    content = content.strip()

    # Prepend replied-to message as context
    if message.reference and message.reference.resolved:
        resolved = message.reference.resolved
        if isinstance(resolved, discord.Message) and resolved.content:
            content = f"[Replying to: \"{resolved.content}\"]\n\n{content}"

    async with message.channel.typing():
        reply = await generate_response(channel_id, content)

    if len(reply) > 2000:
        reply = reply[:1997] + "..."

    await message.reply(reply, mention_author=False)


client.run(DISCORD_TOKEN)
