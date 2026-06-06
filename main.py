import discord
import os
import sys
import time
import random
import asyncio
import threading
import urllib.request
import urllib.error
import json
import simdjson
from datetime import datetime
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")

# ============================================================
# CONFIG — edit these to switch behavior
# ============================================================
# Which system prompt file to load (relative to this script).
# Try e.g. "system.txt" or "system_prompts/realone.txt".
SYSTEM_PROMPT_FILE = "system_prompts/malay_aggressive.txt"

# Append the Malay slang reference (system_prompts/malay_slang.txt)?
USE_MALAY_SLANG = False

# Instant mode: reply immediately whenever the bot is mentioned or replied to, with
# NO awareness window, NO burst-merging, and NO debounce wait. When True, the four
# timing settings below are ignored.
INSTANT_MODE = True

# After someone mentions/replies to the bot, it stays "aware" in that channel
# for this many seconds and keeps replying to THAT person without needing another
# mention. Each reply slides the window forward. Set to 0 to only reply when
# directly mentioned or replied to.
AWARENESS_SECONDS = 90

# People often split one thought across several messages (e.g. "wey" then "@asd").
# When the bot is triggered, it merges that person's recent un-answered lines from
# the last this-many seconds into one prompt, so split-up messages read together.
MESSAGE_MERGE_SECONDS = 45

# Before replying, the bot waits until the person has been quiet for this many
# seconds, so it answers a finished burst instead of interrupting mid-thought.
# Each new message resets this timer.
RESPONSE_DEBOUNCE_SECONDS = 3

# While the person is actively typing, keep waiting up to this long for their next
# message before replying anyway. Bridges the pauses between messages in a burst.
TYPING_PATIENCE_SECONDS = 8
# ============================================================

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

_system_prompt_path = os.path.join(os.path.dirname(__file__), SYSTEM_PROMPT_FILE)
try:
    with open(_system_prompt_path, encoding="utf-8") as _f:
        SYSTEM_PROMPT = _f.read().strip()
    if not SYSTEM_PROMPT:
        raise ValueError(f"{SYSTEM_PROMPT_FILE} is empty")
    SYSTEM_PROMPT += (
        "\n\nYou are chatting in a Discord server. Write like a human would: "
        "no punctuation, all lowercase, and reply with one short single-line message only."
    )
except (OSError, ValueError) as e:
    raise SystemExit(
        f"ERROR: Could not load system prompt '{SYSTEM_PROMPT_FILE}' ({e})\n"
        "Check SYSTEM_PROMPT_FILE at the top of main.py."
    ) from e

# --- Optional Malay slang reference ---

_slang = ""
if USE_MALAY_SLANG:
    _slang_path = os.path.join(os.path.dirname(__file__), "system_prompts", "malay_slang.txt")
    try:
        with open(_slang_path, encoding="utf-8") as _f:
            _slang_raw = _f.read()
    except OSError:
        _slang_raw = ""

    # Strip note lines (// ...) meant for the human editor, keep the rest
    _slang = "\n".join(
        line for line in _slang_raw.splitlines() if not line.lstrip().startswith("//")
    ).strip()

if _slang:
    SYSTEM_PROMPT += (
        "\n\nThis server is mostly Malaysian and talks in Gen Z Malay slang and shortform. "
        "Match the language of whoever you reply to: use this Malay slang when they write in Malay, "
        "plain casual English when they write in English. Lean toward the Malay slang as your default vibe. "
        "Use the GLOSSARY to understand and pick words, and copy the feel of the EXAMPLES. "
        "Never translate or explain the slang, just talk like them.\n\n"
        + _slang
    )

print(f"Prompt: '{SYSTEM_PROMPT_FILE}' | malay slang: {'on' if _slang else 'off'}")

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

# Conversations are keyed by USER id: each person has their own thread with the bot.
_conversations: dict[str, list] = _load_conversations()

# Per-channel awareness: channel_id -> (user_id the bot is engaged with, monotonic deadline)
_aware: dict[str, tuple[int, float]] = {}

# Rolling buffer of recent channel chatter (from anyone), injected as ambient context
# so the bot can relate to what's going on around it. In-memory only, not persisted.
_CHANNEL_CONTEXT_MAX = 8
_channel_context: dict[str, deque] = {}

# Recent un-answered lines per (channel_id, user_id) -> list of (monotonic_ts, text).
# Used to merge split-up messages when the bot gets triggered.
_pending: dict[tuple[str, int], list[tuple[float, str]]] = {}

# Per (channel_id, user_id): the scheduled "reply once they're done" task, and the
# monotonic time until which we still consider the person to be typing.
_pending_tasks: dict[tuple[str, int], asyncio.Task] = {}
_typing_until: dict[tuple[str, int], float] = {}

# User ids the bot is "baited" on: it replies to everything they say. In-memory only.
_baited: set[int] = set()

# ---

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

# Event loop handle, set in on_ready, used by the console thread to apply changes safely.
_loop: asyncio.AbstractEventLoop | None = None


async def generate_response(
    user_id: str, user_content: str, context: list[str], turn_hint: str | None = None
) -> str:
    history = _conversations.setdefault(user_id, [])
    history.append({"role": "user", "content": user_content})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"current date and time: {datetime.now():%A, %d %B %Y, %I:%M %p}"},
    ]
    if context:
        messages.append({
            "role": "system",
            "content": "for context, recent messages from others in this channel:\n"
            + "\n".join(context),
        })
    if turn_hint:
        messages.append({"role": "system", "content": turn_hint})
    messages += history

    response = ollama.chat.completions.create(model=OLLAMA_MODEL, messages=messages)
    reply = response.choices[0].message.content

    # Collapse to a single line so the bot always replies with one message
    reply = " ".join(reply.split())

    history.append({"role": "assistant", "content": reply})
    _save_conversations(_conversations)

    return reply


@tree.command(name="bait", description="Make the bot always respond to a user")
@discord.app_commands.describe(user="The user to always respond to")
async def bait(interaction: discord.Interaction, user: discord.User):
    _baited.add(user.id)
    await interaction.response.send_message(
        f"now baiting {user.mention} — i'll respond to everything they say", ephemeral=True
    )


@tree.command(name="unbait", description="Stop always responding to a user")
@discord.app_commands.describe(user="The user to stop responding to")
async def unbait(interaction: discord.Interaction, user: discord.User):
    _baited.discard(user.id)
    await interaction.response.send_message(
        f"stopped baiting {user.mention}", ephemeral=True
    )


@client.event
async def on_ready():
    global _loop
    _loop = asyncio.get_running_loop()
    # Register slash commands instantly in every connected server.
    try:
        for guild in client.guilds:
            tree.copy_global_to(guild=guild)
            await tree.sync(guild=guild)
        print(f"Synced slash commands to {len(client.guilds)} server(s)")
    except Exception as e:
        print(f"Slash command sync failed: {e}")
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("Console ready. Type 'help' for commands.")


async def _do_reply(message: discord.Message, is_mention: bool, merged: str) -> None:
    """Build the prompt from `merged`, generate a reply, and send it."""
    channel_id = str(message.channel.id)
    author_id = message.author.id

    # Prepend replied-to message as context for this turn
    prompt_content = merged
    if message.reference and message.reference.resolved:
        resolved = message.reference.resolved
        if isinstance(resolved, discord.Message) and resolved.content:
            prompt_content = f"[Replying to: \"{resolved.content}\"]\n\n{merged}".strip()

    # Bare ping with no text — fire back a short natural acknowledgement, in tone
    turn_hint = None
    if not prompt_content:
        prompt_content = "(pinged you with no message)"
        turn_hint = (
            "the user just pinged you without typing anything. "
            "respond with a short casual acknowledgement in your usual tone, "
            "like '?', 'hello', 'ye', 'apa', or 'pe'. one or two words only, nothing more."
        )

    context_snapshot = list(_channel_context.get(channel_id, []))

    async with message.channel.typing():
        reply = await generate_response(str(author_id), prompt_content, context_snapshot, turn_hint)

    if len(reply) > 2000:
        reply = reply[:1997] + "..."

    # Use Discord's quoted reply only when mentioned, and only half the time.
    # Otherwise just send a normal message to the channel, like a human would.
    if is_mention and random.random() < 0.5:
        await message.reply(reply, mention_author=False)
    else:
        await message.channel.send(reply)

    # Open / slide the awareness window (skipped in instant mode)
    if not INSTANT_MODE and AWARENESS_SECONDS > 0:
        _aware[channel_id] = (author_id, time.monotonic() + AWARENESS_SECONDS)


async def _respond_when_done(pkey: tuple[str, int], message: discord.Message, is_mention: bool) -> None:
    """Wait until the person has stopped sending messages, then reply to the whole burst."""
    try:
        # Wait for a quiet gap; keep waiting while they're still typing.
        while True:
            await asyncio.sleep(RESPONSE_DEBOUNCE_SECONDS)
            if time.monotonic() >= _typing_until.get(pkey, 0.0):
                break
    except asyncio.CancelledError:
        return

    _pending_tasks.pop(pkey, None)
    _typing_until.pop(pkey, None)

    # Merge the person's recent un-answered lines into one prompt, then consume them.
    merged = "\n".join(txt for (_, txt) in _pending.pop(pkey, []))
    await _do_reply(message, is_mention, merged)


@client.event
async def on_typing(channel, user, when):
    # Note that someone is composing, so the debounce keeps waiting for their message.
    if getattr(user, "bot", False):
        return
    _typing_until[(str(channel.id), user.id)] = time.monotonic() + TYPING_PATIENCE_SECONDS


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    channel_id = str(message.channel.id)
    author_id = message.author.id

    is_mention = client.user in message.mentions
    is_reply_to_bot = (
        message.reference is not None
        and message.reference.resolved is not None
        and isinstance(message.reference.resolved, discord.Message)
        and message.reference.resolved.author == client.user
    )
    # Baited users always get a response, like a standing mention.
    forced = is_mention or is_reply_to_bot or author_id in _baited

    # Awareness: keep replying to the person who pulled us in, until the window lapses
    aware_user, aware_until = _aware.get(channel_id, (None, 0.0))
    aware = (
        AWARENESS_SECONDS > 0
        and author_id == aware_user
        and time.monotonic() < aware_until
    )

    # Strip bot mention from content
    content = message.content
    for mention in message.mentions:
        content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
    content = content.strip()

    # Record ambient channel chatter (even messages we don't reply to) for context.
    if content:
        buf = _channel_context.setdefault(channel_id, deque(maxlen=_CHANNEL_CONTEXT_MAX))
        buf.append(f"{message.author.display_name}: {content}")

    # Instant mode: reply right away if addressed, skipping awareness/merge/debounce.
    if INSTANT_MODE:
        if forced:
            await _do_reply(message, is_mention, content)
        return

    # Accumulate this person's recent lines so a burst of messages can be merged.
    now = time.monotonic()
    pkey = (channel_id, author_id)
    pending = _pending.setdefault(pkey, [])
    pending[:] = [(t, txt) for (t, txt) in pending if now - t <= MESSAGE_MERGE_SECONDS]
    if content:
        pending.append((now, content))

    # Engage if the bot is addressed, still aware, or already collecting this burst.
    if not (forced or aware or pkey in _pending_tasks):
        return

    # Don't reply yet — wait until the person finishes their burst. A new message
    # cancels and reschedules this, so we only respond once they've gone quiet.
    existing = _pending_tasks.get(pkey)
    if existing:
        existing.cancel()
    _pending_tasks[pkey] = asyncio.create_task(_respond_when_done(pkey, message, is_mention))


# --- Console command controller ---

def _reset_user(user_id: str) -> None:
    existed = _conversations.pop(user_id, None) is not None
    # also drop any active awareness windows for this user
    for cid, (auser, _) in list(_aware.items()):
        if str(auser) == user_id:
            _aware.pop(cid, None)
    _save_conversations(_conversations)
    print(f"reset conversation for user {user_id}" if existed
          else f"no conversation found for user {user_id}")


def _reset_all() -> None:
    _conversations.clear()
    _aware.clear()
    _save_conversations(_conversations)
    print("reset all conversations")


def _schedule(fn) -> None:
    """Run a state-mutating function on the bot's event loop thread."""
    if _loop is None:
        print("bot not connected yet, try again in a moment")
        return
    _loop.call_soon_threadsafe(fn)


def _print_help() -> None:
    print(
        "commands:\n"
        "  list                 show stored conversations (user id -> message count)\n"
        "  reset <user_id>      clear one user's conversation + awareness\n"
        "  reset all            clear every conversation\n"
        "  help                 show this help"
    )


def _console_loop() -> None:
    _print_help()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        cmd, *args = line.split()
        cmd = cmd.lower()

        if cmd in ("help", "?"):
            _print_help()
        elif cmd == "list":
            convs = list(_conversations.items())
            if not convs:
                print("(no conversations)")
            else:
                for uid, msgs in convs:
                    print(f"  {uid}: {len(msgs)} messages")
        elif cmd == "reset":
            if not args:
                print("usage: reset <user_id> | reset all")
            elif args[0].lower() == "all":
                _schedule(_reset_all)
            else:
                _schedule(lambda uid=args[0]: _reset_user(uid))
        else:
            print(f"unknown command: {cmd}  (try 'help')")


threading.Thread(target=_console_loop, daemon=True).start()

client.run(DISCORD_TOKEN)
