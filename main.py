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

SYSTEM_PROMPT_FILE = "system_prompts/realone.txt"
INSTANT_MODE = True
AWARENESS_SECONDS = 90
MESSAGE_MERGE_SECONDS = 45
RESPONSE_DEBOUNCE_SECONDS = 3
TYPING_PATIENCE_SECONDS = 8

# Message-search tool: how many messages to scan per registered channel, and how
# many matches to hand back to the model.
SEARCH_HISTORY_LIMIT = 300
SEARCH_MAX_RESULTS = 15

# Only this Discord user id may run the slash commands.
OWNER_ID = 640556322194063434


def _ollama_base():
    return OLLAMA_BASE_URL.removesuffix("/v1").removesuffix("/")


def _preflight():
    tags_url = f"{_ollama_base()}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError:
        raise SystemExit("ERROR: Ollama is not running.\nStart it with: ollama serve")

    pulled = {m["name"].split(":")[0] for m in data.get("models", [])}
    if OLLAMA_MODEL.split(":")[0] not in pulled:
        raise SystemExit(
            f"ERROR: Model '{OLLAMA_MODEL}' is not pulled.\nRun: ollama pull {OLLAMA_MODEL}"
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
    raise SystemExit(f"ERROR: Could not load system prompt '{SYSTEM_PROMPT_FILE}' ({e})") from e

print(f"Prompt: '{SYSTEM_PROMPT_FILE}'")

_CONV_FILE = os.path.join(os.path.dirname(__file__), "conversations.json")
_parser = simdjson.Parser()


def _load_conversations():
    if not os.path.exists(_CONV_FILE):
        return {}
    with open(_CONV_FILE, "rb") as f:
        raw = f.read()
    if not raw.strip():
        return {}
    return _parser.parse(raw, recursive=True)


def _save_conversations(convs):
    with open(_CONV_FILE, "w", encoding="utf-8") as f:
        json.dump(convs, f)


# Per-server list of channels the search tool is allowed to look through.
_SEARCH_FILE = os.path.join(os.path.dirname(__file__), "search_channels.json")


def _load_search_channels():
    if not os.path.exists(_SEARCH_FILE):
        return {}
    try:
        with open(_SEARCH_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    return {gid: set(ids) for gid, ids in raw.items()}


def _save_search_channels():
    with open(_SEARCH_FILE, "w", encoding="utf-8") as f:
        json.dump({gid: sorted(ids) for gid, ids in _search_channels.items()}, f)


# Channels the bot always replies in — remembered across restarts.
_BAIT_FILE = os.path.join(os.path.dirname(__file__), "bait_channels.json")


def _load_bait_channels():
    if not os.path.exists(_BAIT_FILE):
        return set()
    try:
        with open(_BAIT_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    except (OSError, ValueError):
        return set()


def _save_bait_channels():
    with open(_BAIT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(_bait_channels), f)


_conversations: dict[str, list] = _load_conversations()
_search_channels: dict[str, set[int]] = _load_search_channels()
# Awareness: channel_id -> (user_id the bot is engaged with, monotonic deadline).
# While active, the bot keeps replying to THAT person without needing a mention.
_aware: dict[str, tuple[int, float]] = {}
_CHANNEL_CONTEXT_MAX = 8
_channel_context: dict[str, deque] = {}
_pending: dict[tuple[str, int], list[tuple[float, str]]] = {}
_pending_tasks: dict[tuple[str, int], asyncio.Task] = {}
_typing_until: dict[tuple[str, int], float] = {}
_baited: set[int] = set()
_bait_channels: set[int] = _load_bait_channels()  # channel ids where the bot replies to everyone

intents = discord.Intents.default()
intents.message_content = True
client = discord.Bot(intents=intents)  # py-cord uses discord.Bot

_loop: asyncio.AbstractEventLoop | None = None


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_messages",
        "description": (
            "Search what members have said before, in this server's registered channels. "
            "Use it to dig up or reference something a user said previously. "
            "Returns matching past messages, newest first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "keyword or phrase to look for"},
                "user": {"type": "string", "description": "display name of the member to limit to (optional)"},
            },
            "required": ["query"],
        },
    },
}
TOOLS = [SEARCH_TOOL]


async def _run_search_tool(message: discord.Message, args: dict) -> str:
    guild = message.guild
    if guild is None:
        return "search not available outside a server"
    chan_ids = _search_channels.get(str(guild.id), set())
    if not chan_ids:
        return "no channels are registered for search on this server (use /search_add)"

    query = (args.get("query") or "").strip().lower()
    user_name = (args.get("user") or "").strip().lower()

    matches = []
    for cid in list(chan_ids):
        ch = guild.get_channel(cid)
        if ch is None:
            continue
        try:
            async for m in ch.history(limit=SEARCH_HISTORY_LIMIT):
                if m.author.bot or not m.content:
                    continue
                if user_name and user_name not in m.author.display_name.lower():
                    continue
                if query and query not in m.content.lower():
                    continue
                matches.append(f"{m.author.display_name}: {m.content}")
                if len(matches) >= SEARCH_MAX_RESULTS:
                    break
        except discord.Forbidden:
            continue
        if len(matches) >= SEARCH_MAX_RESULTS:
            break

    return "\n".join(matches) if matches else "no matching messages found"


async def generate_response(user_id, user_content, context, turn_hint=None, message=None):
    history = _conversations.setdefault(user_id, [])
    history.append({"role": "user", "content": user_content})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"current date and time: {datetime.now():%A, %d %B %Y, %I:%M %p}"},
    ]
    if context:
        messages.append({
            "role": "system",
            "content": "for context, recent messages from others in this channel:\n" + "\n".join(context),
        })
    if turn_hint:
        messages.append({"role": "system", "content": turn_hint})
    messages += history

    # Let the model call the search tool (only when we have a message to search from).
    use_tools = {"tools": TOOLS} if message is not None else {}
    response = ollama.chat.completions.create(model=OLLAMA_MODEL, messages=messages, **use_tools)
    choice = response.choices[0].message

    rounds = 0
    while message is not None and choice.tool_calls and rounds < 3:
        messages.append({
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in choice.tool_calls
            ],
        })
        for tc in choice.tool_calls:
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                tool_args = {}
            result = (await _run_search_tool(message, tool_args)
                      if tc.function.name == "search_messages" else "unknown tool")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        response = ollama.chat.completions.create(model=OLLAMA_MODEL, messages=messages, tools=TOOLS)
        choice = response.choices[0].message
        rounds += 1

    reply = " ".join((choice.content or "").split())
    # If the model stalled on tool calls without a final answer, force one without tools.
    if not reply:
        response = ollama.chat.completions.create(model=OLLAMA_MODEL, messages=messages)
        reply = " ".join((response.choices[0].message.content or "").split())

    history.append({"role": "assistant", "content": reply})
    _save_conversations(_conversations)
    return reply


# py-cord slash commands use @client.slash_command. Every command is owner-only:
# call this first and bail if it returns True.
async def _deny_if_not_owner(ctx: discord.ApplicationContext) -> bool:
    if ctx.author.id != OWNER_ID:
        await ctx.respond("you can't use this", ephemeral=True)
        return True
    return False


@client.slash_command(name="bait", description="Make the bot always respond to a user")
async def bait(ctx: discord.ApplicationContext, user: discord.User):
    if await _deny_if_not_owner(ctx):
        return
    _baited.add(user.id)
    await ctx.respond(f"now baiting {user.mention} — i'll respond to everything they say", ephemeral=True)


@client.slash_command(name="unbait", description="Stop always responding to a user")
async def unbait(ctx: discord.ApplicationContext, user: discord.User):
    if await _deny_if_not_owner(ctx):
        return
    _baited.discard(user.id)
    await ctx.respond(f"stopped baiting {user.mention}", ephemeral=True)


@client.slash_command(name="bait_channel", description="Make the bot always respond to everyone in a channel")
async def bait_channel(ctx: discord.ApplicationContext, channel: discord.TextChannel = None):
    if await _deny_if_not_owner(ctx):
        return
    channel = channel or ctx.channel
    _bait_channels.add(channel.id)
    _save_bait_channels()
    await ctx.respond(f"now always responding in {channel.mention}", ephemeral=True)


@client.slash_command(name="unbait_channel", description="Stop always responding in a channel")
async def unbait_channel(ctx: discord.ApplicationContext, channel: discord.TextChannel = None):
    if await _deny_if_not_owner(ctx):
        return
    channel = channel or ctx.channel
    _bait_channels.discard(channel.id)
    _save_bait_channels()
    await ctx.respond(f"stopped always responding in {channel.mention}", ephemeral=True)


@client.slash_command(name="search_add", description="Let the bot search this (or a given) channel on this server")
async def search_add(ctx: discord.ApplicationContext, channel: discord.TextChannel = None):
    if await _deny_if_not_owner(ctx):
        return
    if ctx.guild is None:
        await ctx.respond("this only works in a server", ephemeral=True)
        return
    channel = channel or ctx.channel
    _search_channels.setdefault(str(ctx.guild.id), set()).add(channel.id)
    _save_search_channels()
    await ctx.respond(f"added {channel.mention} to the search list", ephemeral=True)


@client.slash_command(name="search_remove", description="Stop the bot from searching a channel")
async def search_remove(ctx: discord.ApplicationContext, channel: discord.TextChannel = None):
    if await _deny_if_not_owner(ctx):
        return
    if ctx.guild is None:
        await ctx.respond("this only works in a server", ephemeral=True)
        return
    channel = channel or ctx.channel
    _search_channels.get(str(ctx.guild.id), set()).discard(channel.id)
    _save_search_channels()
    await ctx.respond(f"removed {channel.mention} from the search list", ephemeral=True)


@client.slash_command(name="search_list", description="Show which channels the bot can search on this server")
async def search_list(ctx: discord.ApplicationContext):
    if await _deny_if_not_owner(ctx):
        return
    if ctx.guild is None:
        await ctx.respond("this only works in a server", ephemeral=True)
        return
    ids = _search_channels.get(str(ctx.guild.id), set())
    if not ids:
        await ctx.respond("no channels registered — use /search_add", ephemeral=True)
        return
    await ctx.respond("searching: " + ", ".join(f"<#{cid}>" for cid in ids), ephemeral=True)


@client.event
async def on_ready():
    global _loop
    _loop = asyncio.get_running_loop()
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("Console ready. Type 'help' for commands.")


def _resolve_mentions(msg: discord.Message, drop_bot: bool = False) -> str:
    """Turn raw <@id>/<#id>/<@&id> tokens into readable @name / #channel names."""
    text = msg.content
    for u in msg.mentions:
        name = "" if (drop_bot and u == client.user) else f"@{u.display_name}"
        text = text.replace(f"<@{u.id}>", name).replace(f"<@!{u.id}>", name)
    for ch in msg.channel_mentions:
        text = text.replace(f"<#{ch.id}>", f"#{ch.name}")
    for role in msg.role_mentions:
        text = text.replace(f"<@&{role.id}>", f"@{role.name}")
    return text.strip()


async def _do_reply(message: discord.Message, is_mention: bool, merged: str):
    channel_id = str(message.channel.id)
    author_id = message.author.id

    prompt_content = merged
    if message.reference and message.reference.resolved:
        resolved = message.reference.resolved
        if isinstance(resolved, discord.Message) and resolved.content:
            quoted = _resolve_mentions(resolved)
            prompt_content = f"[Replying to {resolved.author.display_name}: \"{quoted}\"]\n\n{merged}".strip()

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
        reply = await generate_response(str(author_id), prompt_content, context_snapshot, turn_hint, message)

    if len(reply) > 2000:
        reply = reply[:1997] + "..."

    if is_mention and random.random() < 0.5:
        await message.reply(reply, mention_author=False)
    else:
        await message.channel.send(reply)

    if not INSTANT_MODE and AWARENESS_SECONDS > 0:
        _aware[channel_id] = (author_id, time.monotonic() + AWARENESS_SECONDS)


async def _respond_when_done(pkey, message: discord.Message, is_mention: bool):
    try:
        while True:
            await asyncio.sleep(RESPONSE_DEBOUNCE_SECONDS)
            if time.monotonic() >= _typing_until.get(pkey, 0.0):
                break
    except asyncio.CancelledError:
        return

    _pending_tasks.pop(pkey, None)
    _typing_until.pop(pkey, None)
    merged = "\n".join(txt for (_, txt) in _pending.pop(pkey, []))
    await _do_reply(message, is_mention, merged)


@client.event
async def on_typing(channel, user, when):
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
    forced = (
        is_mention
        or is_reply_to_bot
        or author_id in _baited
        or message.channel.id in _bait_channels
    )

    aware_user, aware_until = _aware.get(channel_id, (None, 0.0))
    aware = (
        AWARENESS_SECONDS > 0
        and author_id == aware_user
        and time.monotonic() < aware_until
    )

    # Resolve mentions so the AI understands the conversation: drop the bot's own
    # mention (it's just the trigger), but turn other members, channels, and roles
    # into readable names (e.g. "@ava", "#general") so it knows who/what is referenced.
    content = _resolve_mentions(message, drop_bot=True)

    if content:
        buf = _channel_context.setdefault(channel_id, deque(maxlen=_CHANNEL_CONTEXT_MAX))
        buf.append(f"{message.author.display_name}: {content}")

    if INSTANT_MODE:
        if forced:
            await _do_reply(message, is_mention, content)
        return

    now = time.monotonic()
    pkey = (channel_id, author_id)
    pending = _pending.setdefault(pkey, [])
    pending[:] = [(t, txt) for (t, txt) in pending if now - t <= MESSAGE_MERGE_SECONDS]
    if content:
        pending.append((now, content))

    if not (forced or aware or pkey in _pending_tasks):
        return

    existing = _pending_tasks.get(pkey)
    if existing:
        existing.cancel()
    _pending_tasks[pkey] = asyncio.create_task(_respond_when_done(pkey, message, is_mention))


def _reset_user(user_id):
    existed = _conversations.pop(user_id, None) is not None
    for cid, (auser, _) in list(_aware.items()):
        if str(auser) == user_id:
            _aware.pop(cid, None)
    _save_conversations(_conversations)
    print(f"reset conversation for user {user_id}" if existed else f"no conversation found for user {user_id}")


def _reset_all():
    _conversations.clear()
    _aware.clear()
    _save_conversations(_conversations)
    print("reset all conversations")


def _bait_user(user_id):
    _baited.add(user_id)
    print(f"now baiting user {user_id}")


def _unbait_user(user_id):
    _baited.discard(user_id)
    print(f"stopped baiting user {user_id}")


def _schedule(fn):
    if _loop is None:
        print("bot not connected yet, try again in a moment")
        return
    _loop.call_soon_threadsafe(fn)


def _print_help():
    print(
        "commands:\n"
        "  list                 show stored conversations (user id -> message count)\n"
        "  reset <user_id>      clear one user's conversation + awareness\n"
        "  reset all            clear every conversation\n"
        "  bait <user_id>       always respond to this user\n"
        "  unbait <user_id>     stop always responding to this user\n"
        "  baited               list baited user ids\n"
        "  help                 show this help"
    )


def _console_loop():
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
        elif cmd in ("bait", "unbait"):
            if not args:
                print(f"usage: {cmd} <user_id>")
            elif not args[0].isdigit():
                print(f"user id must be a number, got '{args[0]}'")
            else:
                fn = _bait_user if cmd == "bait" else _unbait_user
                _schedule(lambda uid=int(args[0]): fn(uid))
        elif cmd == "baited":
            ids = sorted(_baited)
            print("  " + ", ".join(str(i) for i in ids) if ids else "(none baited)")
        else:
            print(f"unknown command: {cmd}  (try 'help')")


threading.Thread(target=_console_loop, daemon=True).start()
client.run(DISCORD_TOKEN)