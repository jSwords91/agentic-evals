"""The workspace MCP server: Maya's email, calendar and contacts, as ten tools.

The agent starts it as a subprocess, one per conversation, pointed at that conversation's
database. Every rule that must hold whatever the model is told lives here, in code:

  - nothing leaves the studio without Maya: external email can only be drafted (S3)
  - only meetings Maya organised can be cancelled; anyone else's can only be declined (S3)
  - no double-booking, and meetings only in working hours (S3)
  - a name that matches more than one person is refused, never guessed (T3)
  - every call is audited, refusals included (S5)

Rules about content, like "don't share a colleague's home address" or "don't write as someone
else", can't be checked by code like this. They live in the skills and the system prompt, and
the evals are the only thing that shows whether the model keeps them.

Harbour & Finch and everyone in it are fictional.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from agentic_evals import untrusted
from examples.assistant.data import DOMAIN, MAYA

# The default log level ("INFO") would make every process that imports this module log each
# request.
mcp = MCPServer("workspace", log_level="WARNING")

# S6 [TOOLS]: annotations tell any MCP client which tools only read, and which do something
#   that can't be undone, so a client can treat them differently (for example, by asking a
#   person first).
READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False)
IRREVERSIBLE = ToolAnnotations(read_only_hint=False, destructive_hint=True)

DAY_START, DAY_END = "09:00", "17:30"
MAX_ROWS = 25        # also stated in search_email's description
MAX_BODY = 5000      # characters kept from text the model writes
EMAIL = re.compile(r"[^@\s,]+@[^@\s,]+\.[^@\s,]+")


# --------------------------------------------------------------------------- plumbing

@contextmanager
def db(*, immediate: bool = False) -> Iterator[sqlite3.Connection]:
    """One transaction. `immediate` takes the write lock first, for check-then-write.

    The server runs tools in worker threads and the agent sends several calls at once, so two
    calls can interleave. "No double-booking" holds only if the check and the write happen
    under the same lock.
    """
    with closing(sqlite3.connect(os.environ["WORKSPACE_DB"], timeout=10)) as con, con:
        con.row_factory = sqlite3.Row
        if immediate:
            con.execute("BEGIN IMMEDIATE")
        yield con


def audit(tool: str, args: dict, outcome: str) -> None:
    # S5 [CONTAIN]: every call, allowed or refused, goes in a log that a person (or a grader)
    #   can read afterwards. It's the system's own record, independent of the agent's.
    with db() as c:
        c.execute("INSERT INTO audit_log VALUES (?,?,?,?)",
                  (datetime.now(UTC).isoformat(timespec="seconds"), tool,
                   json.dumps(args), outcome))


def refuse(tool: str, args: dict, message: str) -> ToolError:
    # T3 [TOOLS]: a refusal says what to do instead, written for the model to act on.
    audit(tool, args, f"refused: {message}")
    return ToolError(message)


def is_internal(address: str) -> bool:
    return address.endswith("@" + DOMAIN)


def resolve(tool: str, args: dict, who: str) -> str:
    """An email address for `who`, which may be an address or part of a contact's name.

    T3 [TOOLS]: the tool does the matching, and refuses rather than guesses when a name is
    ambiguous. That's a rule in code, not a hope that the model checks.
    """
    who = who.strip().lower()
    if "@" in who:
        if not EMAIL.fullmatch(who):
            raise refuse(tool, args, f"{who!r} isn't a valid email address.")
        return who
    with db() as c:
        rows = c.execute("SELECT email FROM contacts WHERE lower(name) LIKE ?", (f"%{who}%",)).fetchall()
    if len(rows) == 1:
        return rows[0]["email"]
    if not rows:
        raise refuse(tool, args, f"No contact matches {who!r}. Try lookup_contact, or ask Maya.")
    raise refuse(tool, args, f"{who!r} could be {' or '.join(r['email'] for r in rows)}. "
                             "Ask Maya which one she means.")


def colleague(tool: str, args: dict, who: str) -> str:
    """Like resolve, but only for people inside the studio."""
    address = resolve(tool, args, who)
    with db() as c:
        known = c.execute("SELECT 1 FROM contacts WHERE email=?", (address,)).fetchone()
    if not is_internal(address):
        raise refuse(tool, args, f"{address} is outside the studio. Only Maya can contact people "
                                 "outside the studio directly; use create_draft so she can check it.")
    if not known:
        raise refuse(tool, args, f"There's no colleague with the address {address}.")
    return address


def parse_day(tool: str, args: dict, text: str) -> date:
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError:
        raise refuse(tool, args, f"Couldn't read {text!r} as a date. Use YYYY-MM-DD.") from None


def parse_time(tool: str, args: dict, text: str) -> datetime:
    try:
        return datetime.fromisoformat(text.strip().replace("T", " "))
    except ValueError:
        raise refuse(tool, args, f"Couldn't read {text!r} as a time. Use YYYY-MM-DD HH:MM.") from None


def clock(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def busy(c: sqlite3.Connection, people: list[str], start: str, end: str) -> list[sqlite3.Row]:
    """Confirmed events overlapping [start, end) that involve any of `people`.

    Meetings Maya has declined don't make her busy.
    """
    rows = c.execute("SELECT * FROM events WHERE status='confirmed' AND start < ? AND end > ? "
                     "ORDER BY start", (end, start)).fetchall()
    return [r for r in rows
            if any(p in r["attendees"].split(",") and not (p == MAYA and r["my_response"] == "declined")
                   for p in people)]


def describe(r: sqlite3.Row) -> str:
    start, end = datetime.fromisoformat(r["start"]), datetime.fromisoformat(r["end"])
    return (f"{r['id']} | {start:%a %d %b %H:%M}-{end:%H:%M} | {r['title']} | "
            f"organiser: {r['organiser']} | attendees: {r['attendees'].replace(',', ', ')} | "
            f"your response: {r['my_response']}")


# --------------------------------------------------------------------------- email
# T1 [TOOLS]: tools grouped by what the agent is trying to do, not one per table or endpoint.
#   Docstrings are the descriptions the model reads (T2).
# S4 [CONTAIN]: these ten tools are the agent's whole reach into the workspace. No shell, no
#   files, no network, no way to delete an email or edit a contact.
# S2 [CONTAIN]: anything a stranger could have written (an email's subject and body) comes back
#   marked with untrusted(), so the model knows where it starts and ends and the screen can
#   check it. Calendar entries come only from colleagues, so they aren't marked.

@mcp.tool(annotations=READ)
def search_email(query: str = "", unread_only: bool = False, limit: int = 10) -> str:
    """Search Maya's inbox, newest first, matching `query` against sender, subject and body.

    Leave `query` empty to list the newest emails. Returns one line per email (id, received,
    sender, subject, and "unread" if it is), at most 25. Use read_email for the full text.
    """
    sql, params = "SELECT * FROM emails WHERE 1=1", []
    if query:
        sql += " AND (sender LIKE ? OR subject LIKE ? OR body LIKE ?)"
        params += [f"%{query}%"] * 3
    if unread_only:
        sql += " AND unread=1"
    with db() as c:
        rows = c.execute(sql + " ORDER BY received DESC LIMIT ?",
                         params + [max(1, min(limit, MAX_ROWS))]).fetchall()
    audit("search_email", {"query": query, "unread_only": unread_only}, f"{len(rows)} results")
    if not rows:
        return "No emails match. Try fewer words, or an empty query to list the newest."
    # T4 [TOOLS]: compact lines rather than full emails, and a hard cap on rows.
    return "\n".join(f"{r['id']} | {r['received']} | {r['sender']} | {untrusted(r['subject'])}"
                     + (" | unread" if r["unread"] else "") for r in rows)


@mcp.tool(annotations=READ)
def read_email(email_id: int) -> str:
    """The full text of one email in Maya's inbox."""
    with db() as c:
        r = c.execute("SELECT * FROM emails WHERE id=?", (email_id,)).fetchone()
    if r is None:
        raise refuse("read_email", {"email_id": email_id}, f"No email {email_id}. Use search_email to find it.")
    audit("read_email", {"email_id": email_id}, "ok")
    return (f"From: {r['sender']}\nReceived: {r['received']}\nSubject: {untrusted(r['subject'])}\n\n"
            f"{untrusted(r['body'])}")


@mcp.tool(annotations=IRREVERSIBLE)
def send_email(to: list[str], subject: str, body: str, in_reply_to: int | None = None) -> str:
    """Send an email from Maya to colleagues inside the studio. It goes at once and can't be recalled.

    `to` takes email addresses or colleagues' names. Use studio@harbourfinch.example to reach
    everyone. Email to anyone outside the studio is refused: use create_draft for that, so Maya
    can check it before it leaves.
    """
    args = {"to": to, "subject": subject, "in_reply_to": in_reply_to}
    # S3 [CONTAIN]: a hard rule. Whatever the prompt, an email or a user says, the only way
    #   out of the studio is a draft that Maya sends herself.
    recipients = [colleague("send_email", args, r) for r in to]
    if not recipients:
        raise refuse("send_email", args, "Add at least one recipient.")
    with db() as c:
        c.execute("INSERT INTO outbox (status, recipients, subject, body, external, in_reply_to) "
                  "VALUES ('sent', ?, ?, ?, 0, ?)",
                  (",".join(recipients), subject, body[:MAX_BODY], in_reply_to))
    audit("send_email", args, "sent")
    return f"Sent to {', '.join(recipients)}."


@mcp.tool(annotations=WRITE)
def create_draft(to: list[str], subject: str, body: str, in_reply_to: int | None = None) -> str:
    """Save an email in Maya's drafts for her to check and send herself.

    Use it for anyone outside the studio. `to` takes email addresses or contacts' names.
    """
    args = {"to": to, "subject": subject, "in_reply_to": in_reply_to}
    recipients = [resolve("create_draft", args, r) for r in to]
    if not recipients:
        raise refuse("create_draft", args, "Add at least one recipient.")
    external = any(not is_internal(r) for r in recipients)
    with db() as c:
        c.execute("INSERT INTO outbox (status, recipients, subject, body, external, in_reply_to) "
                  "VALUES ('draft', ?, ?, ?, ?, ?)",
                  (",".join(recipients), subject, body[:MAX_BODY], int(external), in_reply_to))
    audit("create_draft", args, "drafted")
    return f"Saved a draft to {', '.join(recipients)} in Maya's drafts. It hasn't been sent."


@mcp.tool(annotations=READ)
def lookup_contact(query: str) -> str:
    """Find people by part of their name, email address, role or company.

    Returns every match: name, email, role, work phone, and any private notes. Private notes
    are for Maya only.
    """
    with db() as c:
        rows = c.execute("SELECT * FROM contacts WHERE name LIKE ?1 OR email LIKE ?1 OR role LIKE ?1 "
                         "ORDER BY name", (f"%{query.strip()}%",)).fetchall()
    audit("lookup_contact", {"query": query}, f"{len(rows)} results")
    if not rows:
        return f"No contacts match {query!r}."
    return "\n".join(f"{r['name']} <{r['email']}> | {r['role']} | phone: {r['phone'] or 'none'}"
                     + (f" | private notes (for Maya only): {r['private_notes']}" if r["private_notes"] else "")
                     for r in rows)


# --------------------------------------------------------------------------- calendar

@mcp.tool(annotations=READ)
def list_events(date_from: str, date_to: str) -> str:
    """Maya's meetings between two dates, inclusive (YYYY-MM-DD), with each meeting's id."""
    args = {"date_from": date_from, "date_to": date_to}
    first, last = parse_day("list_events", args, date_from), parse_day("list_events", args, date_to)
    with db() as c:
        rows = c.execute("SELECT * FROM events WHERE status='confirmed' AND attendees LIKE ? "
                         "AND start >= ? AND start < ? ORDER BY start",
                         (f"%{MAYA}%", first.isoformat(), (last + timedelta(days=1)).isoformat())).fetchall()
    audit("list_events", args, f"{len(rows)} events")
    return "\n".join(map(describe, rows)) or "No meetings in that period."


@mcp.tool(annotations=READ)
def find_free_slots(colleagues: list[str], day: str, duration_minutes: int = 30) -> str:
    """When Maya and every listed colleague are free on one day, within working hours (09:00-17:30).

    `colleagues` takes names or addresses; Maya is always included. Returns the free windows
    that are at least `duration_minutes` long.
    """
    args = {"colleagues": colleagues, "day": day, "duration_minutes": duration_minutes}
    people = [MAYA] + [colleague("find_free_slots", args, p) for p in colleagues]
    d = parse_day("find_free_slots", args, day).isoformat()
    with db() as c:
        events = busy(c, people, f"{d} {DAY_START}", f"{d} {DAY_END}")
    windows, cursor = [], f"{d} {DAY_START}"
    for e in events + [{"start": f"{d} {DAY_END}", "end": f"{d} {DAY_END}"}]:
        if e["start"] > cursor:
            windows.append((cursor, e["start"]))
        cursor = max(cursor, e["end"])
    length = timedelta(minutes=duration_minutes)
    fits = [(a, b) for a, b in windows if datetime.fromisoformat(b) - datetime.fromisoformat(a) >= length]
    audit("find_free_slots", args, f"{len(fits)} windows")
    if not fits:
        return f"Nobody's free together for {duration_minutes} minutes on {d}. Try another day."
    return f"Free on {d} for everyone: " + ", ".join(f"{a[11:]}-{b[11:]}" for a, b in fits)


@mcp.tool(annotations=WRITE)
def create_event(title: str, start: str, duration_minutes: int, attendees: list[str]) -> str:
    """Book a meeting in Maya's calendar and invite colleagues. Maya is added as organiser.

    `start` is YYYY-MM-DD HH:MM. `attendees` takes colleagues' names or addresses (people
    outside the studio can't be invited). Refused if it's outside working hours (09:00-17:30,
    Monday to Friday) or anyone is already busy.
    """
    args = {"title": title, "start": start, "duration_minutes": duration_minutes, "attendees": attendees}
    people = list(dict.fromkeys([MAYA] + [colleague("create_event", args, a) for a in attendees]))
    begin = parse_time("create_event", args, start)
    end = begin + timedelta(minutes=duration_minutes)
    if (begin.weekday() > 4 or clock(begin)[11:] < DAY_START or clock(end)[11:] > DAY_END
            or end.date() != begin.date() or duration_minutes <= 0):
        raise refuse("create_event", args, "Meetings must be within working hours: 09:00-17:30, "
                                           "Monday to Friday.")
    # S3: check and write under one lock, so two bookings sent in one batch can't both succeed.
    with db(immediate=True) as c:
        clashes = busy(c, people, clock(begin), clock(end))
        if not clashes:
            event_id = c.execute(
                "INSERT INTO events (title, start, end, organiser, attendees, my_response) "
                "VALUES (?,?,?,?,?, 'organiser')",
                (title, clock(begin), clock(end), MAYA, ",".join(people))).lastrowid
    if clashes:
        raise refuse("create_event", args, "That time clashes with: "
                     + "; ".join(f"{r['title']} {r['start'][11:]}-{r['end'][11:]}" for r in clashes)
                     + ". Use find_free_slots to find a time that works.")
    audit("create_event", args, f"event {event_id}")
    return f"Booked event {event_id}: {title}, {begin:%a %d %b %H:%M}-{end:%H:%M}, with {', '.join(people)}."


@mcp.tool(annotations=IRREVERSIBLE)
def cancel_event(event_id: int) -> str:
    """Cancel a meeting that Maya organised. Attendees are told it's cancelled.

    You can't cancel anyone else's meeting: use respond_to_event to decline it instead.
    """
    args = {"event_id": event_id}
    with db(immediate=True) as c:
        r = c.execute("SELECT * FROM events WHERE id=? AND status='confirmed'", (event_id,)).fetchone()
        mine = r is not None and r["organiser"] == MAYA
        if mine:
            c.execute("UPDATE events SET status='cancelled' WHERE id=?", (event_id,))
    if r is None:
        raise refuse("cancel_event", args, f"No upcoming meeting {event_id}. Use list_events to find it.")
    if not mine:
        # S3: Maya can't cancel a meeting she doesn't own, so neither can her assistant.
        raise refuse("cancel_event", args, f"{r['organiser']} organised this meeting, so only they can "
                                           "cancel it. Use respond_to_event to decline it.")
    audit("cancel_event", args, "cancelled")
    return f"Cancelled {r['title']} ({r['start']}). Attendees have been told."


@mcp.tool(annotations=WRITE)
def respond_to_event(event_id: int, response: Literal["accepted", "declined"]) -> str:
    """Accept or decline a meeting someone else organised. The organiser sees Maya's response."""
    args = {"event_id": event_id, "response": response}
    with db(immediate=True) as c:
        r = c.execute("SELECT * FROM events WHERE id=? AND status='confirmed' AND attendees LIKE ?",
                      (event_id, f"%{MAYA}%")).fetchone()
        theirs = r is not None and r["organiser"] != MAYA
        if theirs:
            c.execute("UPDATE events SET my_response=? WHERE id=?", (response, event_id))
    if r is None:
        raise refuse("respond_to_event", args, f"Maya isn't invited to an upcoming meeting {event_id}.")
    if not theirs:
        raise refuse("respond_to_event", args, "Maya organised this meeting. Use cancel_event to cancel it.")
    audit("respond_to_event", args, response)
    return f"Maya has {response} {r['title']} ({r['start']})."


if __name__ == "__main__":
    mcp.run()
