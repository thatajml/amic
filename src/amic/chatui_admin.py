"""Staff tool for the chat UI's accounts.

    python -m amic.chatui_admin list
    python -m amic.chatui_admin reset-pin "Ala"            # asks for the new PIN
    python -m amic.chatui_admin reset-pin "Ala" --pin 1234

Uses the same accounts file as the chat UI (AMIC_CHAT_USERS_DB, default data/chat_users.sqlite).
The learner id is kept, so a learner's memory and profile stay attached after a PIN reset.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from datetime import datetime

from amic.chatui import UserRegistry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show every account")
    rp = sub.add_parser("reset-pin", help="set a new PIN for an account")
    rp.add_argument("name")
    rp.add_argument("--pin", help="new PIN (prompted for when omitted)")
    ap.add_argument("--db", default=os.environ.get("AMIC_CHAT_USERS_DB", "data/chat_users.sqlite"))
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]  # names are Unicode
    reg = UserRegistry(a.db)
    if a.cmd == "list":
        for uid, name, created in reg.list_users():
            print(f"{uid:<28} {name:<24} {datetime.fromtimestamp(created):%Y-%m-%d %H:%M}")
        return 0
    pin = a.pin or getpass.getpass("New PIN: ")
    try:
        print(f"PIN reset for {reg.set_pin(a.name, pin)}.")
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
