"""Alerts: Telegram and email through env vars, and always a local JSONL log.

Nothing is configured in the repo. Set ITRADER_TG_TOKEN and ITRADER_TG_CHAT for Telegram;
ITRADER_SMTP_URL (smtp://user:pass@host:port) and ITRADER_ALERT_EMAIL for email. Failures to
deliver never stop the loop; they are logged.
"""
from __future__ import annotations

import json
import os
import smtplib
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data.feeds import _CTX
from .settings import Settings, settings as default_settings


class Alerts:
    def __init__(self, s: Optional[Settings] = None, log_path: Optional[Path] = None, enabled: bool = True) -> None:
        self.s = s or default_settings()
        self.a = self.s.alerts
        self.log_path = Path(log_path) if log_path else None
        self.enabled = enabled
        self.sent: List[Dict[str, Any]] = []

    def channels(self) -> Dict[str, bool]:
        return {"telegram": bool(os.environ.get(self.a.telegram_token_env)) and bool(os.environ.get(self.a.telegram_chat_env)),
                "email": bool(os.environ.get(self.a.smtp_url_env)) and bool(os.environ.get(self.a.email_to_env)),
                "log": self.log_path is not None}

    def send(self, kind: str, text: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, "text": text, "data": data or {}, "delivered": []}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            rec["delivered"].append("log")
        if not self.enabled or kind not in set(self.a.on):
            self.sent.append(rec)
            return rec
        msg = f"[iTRADER {kind}] {text}"
        try:
            tok, chat = os.environ.get(self.a.telegram_token_env), os.environ.get(self.a.telegram_chat_env)
            if tok and chat:
                body = urllib.parse.urlencode({"chat_id": chat, "text": msg[:4000]}).encode()
                urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=body), timeout=10, context=_CTX).read()
                rec["delivered"].append("telegram")
        except Exception as exc:  # noqa: BLE001
            rec["telegram_error"] = f"{type(exc).__name__}: {exc}"
        try:
            url, to = os.environ.get(self.a.smtp_url_env), os.environ.get(self.a.email_to_env)
            if url and to:
                u = urllib.parse.urlparse(url)
                em = EmailMessage()
                em["Subject"], em["From"], em["To"] = msg[:120], u.username or to, to
                em.set_content(msg + "\n\n" + json.dumps(data or {}, indent=1, default=str))
                with smtplib.SMTP(u.hostname, u.port or 587, timeout=10) as smtp:
                    smtp.starttls()
                    if u.username:
                        smtp.login(u.username, u.password or "")
                    smtp.send_message(em)
                rec["delivered"].append("email")
        except Exception as exc:  # noqa: BLE001
            rec["email_error"] = f"{type(exc).__name__}: {exc}"
        self.sent.append(rec)
        return rec
