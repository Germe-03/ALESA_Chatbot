from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Optional


def _smtp_client():
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "0") or 0)
    user = os.getenv("SMTP_USERNAME")
    pwd = os.getenv("SMTP_PASSWORD")
    sec = (os.getenv("SMTP_SECURITY", "starttls") or "").lower()  # starttls|ssl|none
    if not host or not port:
        return None
    if sec == "ssl":
        client = smtplib.SMTP_SSL(host=host, port=port, timeout=20)
    else:
        client = smtplib.SMTP(host=host, port=port, timeout=20)
        client.ehlo()
        if sec == "starttls":
            client.starttls()
            client.ehlo()
    if user:
        client.login(user, pwd or "")
    return client


def _write_outbox(msg: EmailMessage) -> bool:
    """
    Fallback: schreibt die E-Mail in data/outbox/ fuer lokale Tests.
    """
    try:
        root = Path("data") / "outbox"
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        to = msg.get("To", "unknown").replace("@", "_at_").replace(".", "_")
        fn = root / f"{ts}_to_{to}.eml"
        with open(fn, "wb") as f:
            f.write(bytes(msg))
        return True
    except Exception:
        return False


def send_mail(to: str, subject: str, text: str, html: Optional[str] = None) -> bool:
    """Send a simple UTF-8 email. Returns True on success.

    Env vars:
      SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_SECURITY(starttls|ssl|none), MAIL_FROM
    Fallback: if SMTP not configured, writes .eml to data/outbox/ and returns True.
    """
    to = (to or "").strip()
    if not to:
        return False
    sender = os.getenv("MAIL_FROM", os.getenv("SMTP_USERNAME", "noreply@example.com"))
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    if html:
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
    else:
        msg.set_content(text)

    try:
        client = _smtp_client()
        if client is None:
            return _write_outbox(msg)
        with client as c:
            c.send_message(msg)
        return True
    except Exception:
        # Keine Exception weiterreichen, aber False melden, sodass Aufrufer handeln kann.
        return False


def send_order_confirmation(
    to: str,
    order_id: str,
    items: Iterable[dict],
    customer_name: str = "",
    comment: str | None = None,
) -> bool:
    """
    Baut eine schlanke Bestellbestaetigung (Text + HTML) und versendet sie.
    """
    if not order_id:
        return False

    lines = [
        "Vielen Dank fuer Ihre Bestellung bei ALESA.",
        f"Bestell-ID: {order_id}",
        "",
        "Ihre Positionen:",
    ]
    html_lines = [
        "<p>Vielen Dank f&uuml;r Ihre Bestellung bei ALESA.</p>",
        f"<p><strong>Bestell-ID:</strong> {order_id}</p>",
        "<p><strong>Ihre Positionen:</strong></p>",
        "<ul>",
    ]

    for it in items or []:
        art = it.get("artikelnummer", "?")
        qty = it.get("menge", "?")
        lines.append(f"- {art} x {qty}")
        html_lines.append(f"<li>{art} &times; {qty}</li>")

    html_lines.append("</ul>")

    if comment:
        lines += ["", f"Hinweis: {comment}"]
        html_lines.append(f"<p><strong>Hinweis:</strong> {comment}</p>")

    lines += ["", "Rueckfragen? Antworten Sie einfach auf diese E-Mail."]
    html_lines.append("<p>Rueckfragen? Antworten Sie einfach auf diese E-Mail.</p>")

    subject = "Ihre ALESA Bestellbestaetigung"
    if customer_name:
        subject = f"{subject} - {customer_name}"

    return send_mail(to=to, subject=subject, text="\n".join(lines), html="".join(html_lines))
