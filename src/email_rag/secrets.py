import keyring

SERVICE = "email-rag-imap"


def set_imap_password(user: str, password: str) -> None:
    keyring.set_password(SERVICE, user, password)


def get_imap_password(user: str) -> str | None:
    return keyring.get_password(SERVICE, user)
