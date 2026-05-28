def thread_id_for(message_id: str, in_reply_to: str | None, references: list[str]) -> str:
    """Approximate JWZ threading: the root of the reference chain.

    The first References entry is the conversation root. Fall back to
    In-Reply-To, then to the message's own id (a thread of one).
    """
    if references:
        return references[0]
    if in_reply_to:
        return in_reply_to
    return message_id
