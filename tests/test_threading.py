from email_rag.threading_ import thread_id_for


def test_root_message_threads_to_itself():
    assert thread_id_for(message_id="<root@x>", in_reply_to=None, references=[]) == "<root@x>"


def test_reply_uses_first_reference():
    tid = thread_id_for(
        message_id="<c@x>",
        in_reply_to="<b@x>",
        references=["<root@x>", "<b@x>"],
    )
    assert tid == "<root@x>"


def test_reply_without_references_uses_in_reply_to():
    tid = thread_id_for(message_id="<b@x>", in_reply_to="<root@x>", references=[])
    assert tid == "<root@x>"
