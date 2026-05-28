import email_rag.secrets as secrets


def test_set_and_get(monkeypatch):
    store = {}
    monkeypatch.setattr(
        secrets.keyring, "set_password",
        lambda svc, user, pw: store.__setitem__((svc, user), pw),
    )
    monkeypatch.setattr(
        secrets.keyring, "get_password",
        lambda svc, user: store.get((svc, user)),
    )
    secrets.set_imap_password("ochoa@ebi.ac.uk", "hunter2")
    assert secrets.get_imap_password("ochoa@ebi.ac.uk") == "hunter2"
    assert secrets.get_imap_password("nobody@x.com") is None
