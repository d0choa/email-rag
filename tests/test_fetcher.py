import pytest

from email_rag.fetcher import Fetcher


class FakeIMAP:
    """Minimal stand-in for IMAPClient covering the calls Fetcher makes."""

    def __init__(self, messages, uidvalidity=100):
        # messages: dict[uid] = raw_bytes
        self._messages = messages
        self._uidvalidity = uidvalidity
        self.logged_out = False

    def login(self, user, password):
        self.user = user

    def select_folder(self, folder, readonly=True):
        self.folder = folder
        return {b"UIDVALIDITY": self._uidvalidity}

    def search(self, criteria):
        # criteria like ['UID', '5:*'] or ['SINCE', date]
        if "UID" in criteria:
            lo = int(criteria[criteria.index("UID") + 1].split(":")[0])
            return [u for u in self._messages if u >= lo]
        return list(self._messages)

    def fetch(self, uids, data):
        return {u: {b"BODY[]": self._messages[u]} for u in uids}

    def logout(self):
        self.logged_out = True


RAW = b"From: a@x\r\nSubject: hi\r\nMessage-ID: <m@x>\r\n\r\nbody"


def test_check_login_success_logs_out():
    fake = FakeIMAP({})
    f = Fetcher("host", 993, "user", "pw", client_factory=lambda **k: fake)
    f.check_login()  # must not raise
    assert fake.user == "user"
    assert fake.logged_out is True


def test_check_login_propagates_auth_failure():
    class FailIMAP(FakeIMAP):
        def login(self, user, password):
            raise RuntimeError("[AUTHENTICATIONFAILED] bad credentials")

    fail = FailIMAP({})
    f = Fetcher("host", 993, "user", "bad", client_factory=lambda **k: fail)
    with pytest.raises(RuntimeError, match="AUTHENTICATIONFAILED"):
        f.check_login()
    assert fail.logged_out is True  # finally still ran


def test_first_sync_fetches_all_and_returns_state():
    fake = FakeIMAP({1: RAW, 2: RAW, 3: RAW})
    f = Fetcher("host", 993, "user", "pw", client_factory=lambda **k: fake)
    fetched = list(f.sync_folder("INBOX", last_seen_uid=0, uidvalidity=None, since=None))
    assert [uid for uid, _raw in fetched] == [1, 2, 3]
    assert f.last_uidvalidity == 100
    assert f.last_uid == 3


def test_incremental_only_fetches_new_uids():
    fake = FakeIMAP({1: RAW, 2: RAW, 3: RAW})
    f = Fetcher("host", 993, "user", "pw", client_factory=lambda **k: fake)
    fetched = list(f.sync_folder("INBOX", last_seen_uid=2, uidvalidity=100, since=None))
    assert [uid for uid, _raw in fetched] == [3]


def test_uidvalidity_change_triggers_full_rescan():
    fake = FakeIMAP({1: RAW, 2: RAW}, uidvalidity=200)
    f = Fetcher("host", 993, "user", "pw", client_factory=lambda **k: fake)
    fetched = list(f.sync_folder("INBOX", last_seen_uid=5, uidvalidity=100, since=None))
    assert [uid for uid, _raw in fetched] == [1, 2]  # rescanned from scratch
