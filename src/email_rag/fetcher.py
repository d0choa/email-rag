from datetime import date

from imapclient import IMAPClient


def _default_factory(host, port, **kwargs):
    return IMAPClient(host, port=port, ssl=True)


class Fetcher:
    def __init__(self, host, port, user, password, client_factory=_default_factory):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self._factory = client_factory
        self.last_uidvalidity = None
        self.last_uid = 0

    def sync_folder(self, folder, last_seen_uid, uidvalidity, since, batch_size=200):
        """Yield (uid, raw_bytes) for new messages; track last_uid/last_uidvalidity.

        since: optional datetime.date for first-run windowing (used only when
        last_seen_uid == 0).
        """
        client = self._factory(host=self.host, port=self.port)
        try:
            client.login(self.user, self.password)
            info = client.select_folder(folder, readonly=True)
            current_validity = info[b"UIDVALIDITY"]
            self.last_uidvalidity = current_validity

            full_rescan = uidvalidity is None or current_validity != uidvalidity
            if full_rescan:
                start_uid = 1
            else:
                start_uid = last_seen_uid + 1

            if last_seen_uid == 0 and since is not None:
                criteria = ["SINCE", since]
            else:
                criteria = ["UID", f"{start_uid}:*"]

            uids = sorted(u for u in client.search(criteria) if u >= start_uid)
            self.last_uid = last_seen_uid
            for i in range(0, len(uids), batch_size):
                batch = uids[i : i + batch_size]
                fetched = client.fetch(batch, ["BODY.PEEK[]"])
                for uid in batch:
                    data = fetched.get(uid, {})
                    raw = data.get(b"BODY[]") or data.get(b"BODY.PEEK[]")
                    if raw is None:
                        continue
                    self.last_uid = max(self.last_uid, uid)
                    yield uid, raw
        finally:
            try:
                client.logout()
            except Exception:
                pass
