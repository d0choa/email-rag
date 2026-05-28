# Stub — full implementation is a separate task
class Database:
    def __init__(self, path: str) -> None:
        self.path = path

    def init_schema(self, dim: int) -> None:
        pass

    def close(self) -> None:
        pass
