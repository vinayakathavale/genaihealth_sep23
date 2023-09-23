from pydantic import BaseModel


class MsgInput(BaseModel):
    text: str


class MsgOutput(BaseModel):
    text: str