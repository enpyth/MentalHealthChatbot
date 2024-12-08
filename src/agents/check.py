from pydantic import BaseModel, EmailStr, Field


class NurseAnswer(BaseModel):
    name: str
    email: str
    gender: int
    is_completed: bool = Field(
        # Check if name, email, gender and age are all set value , True or False.
        description="If all variables have a value, the variable of is_completed should be True, otherwise False."
    )


class EmotionAnswer(BaseModel):
    emotional_tendency: str = Field(
        description="Sentimental Assessment of the statements, should be 'positive' or 'negative'."
    )
