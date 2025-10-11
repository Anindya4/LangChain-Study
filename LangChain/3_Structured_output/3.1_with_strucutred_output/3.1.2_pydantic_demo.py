from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class Student(BaseModel):
    name : str = "Anindya"
    age : Optional[int] = None
    email : EmailStr 
    cgpa : float = Field(ge=0, le=10)

new_student = {'age' : 24, "email":"abc@gmail.com", 'cgpa':10}

student = Student(**new_student)

print(student)
