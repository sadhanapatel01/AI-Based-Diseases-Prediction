from typing import Optional, Literal
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator, ValidationError

class UserRegisterSchema(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100, description="Full Name of the user")
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password of at least 6 characters")
    confirm_password: str = Field(..., min_length=6, max_length=128)

    @field_validator('full_name')
    @classmethod
    def clean_name(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Name must be at least 2 characters long.")
        return cleaned

    @model_validator(mode='after')
    def check_passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match. Please ensure both passwords match.")
        return self

class UserLoginSchema(BaseModel):
    email: EmailStr = Field(..., description="Registered email address")
    password: str = Field(..., min_length=1, description="Account password")

class DiagnosisFormSchema(BaseModel):
    choice: Literal['1', '2'] = Field(..., description="'1' for symptoms, '2' for known disease")
    gender: Literal['Male', 'Female', 'Other'] = Field(..., description="Patient gender")
    age: int = Field(..., ge=1, le=125, description="Patient age between 1 and 125")
    diabetes: Literal['yes', 'no'] = Field(default='no', description="Diabetes status")
    previous_diseases: str = Field(default='none', max_length=500, description="Medical history")
    symptoms: Optional[str] = Field(default=None, max_length=2000)
    disease: Optional[str] = Field(default=None, max_length=300)

    @model_validator(mode='after')
    def validate_choice_inputs(self):
        if self.choice == '1':
            if not self.symptoms or not self.symptoms.strip():
                raise ValueError("Please provide detailed symptoms for analysis.")
        elif self.choice == '2':
            if not self.disease or not self.disease.strip():
                raise ValueError("Please specify the disease or condition name.")
        return self

class MedicineQuerySchema(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Medicine name")
    search_by: Literal['generic', 'brand'] = Field(default='generic')
    country: Optional[str] = Field(default='US', max_length=100)

    @field_validator('name')
    @classmethod
    def clean_medicine_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Medicine name cannot be empty.")
        return cleaned

class ChatQuerySchema(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="User question or prompt")
    topic: Optional[str] = Field(default=None, max_length=200)

def validate_form(schema_cls, form_data: dict):
    """
    Validates form data dictionary against a Pydantic schema class.
    Returns: (parsed_instance_or_none, error_message_or_none, errors_dict)
    """
    # Clean up empty strings or whitespace
    cleaned_dict = {}
    for k, v in form_data.items():
        if isinstance(v, str):
            cleaned_dict[k] = v.strip()
        else:
            cleaned_dict[k] = v
            
    try:
        instance = schema_cls(**cleaned_dict)
        return instance, None, {}
    except ValidationError as e:
        field_errors = {}
        for err in e.errors():
            loc = err['loc'][-1] if err['loc'] else 'general'
            field_errors[str(loc)] = err['msg']
        first_msg = e.errors()[0]['msg'] if e.errors() else "Validation failed"
        # Clean pydantic message prefix like 'Value error, '
        if first_msg.startswith("Value error, "):
            first_msg = first_msg.replace("Value error, ", "")
        return None, first_msg, field_errors
