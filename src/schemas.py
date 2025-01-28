from pydantic import BaseModel, conint, confloat

class HeartDiseaseRequest(BaseModel):
    age: conint(gt=0)  
    sex: conint(ge=0, le=1)  
    cp: conint(ge=1, le=4)  
    trestbps: conint(gt=0)  
    chol: conint(gt=0)  
    fbs: conint(ge=0, le=1)  
    restecg: conint(ge=0, le=2)  
    thalach: conint(gt=0)  
    exang: conint(ge=0, le=1)  
    oldpeak: confloat(ge=-0.1, le=6.2)  
    slope: conint(ge=1, le=3)
