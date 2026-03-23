from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Union, Generator
from sqlmodel import Field, SQLModel, Session, create_engine, select

app = FastAPI()

# Модель для валидации валюты (не хранится в БД)
class Valute(BaseModel):
    name: str
    time_n_date: Union[str, None] = ""
    value: float

# Модель термина (хранится в БД)
class Term(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = Field(default="")

# Подключение к БД MySQL
DATABASE_URL = "mysql+mysqlconnector://mchsdo:mchsdo_pass@database:3306/mchsdo"
engine = create_engine(DATABASE_URL, echo=True)

# Автоматическая миграция таблиц при старте
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Сессия для работы с БД
def get_session() -> Generator:
    with Session(engine) as session:
        yield session

@app.get("/terms", response_model=list[Term])
def get_all_terms(session: Session = Depends(get_session)):
    return session.exec(select(Term)).all()

@app.get("/terms/{term_name}", response_model=Term)
def get_term(term_name: str, session: Session = Depends(get_session)):
    term = session.exec(select(Term).where(Term.name == term_name)).first()
    if not term:
        raise HTTPException(status_code=404, detail="Term not found")
    return term

@app.post("/terms", response_model=Term)
def create_term(term: Term, session: Session = Depends(get_session)):
    exists = session.exec(select(Term).where(Term.name == term.name)).first()
    if exists:
        raise HTTPException(status_code=400, detail="Term already exists")
    session.add(term)
    session.commit()
    session.refresh(term)
    return term

@app.put("/terms/{term_name}", response_model=Term)
def update_term(term_name: str, term_data: Term, session: Session = Depends(get_session)):
    term = session.exec(select(Term).where(Term.name == term_name)).first()
    if not term:
        raise HTTPException(status_code=404, detail="Term not found")
    term.description = term_data.description
    session.commit()
    session.refresh(term)
    return term


@app.delete("/terms/{term_name}")
def delete_term(term_name: str, session: Session = Depends(get_session)):
    term = session.exec(select(Term).where(Term.name == term_name)).first()
    if not term:
        raise HTTPException(status_code=404, detail="Term not found")
    session.delete(term)
    session.commit()
    return {"result": "deleted successfully"}


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/author')
def read_about():
    from datetime import datetime
    import locale
    try:
        locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
    except:
        pass
    return {'author': "Aleksandr", "datetime": datetime.now().strftime("%A, %d.%m.%Y, %H:%M")}

@app.get("/valute/{valute_id}")
def read_valute(valute_id: str):
    return {"valute_id": valute_id}

@app.put("/valute/{valute_id}")
def update_valute(valute_id: str, _valute: Valute):
    return {"valute_name": _valute.name, "valute_val": _valute.value}
