import re
import os
import shutil
import zipfile
import requests
from pydicom.filereader import dcmread
from pynetdicom import AE, StoragePresentationContexts
from typing import Union
from fastapi import FastAPI, status, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from classifier import preprocess
from typing import Annotated
from sqlmodel import Field, Session, SQLModel, create_engine, select
from datetime import datetime

class History(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    message: str = Field(index=True)
    success: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

def create_zip_from_output(output_folder, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=output_folder)
                zipf.write(file_path, arcname)

def send_dicom(file_paths, session):
    ae = AE(ae_title="HIGIANULOCAL")
    ae.requested_contexts = StoragePresentationContexts
    assoc = ae.associate("10.0.3.32", 11112, ae_title="HIGIANULOCAL")
    if assoc.is_established:
        for file_path in file_paths:
            print(file_path)
            dataset = dcmread(file_path, force=True)
            status = assoc.send_c_store(dataset)
            if status and status.Status == 0x0000:
                session.add(History(filename=file_path, message="Arquivo DICOM enviado com sucesso", success=True))
                session.commit()
            else:
                raise Exception(f"Falha ao enviar o arquivo DICOM: {status}")
    else:
        raise Exception("Falha ao estabelecer conexão DICOM")
    assoc.release()

@app.get("/send-dicom")
async def send_dicom_route(session: SessionDep):
    send_dicom(['sr_example.dcm'], session)
    return {"message": "Arquivos DICOM enviados com sucesso"}

@app.get("/")
async def read_item(session: SessionDep, q: Union[str, None] = None):
    if q is None:
        return JSONResponse(content={"error": "No query provided"}, status_code=status.HTTP_400_BAD_REQUEST)
    
    response = requests.get(q)
    if "Content-Disposition" in response.headers.keys():
            filename = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
    else:
        filename = q.split("/")[-1]

    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        
        try:
            edited_image_paths = preprocess(input_file=filename)
        except Exception as e:
            session.add(History(filename=filename, message=f"Erro ao processar o arquivo: {str(e)}", success=False))
            session.commit()
            return JSONResponse(
                content={"error": f"Erro ao processar o arquivo: {str(e)}"},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # session.add(History(filename=filename, message="Arquivo processado com sucesso!", success=True))
        # session.commit()
        send_dicom(edited_image_paths, session)
        # os.remove(filename)
        create_zip_from_output("output", filename)
        # shutil.rmtree("zip")
        # shutil.rmtree("output")
        return JSONResponse(
            content={
                "filename": filename,
                "message": "Arquivo processado com sucesso!"
            }, 
            status_code=status.HTTP_200_OK
        )
    else:
        session.add(History(filename=filename, message="Falha ao baixar o arquivo", success=False))
        session.commit()
        return JSONResponse(
            content={"error": "Falha ao baixar o arquivo"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )