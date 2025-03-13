import re
import requests
from typing import Union
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from classifier import main

app = FastAPI()

@app.get("/")
async def read_item(q: Union[str, None] = None):
    if q is None:
        return JSONResponse(content={"error": "No query provided"}, status_code=status.HTTP_400_BAD_REQUEST)
    
    response = requests.get(q)

    if response.status_code == 200:
        # Extrai o nome do arquivo
        if "Content-Disposition" in response.headers.keys():
            filename = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
        else:
            filename = q.split("/")[-1]
        
        # Salva o arquivo localmente
        with open(filename, 'wb') as file:
            file.write(response.content)
        
        # Chama a função main com o arquivo baixado como parâmetro dcm
        try:
            main(dcm=filename)
        except Exception as e:
            return JSONResponse(
                content={"error": f"Erro ao processar o arquivo: {str(e)}"},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return JSONResponse(
            content={
                "filename": filename,
                "message": "Arquivo processado com sucesso!"
            }, 
            status_code=status.HTTP_200_OK
        )
    else:
        return JSONResponse(
            content={"error": "Falha ao baixar o arquivo"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )