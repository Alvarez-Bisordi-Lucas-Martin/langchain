# Version
- Python 3.12


# Configuracion Inicial
- Cree un .env utilizando las variables del archivo envtemplate.

- Cree un entorno virtual (venv) con la version de python especificada.

- Instale las librerias necesarias con sus ultimas versiones:
``` shell
pip install python-dotenv

pip install langchain
pip install langchain-google-genai
pip install langchain-community
pip install langchain-postgres

pip install huggingface_hub

pip install pypdf

pip install psycopg2-binary
pip install pgvector
pip install faiss-cpu
```

- O bien ejecute el siguiente comando:
``` shell
pip install -r requirements.txt
```


# Comandos de Ejecucion
``` shell
python main.py
```


# Credenciales de Gemini
- Token: https://aistudio.google.com/app/apikey

- Consumo: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas


# Credenciales de Llama
- Modelo: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

- Token: https://huggingface.co/settings/tokens

- Consumo: https://huggingface.co/settings/billing
