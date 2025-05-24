# Use uma imagem base oficial, estável e enxuta (slim) do Python
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências primeiro (para otimizar o cache)
COPY requirements.txt ./requirements.txt

# Instala as dependências
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copia o resto dos arquivos do seu projeto para o diretório de trabalho
COPY . .

# Expõe a porta que o Streamlit vai usar
EXPOSE 8080

# Define o comando para executar a aplicação quando o container iniciar
# As flags --server.port e --server.headless são importantes para o Cloud Run
CMD ["streamlit", "run", "dashboard_edupulses.py", "--server.port=8080", "--server.headless=true"]