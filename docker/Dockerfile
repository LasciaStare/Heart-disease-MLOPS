# Usar una imagen base de Python ligera
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias
COPY docker/requirements.txt .

# Instalar las dependencias sin cache para mantener la imagen ligera
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el contenido del directorio app al contenedor
COPY app/ app/

# Exponer el puerto 8000
EXPOSE 8000

# Comando para iniciar el servidor uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]