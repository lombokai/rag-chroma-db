FROM python:3.10

WORKDIR /app

# install dependencies
COPY requirements/main.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r main.txt

# copy source code
COPY src/ /app/src
COPY run.py /app/run.py

# expose port 
EXPOSE 7860

# command to run
CMD ["python", "run.py"]
