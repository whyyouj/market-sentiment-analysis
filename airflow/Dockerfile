FROM apache/airflow:2.10.5

# switch to airflow user
USER airflow

# copy and install python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt