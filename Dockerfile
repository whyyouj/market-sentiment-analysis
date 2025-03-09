FROM apache/airflow:2.10.5

# Set the working directory
WORKDIR /opt/airflow

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your DAGs and any other necessary files
COPY dags/ /opt/airflow/dags/