import json
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from confluent_kafka import Producer, Consumer, KafkaException
import pandas as pd
import time
from scripts.feature_transformations import calculate_sentiment_score, ema_price, calculate_rsi, add_bollinger_bands, exp_weighting_sentiment_score
from modelling.utils import load_scaler, predict, prepare_data_for_analysis, analyse_feature_importance
import pytz

# fetch configuration file stored as an airflow variable
DAG_CONFIG = Variable.get("gold_prediction_dag_config", deserialize_json=True)

# define date range for dag run
START_DATE = datetime(2015, 11, 16).replace(tzinfo=pytz.UTC)
END_DATE = datetime(2019, 2, 1).replace(tzinfo=pytz.UTC)

def get_current_date():
    """Get the current date in the format YYYY-MM-DD"""
    return datetime.now().strftime('%Y-%m-%d')

def check_execution_date(**kwargs):
    """Check if the execution date is within our desired range"""
    execution_date = kwargs['execution_date']
    if START_DATE <= execution_date <= END_DATE:
        return True
    else:
        print(f"Execution date {execution_date} is outside our desired range. Skipping this run.")
        return False

def create_kafka_producer():
    """Create and return a Kafka producer instance"""
    kafka_config = {
        'bootstrap.servers': Variable.get('kafka_bootstrap_servers'),
        'client.id': 'gold-prediction-producer'
    }
    return Producer(kafka_config)

def create_kafka_consumer(group_id, topic):
    """Create and return a Kafka consumer instance"""
    kafka_config = {
        'bootstrap.servers': Variable.get('kafka_bootstrap_servers'),
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(kafka_config)
    consumer.subscribe([topic])
    return consumer

def get_mysql_extraction_query(cols_to_extract, date_format_func, mysql_table, execution_date):
    cols = ', '.join(cols_to_extract)
    match date_format_func:
        case "standard_date_format":
            query = f"""
            SELECT {cols} FROM {mysql_table}
            WHERE Date = '{execution_date}'
            """
        case "mdy_format":
            dt = datetime.strptime(execution_date, '%Y-%m-%d')
            formatted_date = f"{dt.month}/{dt.day}/{dt.year}"
            query = f"""
            SELECT {cols} FROM {mysql_table}
            WHERE Date = '{formatted_date}'
            """
        case "dmy_format":
            dt = datetime.strptime(execution_date, '%Y-%m-%d')
            formatted_date = f"{dt.day}/{dt.month}/{dt.year}"
            query = f"""
            SELECT {cols} FROM {mysql_table}
            WHERE Dates = '{formatted_date}'
            """
        case "year_month_format":
            dt = datetime.strptime(execution_date, '%Y-%m-%d')
            period = f"M{'0' + str(dt.month) if dt.month < 10 else str(dt.month)}"
            query = f"""
            SELECT {cols} FROM {mysql_table}
            WHERE Year = {dt.year} and Period = '{period}'
            """
        case _:
            query = ""
    
    return query

def extract_from_mysql(dataset_config, execution_date):
    """Extract data from MySQL for a specific dataset and the given date"""
    dataset_name = dataset_config['name']
    mysql_table = dataset_config['mysql_table']
    date_format_func = dataset_config['date_format_function']
    cols_to_extract = dataset_config['columns_to_extract']
    
    # apply custom date formatting function for data source to get the correct mysql query
    sql_query = get_mysql_extraction_query(cols_to_extract, date_format_func, mysql_table, execution_date)
    
    # connect to MySQL
    mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
    
    try:
        # execute query
        results = mysql_hook.get_pandas_df(sql_query)
        
        # if no data found for the date, create null record
        result_dict = {}
        if results.empty:
            print(f"[MySQL] No data found for {dataset_name} on {execution_date}: Creating null record")
            # create a record with null values
            cols_string = ', '.join(cols_to_extract)
            columns = mysql_hook.get_pandas_df(f"SELECT {cols_string} FROM {mysql_table} LIMIT 0").columns
            null_record = {col: None for col in columns}
            result_dict = null_record
        else:
            print(f"[MySQL] Successfully extracted {dataset_name} data for {execution_date}")
            result_dict = results.iloc[0].to_dict()
        
        return result_dict
    
    except Exception as e:
        print(f"Error extracting {dataset_name} data for {execution_date}: {str(e)}")
        raise

def extract_from_api(dataset_config, execution_date):
    """Extract data from APIs (for real yields, VIX, DXY)"""
    import yfinance as yf
    import pandas_datareader as pdr

    dataset_name = dataset_config['name']
    
    try:
        result_dict = {}
            
        if dataset_name == "real_yields":
            df = pdr.DataReader('DFII10', 'fred', start=execution_date, end=execution_date)
            if not df.empty:
                result_dict = {'DFII10': float(df.iloc[0]['DFII10'])}
        elif dataset_name == "vix":
            df = yf.download('^VIX', start=execution_date, end=execution_date)
            if not df.empty:
                result_dict = {'VIX': float(df.iloc[0]['Close'])}
        elif dataset_name == "usd_rates":
            df = yf.download('DX-Y.NYB', start=execution_date, end=execution_date)
            if not df.empty:
                result_dict = {'DXY': float(df.iloc[0]['Close'])}

        if not result_dict:
            print(f"[API] No data found for {dataset_name} on {execution_date}: Creating null record")
            # create a record with null values
            null_record = {col: None for col in dataset_config['columns_to_extract']}
            result_dict = null_record
        else:
            print(f"[API] Successfully extracted {dataset_name} data for {execution_date}")
        
        return result_dict
    
    except Exception as e:
        print(f"Error extracting {dataset_name} data for {execution_date}: {str(e)}")
        raise
    
def extract_data(dataset_config, execution_date, **kwargs):
    """Extract data from MySQL or API based on the dataset configuration"""
    data_extraction_method = dataset_config['extraction_method']
    dataset_name = dataset_config['name']

    if data_extraction_method == 'mysql':
        # Extract from MySQL
        result_dict = extract_from_mysql(dataset_config, execution_date)
    elif data_extraction_method == 'api':
        # Extract from API
        result_dict = extract_from_api(dataset_config, execution_date)

    # convert to json and send to kafka
    data_json = json.dumps(result_dict).encode('utf-8')
    
    try:
        # produce to Kafka
        producer = create_kafka_producer()
        producer.produce(
            dataset_config['kafka_topic'],
            key=dataset_name,
            value=data_json
        )
        producer.flush()
        
        print(f"[Kafka] Successfully produced {dataset_name} data to Kafka for {execution_date}")
        return True
    
    except Exception as e:
        print(f"Error producing {dataset_name} data to Kafka for {execution_date}: {str(e)}")
        raise

def transform_data(dataset_config, **kwargs):
    """Consume data from Kafka and apply transformations"""
    dataset_name = dataset_config['name']
    kafka_topic = dataset_config['kafka_topic']
    transformation_function = dataset_config['transformation_function']
    
    # create consumer
    consumer = create_kafka_consumer(f"{dataset_name}-consumer-group", kafka_topic)
    
    try:
        # poll for message
        msg = consumer.poll(timeout=30.0)
        
        if msg is None:
            print(f"[Kafka] No message received for {dataset_name}")
            return None
        
        if msg.error():
            raise KafkaException(msg.error())
        
        # parse the message
        data_json = msg.value().decode('utf-8')
        data = json.loads(data_json)
        
        # apply specific transformation based on the data source
        if transformation_function == 'price_transform':
            data['Price'] = data['Price'].replace(',', '').astype(float)

        elif transformation_function == 'calculate_sentiment_score':
            # use hugging face transformer to compute sentiment score of news
            data['Sentiment_Score'] = calculate_sentiment_score(data['News'])
            del data['News']
        elif transformation_function == 'rename_key':
            # rename key to fit data source
            data['CPI'] = data['Value']
            del data['Value']
        
        # store the transformed data in XCom
        kwargs['ti'].xcom_push(key=dataset_name, value=json.dumps(data))
        print(f"[Python] Successfully transformed {dataset_name} data")
        
        return True
    
    except Exception as e:
        print(f"Error transforming {dataset_name} data: {str(e)}")
        raise
    
    finally:
        # close the consumer
        consumer.close()

def combine_datasets(**kwargs):
    """Combine all the transformed datasets into a single row for BigQuery"""
    ti = kwargs['ti']
    datasets_config = DAG_CONFIG['datasets']
    execution_date = kwargs['execution_date'].strftime('%Y-%m-%d')
    
    # dictionary to hold the combined data
    combined_data = {'Date': execution_date}
    
    # retrieve each dataset from XCom
    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        dataset_json = ti.xcom_pull(task_ids=f"transform_{dataset_name}", key=dataset_name)
        
        if dataset_json:
            data = json.loads(dataset_json)
            
            # add all values from this data source to the combined data
            for k, v in data.items():
                combined_data[k] = v     
    
    # store the combined data in XCom
    ti.xcom_push(key='combined_data', value=json.dumps(combined_data))
    print(f"[Python] Successfully combined all datasets for {execution_date}")
    
    return True

def get_historical_data(date_str, bq_hook, bq_config, max_window_size):
    """Get historical data from BigQuery for feature engineering"""
    # calculate the start date for historical data
    start_date = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=max_window_size)).strftime('%Y-%m-%d')
    
    # query historical data
    project_id = bq_config['project_id']
    dataset_id = bq_config['dataset_id']
    gold_market_data_table = bq_config['gold_market_data_table']

    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{gold_market_data_table}
    WHERE Date >= '{start_date}' AND Date < '{date_str}'
    ORDER BY Date ASC
    """
    
    # execute the query
    result = bq_hook.get_pandas_df(query, use_legacy_sql=False)
    
    return result if not result.empty else pd.DataFrame()

def apply_feature_engineering(**kwargs):
    """Apply feature engineering to the combined dataset"""
    ti = kwargs['ti']

    # connect to BigQuery and get config
    bq_hook = BigQueryHook(gcp_conn_id='bigquery_default')
    bq_config = DAG_CONFIG['bigquery']

    # create a dataframe for the current day's combined data
    current_data = ti.xcom_pull(task_ids='combine_datasets', key='combined_data')
    current_data = json.loads(current_data)
    current_df = pd.DataFrame([current_data])

    # get historical data for feature engineering
    max_window_size = max(DAG_CONFIG['window_sizes'].values())
    hist_df = get_historical_data(current_data['Date'], bq_hook, bq_config, max_window_size)
    
    # convert date string to datetime in current data
    if 'Date' in current_df.columns:
        current_df['Date'] = pd.to_datetime(current_df['Date'])
    
    if not hist_df.empty:
        # convert date string to datetime in historical data
        if 'Date' in hist_df.columns:
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        
        # combine historical data with current data
        combined_df = pd.concat([hist_df, current_df]).reset_index(drop=True)
    else:
        combined_df = current_df
    
    # forward fill missing values
    combined_df = combined_df.ffill()
    
    # apply feature engineering functions
    # exponential moving average for price
    combined_df["EMA30"], combined_df["EMA252"] = ema_price(combined_df[["Price"]])

    # RSI
    combined_df['RSI'] = calculate_rsi(combined_df['Price'])

    # Bollinger Bands and Spread
    combined_df['Band_Spread'] = add_bollinger_bands(combined_df['Price'])

    # obtain the current day's data with all features
    full_current_data = combined_df.iloc[-1:].to_dict('records')[0]
    
    # exponential weighting for news sentiment score
    full_current_data['Exponential_Weighted_Score'] = exp_weighting_sentiment_score(bq_config, bq_hook, current_data['Sentiment_Score'], current_data['Date'])

    # store the full data in XCom
    ti.xcom_push(key='full_current_data', value=json.dumps(full_current_data))
    print(f"[apply_feature_engineering] Feature engineering completed for {current_data['Date']}")

    return True

def load_to_bigquery(**kwargs):
    """Load the processed data to BigQuery"""
    ti = kwargs['ti']
    bigquery_config = DAG_CONFIG['bigquery']
    
    project_id = bigquery_config['project_id']
    dataset_id = bigquery_config['dataset_id']
    gold_market_data_table = bigquery_config['gold_market_data_table']
    
    # get current day's data after feature engineering
    full_current_data = ti.xcom_pull(task_ids='apply_feature_engineering', key='full_current_data')
    full_current_data = json.loads(full_current_data)
    
    # connect to BigQuery
    bq_hook = BigQueryHook(gcp_conn_id='bigquery_default')
    
    # get BigQueryHook client and table
    client = bq_hook.get_client()
    table_ref = f"{project_id}.{dataset_id}.{gold_market_data_table}"
    table = client.get_table(table_ref)
    
    # insert the current day's data as a row
    errors = client.insert_rows_json(table, [full_current_data])
    
    if errors:
        raise Exception(f"[load_to_bigquery] Errors inserting data for {full_current_data['Date']} into {table_ref}: {errors}")
    else:
        print(f"[load_to_bigquery] Successfully inserted data for {full_current_data['Date']} into {table_ref}")
    
    return True

def check_model_training_status(**kwargs):
    """Check if models are being retrained, and wait if they are"""
    bq_config = DAG_CONFIG['bigquery']
    project_id = bq_config['project_id']
    dataset_id = bq_config['dataset_id']
    model_training_status_table = bq_config['model_training_status_table']
    
    bq_hook = BigQueryHook(gcp_conn_id='bigquery_default')
    query = f"""
    SELECT training_status
    FROM {project_id}.{dataset_id}.{model_training_status_table}
    WHERE status_id = 'model_training_status'
    """
    
    max_retries = 10
    retry_interval = 120  # 2 minutes
    
    for attempt in range(max_retries):
        status_df = bq_hook.get_pandas_df(query, use_legacy_sql=False)
        
        if status_df.empty:
            raise ValueError("Could not retrieve model training status")
        
        current_status = status_df.iloc[0]['training_status']
        
        if current_status != 'in_progress':
            print(f"[check_model_training_status] Models not in training, proceeding with DAG execution.")
            return True
        
        # If models are being trained, log and wait
        print(f"[check_model_training_status] Models are currently being trained (status check attempt {attempt+1}/{max_retries})")
        print(f"[check_model_training_status] Waiting {retry_interval} seconds before checking again...")
        time.sleep(retry_interval)
    
    # If we've waited too long, proceed anyway
    print("[check_model_training_status] Maximum wait time exceeded. Proceeding with DAG execution.")
    return True

def get_pred_sequence(execution_date, seq_length):
    start_date = (datetime.strptime(execution_date, '%Y-%m-%d') - timedelta(days=seq_length)).strftime('%Y-%m-%d')
    
    # query historical data
    bq_config = DAG_CONFIG['bigquery']
    project_id = bq_config['project_id']
    dataset_id = bq_config['dataset_id']
    gold_market_data_table = bq_config['gold_market_data_table']

    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{gold_market_data_table}
    WHERE Date > '{start_date}' AND Date <= '{execution_date}'
    ORDER BY Date ASC
    """
    
    # execute the query
    bq_hook = BigQueryHook(gcp_conn_id='bigquery_default')
    result = bq_hook.get_pandas_df(query, use_legacy_sql=False)

    return result if not result.empty else pd.DataFrame()

def model_predict(model_config, **kwargs):
    import tensorflow as tf

    ti = kwargs['ti']
    execution_date = kwargs['execution_date']
    
    # Extract model details
    model_name = model_config['name']
    features = model_config['features']
    seq_length = model_config['seq_length']
    
    print(f"[make_predictions] Making predictions with model: {model_name}")
    
    # Load the current data from previous task
    pred_sequence = get_pred_sequence(execution_date, seq_length)
    
    if pred_sequence.empty:
        raise ValueError("[make_predictions] No prediction sequence received")
    
    # Load the model
    model_path = os.path.join(os.getcwd(), 'trained_models', f"{model_name}.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[make_predictions] Model {model_name} not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    # Load the scaler_dict
    scaler_dict = load_scaler()
    
    # Make predictions
    prediction = predict(model, scaler_dict, pred_sequence, features)
    
    print(f"[make_predictions] Model {model_name} prediction for {execution_date}: {prediction}")
    
    # store the prediction in XCom
    ti.xcom_push(key=f'prediction_{model_name}', value=prediction)

    return True

def combine_predictions(**kwargs):
    """Combine all the predictions into a single row for BigQuery"""
    ti = kwargs['ti']
    models_config = DAG_CONFIG['models']
    execution_date = kwargs['execution_date'].strftime('%Y-%m-%d')
    pred_date = (datetime.strptime(execution_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # dictionary to hold the combined data
    combined_data = {'Date': pred_date}
    
    # retrieve each prediction from XCom
    for model_config in models_config:
        model_name = model_config['name']
        model_pred = ti.xcom_pull(task_ids=f"{model_name}_predict", key=f"prediction_{model_name}")
            
        # add model prediction to the combined data
        combined_data[model_name] = model_pred     
    
    # store the combined data in XCom
    ti.xcom_push(key='combined_pred', value=json.dumps(combined_data))
    print(f"Successfully combined all model predictions for {execution_date}")
    
    return True

def store_predictions(**kwargs):
    ti = kwargs['ti']
    
    # Get the prediction results
    prediction_result = ti.xcom_pull(task_ids="combine_predictions", key='combined_pred')
    prediction_result = json.loads(prediction_result)
    
    # Connect to BigQuery
    bq_config = DAG_CONFIG['bigquery']
    project_id = bq_config['project_id']
    dataset_id = bq_config['dataset_id']
    model_predictions_table = bq_config['model_predictions_table']
    
    bq_hook = BigQueryHook(gcp_conn_id='bigquery_default')
    client = bq_hook.get_client()
    table_ref = f"{project_id}.{dataset_id}.{model_predictions_table}"
    table = client.get_table(table_ref)
    
    # Insert predictions into BigQuery
    errors = client.insert_rows_json(table, [prediction_result])
    
    if errors:
        raise Exception(f"[store_predictions] Errors inserting predictions for {prediction_result['Date']}: {errors}")
    else:
        print(f"[store_predictions] Successfully stored predictions for {prediction_result['Date']}")
        
    return True

def model_analyse(model_config, **kwargs):
    import tensorflow as tf

    ti = kwargs['ti']

    # Fetch historical data for analysis
    execution_date = kwargs['execution_date'].strftime('%Y-%m-%d')
    bq_hook = BigQueryHook(gcp_conn_id='bigquery_default')
    bq_config = DAG_CONFIG['bigquery']
    project_id = bq_config['project_id']
    dataset_id = bq_config['dataset_id']
    gold_market_data_table = bq_config['gold_market_data_table']

    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{gold_market_data_table}
    ORDER BY Date ASC
    """

    result = bq_hook.get_pandas_df(query, use_legacy_sql=False)

    # Create new column for the actual price and drop rows with missing values
    result['Actual'] = result['Price'].shift(-1)
    result = result.dropna()

    # Load the scaler_dict
    scaler_dict = load_scaler()

    # Prepare data for feature importance computation
    features = model_config['features']
    seq_length = model_config['seq_length']
    X_test, y_test = prepare_data_for_analysis(scaler_dict, result, features, seq_length)

    # Load the model
    model_name = model_config['name']
    model_path = os.path.join(os.getcwd(), 'trained_models', f"{model_name}.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[model_analyse] Model {model_name} not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)

    # Analyse feature importances with model
    mean_importances = analyse_feature_importance(model, X_test, y_test, features)
    mean_importances['Date'] = execution_date
    mean_importances['Model'] = model_name

    # store the model's mean feature importances in XCom
    ti.xcom_push(key=f'mean_importances_{model_name}', value=json.dumps(mean_importances))
    print(f"[model_analyse] Mean feature importance scores for {model_name} on {execution_date} computed")

    return True

def store_analysis(model_config, **kwargs):
    # Fetch analysis results
    ti = kwargs['ti']
    model_name = model_config['name']
    mean_importances = ti.xcom_pull(task_ids=f"model_analyse_{model_name}", key=f'mean_importances_{model_name}')
    mean_importances = json.loads(mean_importances)

    # Connect to BigQuery
    bq_config = DAG_CONFIG['bigquery']
    project_id = bq_config['project_id']
    dataset_id = bq_config['dataset_id']
    feature_importances_table = bq_config['feature_importances_table']
    
    bq_hook = BigQueryHook(gcp_conn_id='bigquery_default')
    client = bq_hook.get_client()
    table_ref = f"{project_id}.{dataset_id}.{feature_importances_table}"
    table = client.get_table(table_ref)
    
    # Insert feature importances into BigQuery
    errors = client.insert_rows_json(table, [mean_importances])
    
    if errors:
        raise Exception(f"[store_predictions] Errors inserting predictions for {mean_importances['Date']}: {errors}")
    else:
        print(f"[store_predictions] Successfully stored predictions for {mean_importances['Date']}")
        
    return True

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# create the DAG
with DAG(
    'gold_price_prediction_pipeline',
    default_args=default_args,
    description='A DAG to gather and process gold-related financial data for ML predictions',
    schedule_interval='@daily',
    catchup=True,  # enable catchup to process historical data
    max_active_runs=1
) as dag:
    
    # start task
    start_task = DummyOperator(task_id='start')
    
    # check execution date task
    check_date_task = PythonOperator(
        task_id='check_execution_date',
        python_callable=check_execution_date,
        provide_context=True
    )
    
    # create dynamic tasks for each data source based on datasets config
    extract_tasks = []
    transform_tasks = []
    
    for dataset_config in DAG_CONFIG['datasets']:
        dataset_name = dataset_config['name']
        
        # extract task
        extract_task = PythonOperator(
            task_id=f"extract_{dataset_name}",
            python_callable=extract_from_mysql,
            op_kwargs={'dataset_config': dataset_config, 'execution_date': '{{ ds }}'},
            provide_context=True
        )
        extract_tasks.append(extract_task)
        
        # transform task
        transform_task = PythonOperator(
            task_id=f"transform_{dataset_name}",
            python_callable=transform_data,
            op_kwargs={'dataset_config': dataset_config},
            provide_context=True
        )
        transform_tasks.append(transform_task)
        
        # set extract + transform dependencies for each data source
        check_date_task >> extract_task >> transform_task
    
    # combine datasets task
    combine_data_task = PythonOperator(
        task_id='combine_datasets',
        python_callable=combine_datasets,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS # Only run if all upstream tasks succeed
    )

    # feature engineering task
    feature_eng_task = PythonOperator(
        task_id='apply_feature_engineering',
        python_callable=apply_feature_engineering,
        provide_context=True
    )
    
    # load to BigQuery task
    load_bq_task = PythonOperator(
        task_id='load_to_bigquery',
        python_callable=load_to_bigquery,
        provide_context=True
    )

    # check model training status task
    check_model_status_task = PythonOperator(
        task_id='check_model_training_status',
        python_callable=check_model_training_status,
        provide_context=True
    )

    # create dynamic prediction tasks for each model based on models config
    model_predict_tasks = []
    for model_config in DAG_CONFIG['models']:
        model_name = model_config['name']

        model_predict_task = PythonOperator(
            task_id=f"{model_name}_predict",
            python_callable=model_predict,
            op_kwargs={'model_config': model_config},
            provide_context=True
        )
        model_predict_tasks.append(model_predict_task)

    # combine model predictions task
    combine_pred_task = PythonOperator(
        task_id='combine_predictions',
        python_callable=combine_predictions,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS # Only run if all upstream tasks succeed
    )

    # store model predictions in BigQuery task
    store_pred_task = PythonOperator(
        task_id='store_predictions',
        python_callable=store_predictions,
        provide_context=True
    )

    # create dynamic analysis tasks for each model based on models config
    model_analysis_tasks = []
    store_analysis_tasks = []
    for model_config in DAG_CONFIG['models']:
        model_name = model_config['name']

        # analyse model's feature importances task
        model_analysis_task = PythonOperator(
            task_id=f"model_analyse_{model_name}",
            python_callable=model_analyse,
            op_kwargs={'model_config': model_config},
            provide_context=True
        )
        model_analysis_tasks.append(model_analysis_task)

        # store model's feature importances in BigQuery task
        store_analysis_task = PythonOperator(
            task_id=f"store_analysis_{model_name}",
            python_callable=store_analysis,
            op_kwargs={'model_config': model_config},
            provide_context=True
        )
        store_analysis_tasks.append(store_analysis_task)

        # set analyse + store analysis dependencies for each model
        store_pred_task >> model_analysis_task >> store_analysis_task
    
    # end task
    end_task = DummyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED  # Run even if some tasks are skipped
    )
    
    # set the final dependencies
    start_task >> check_date_task
    transform_tasks >> combine_data_task >> feature_eng_task >> load_bq_task >> check_model_status_task >> model_predict_tasks >> combine_pred_task >> store_pred_task
    store_analysis_tasks >> end_task