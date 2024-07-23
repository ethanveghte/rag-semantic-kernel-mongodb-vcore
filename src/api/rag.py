# Import the required libraries
import time
import json
import uuid
import urllib 
import ijson
import zipfile
from dotenv import dotenv_values
from openai import AzureOpenAI
from azure.core.exceptions import AzureError
from azure.cosmos import PartitionKey, exceptions
from time import sleep
import gradio as gr

# Cosmos DB imports
from azure.cosmos import CosmosClient
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# Load configuration
env_name = "sample_env_file.env"
config = dotenv_values(env_name)

cosmos_conn = config['cosmos_uri']
cosmos_key = config['cosmos_key']
cosmos_database = config['cosmos_database_name']
cosmos_collection = config['cosmos_collection_name']
cosmos_vector_property = config['cosmos_vector_property_name']
comsos_cache_db = config['cosmos_cache_database_name']
cosmos_cache = config['cosmos_cache_collection_name']

# Create the Azure Cosmos DB for NoSQL async client for faster data loading
cosmos_client = CosmosClient(url=cosmos_conn, credential=cosmos_key)

openai_endpoint = config['openai_endpoint']
openai_key = config['openai_key']
openai_api_version = config['openai_api_version']
openai_embeddings_deployment = config['openai_embeddings_deployment']
openai_embeddings_dimensions = int(config['openai_embeddings_dimensions'])
openai_completions_deployment = config['openai_completions_deployment']

# Movies file url
storage_file_url = config['storage_file_url']
load_data = config['load_data']

# Create the OpenAI client
openai_client = AzureOpenAI(azure_endpoint=openai_endpoint, api_key=openai_key, api_version=openai_api_version)

db = cosmos_client.create_database_if_not_exists(cosmos_database)

# Create the vector embedding policy to specify vector details
vector_embedding_policy = {
    "vectorEmbeddings": [ 
        { 
            "path":"/" + cosmos_vector_property,
            "dataType":"float32",
            "distanceFunction":"dotproduct",
            "dimensions":openai_embeddings_dimensions
        }, 
    ]
}

# Create the vector index policy to specify vector details
indexing_policy = {
    "vectorIndexes": [ 
        {
            "path": "/"+cosmos_vector_property, 
            "type": "quantizedFlat" 
        }
    ]
} 

# Create the data collection with vector index (note: this creates a container with 10000 RUs to allow fast data load)
try:
    movies_container = db.create_container_if_not_exists(id=cosmos_collection, 
                                                  partition_key=PartitionKey(path='/id'), 
                                                  indexing_policy=indexing_policy,
                                                  vector_embedding_policy=vector_embedding_policy) 
    print('Container with id \'{0}\' created'.format(movies_container.id)) 

except exceptions.CosmosHttpResponseError: 
    raise 

# Create the cache collection with vector index
try:
    cache_container = db.create_container_if_not_exists(id=cosmos_cache, 
                                                  partition_key=PartitionKey(path='/id'), 
                                                  indexing_policy=indexing_policy,
                                                  vector_embedding_policy=vector_embedding_policy) 
    print('Container with id \'{0}\' created'.format(cache_container.id)) 

except exceptions.CosmosHttpResponseError: 
    raise

# from tenacity import retry, stop_after_attempt, wait_random_exponential
# @retry(wait=wait_random_exponential(min=2, max=300), stop=stop_after_attempt(20))
def generate_embeddings(text):
    print('creating embeddings')
    response = openai_client.embeddings.create(
        input=text,
        model=openai_embeddings_deployment,
        #dimensions=openai_embeddings_dimensions
    )
    print('embeddings created')
    embeddings = response.model_dump()
    return embeddings['data'][0]['embedding']

# Unzip the data file
with zipfile.ZipFile("./MovieLens-4489-256D.zip", 'r') as zip_ref: 
    zip_ref.extractall("/Data")
zip_ref.close()
# Load the data file
data =[]
with open('data/MovieLens-4489-256D.json', 'r') as d:
    data = json.load(d)
# View the number of documents in the data (4489)
len(data) 

async def generate_vectors(items, vector_property):
    for item in items:
        vectorArray = await generate_embeddings(item['overview'])
        item[vector_property] = vectorArray
    return items

async def insert_data():
    start_time = time.time()  # Record the start time
    
    counter = 0
    tasks = []
    max_concurrency = 20  # Adjust this value to control the level of concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    print("Starting doc load, please wait...")
    
    def upsert_item_sync(obj):
        movies_container.upsert_item(body=obj)
    
    async def upsert_object(obj):
        nonlocal counter
        async with semaphore:
            await asyncio.get_event_loop().run_in_executor(None, upsert_item_sync, obj)
            # Progress reporting
            counter += 1
            if counter % 100 == 0:
                print(f"Sent {counter} documents for insertion into collection.")
    
    for obj in data:
        tasks.append(asyncio.create_task(upsert_object(obj)))
    
    # Run all upsert tasks concurrently within the limits set by the semaphore
    await asyncio.gather(*tasks)
    
    end_time = time.time()  # Record the end time
    duration = end_time - start_time  # Calculate the duration
    print(f"All {counter} documents inserted!")
    print(f"Time taken: {duration:.2f} seconds ({duration:.3f} milliseconds)")

# Run the async function
if load_data:
    insert_data()
 
# Perform a vector search on the Cosmos DB container
def vector_search(container, vectors, similarity_score=0.02, num_results=5):
    # Execute the query
    results = container.query_items(
        query= '''
        SELECT TOP @num_results  c.overview, VectorDistance(c.vector, @embedding) as SimilarityScore 
        FROM c
        WHERE VectorDistance(c.vector,@embedding) > @similarity_score
        ORDER BY VectorDistance(c.vector,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score}
        ],
        enable_cross_partition_query=True, populate_query_metrics=True)
    results = list(results)
    # Extract the necessary information from the results
    formatted_results = []
    for result in results:
        score = result.pop('SimilarityScore')
        formatted_result = {
            'SimilarityScore': score,
            'document': result
        }
        formatted_results.append(formatted_result)

    # #print(formatted_results)
    metrics_header = dict(container.client_connection.last_response_headers)
    #print(json.dumps(metrics_header,indent=4))
    return formatted_results

def get_chat_history(container, completions=3):
    results = container.query_items(
        query= '''
        SELECT TOP @completions *
        FROM c
        ORDER BY c._ts DESC
        ''',
        parameters=[
            {"name": "@completions", "value": completions},
        ], enable_cross_partition_query=True)
    results = list(results)
    return results

def generate_completion(user_prompt, vector_search_results, chat_history):
    
    system_prompt = '''
    You are an intelligent assistant for movies. You are designed to provide helpful answers to user questions about movies in your database.
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses, but still friendly.
        - Only answer questions related to the information provided below. Provide at least 3 candidate movie answers in a list.
        - Write two lines of whitespace between each answer in the list.
    '''

    # Create a list of messages as a payload to send to the OpenAI Completions API

    # system prompt
    messages = [{'role': 'system', 'content': system_prompt}]

    #chat history
    for chat in chat_history:
        messages.append({'role': 'user', 'content': chat['prompt'] + " " + chat['completion']})
    
    #user prompt
    messages.append({'role': 'user', 'content': user_prompt})

    #vector search results
    for result in vector_search_results:
        messages.append({'role': 'system', 'content': json.dumps(result['document'])})

    print("Messages going to openai", messages)
    # Create the completion
    response = openai_client.chat.completions.create(
        model = openai_completions_deployment,
        messages = messages,
        temperature = 0.1
    )    
    return response.model_dump()

def chat_completion(cache_container, movies_container, user_input):
    print("starting completion")
    # Generate embeddings from the user input
    user_embeddings = generate_embeddings(user_input)
    print('generated embeddings')
    # Query the chat history cache first to see if this question has been asked before
    cache_results = get_cache(container = cache_container, vectors = user_embeddings, similarity_score=0.99, num_results=1)
    print('retrieved cache results')
    if len(cache_results) > 0:
        print("Cached Result\n")
        return cache_results[0]['completion'], True
        
    else:
        #perform vector search on the movie collection
        print("New result\n")
        search_results = vector_search(movies_container, user_embeddings)

        print("Getting Chat History\n")
        #chat history
        chat_history = get_chat_history(cache_container, 3)
        #generate the completion
        print("Generating completions \n")
        completions_results = generate_completion(user_input, search_results, chat_history)

        print("Caching response \n")
        #cache the response
        cache_response(cache_container, user_input, user_embeddings, completions_results)

        print("\n")
        # Return the generated LLM completion
        return completions_results['choices'][0]['message']['content'], False
    
def cache_response(container, user_prompt, prompt_vectors, response):
    # Create a dictionary representing the chat document
    chat_document = {
        'id':  str(uuid.uuid4()),  
        'prompt': user_prompt,
        'completion': response['choices'][0]['message']['content'],
        'completionTokens': str(response['usage']['completion_tokens']),
        'promptTokens': str(response['usage']['prompt_tokens']),
        'totalTokens': str(response['usage']['total_tokens']),
        'model': response['model'],
        'vector': prompt_vectors
    }
    # Insert the chat document into the Cosmos DB container
    container.create_item(body=chat_document)
    print("item inserted into cache.", chat_document)

# Perform a vector search on the Cosmos DB container
def get_cache(container, vectors, similarity_score=0.0, num_results=5):
    # Execute the query
    results = container.query_items(
        query= '''
        SELECT TOP @num_results *
        FROM c
        WHERE VectorDistance(c.vector,@embedding) > @similarity_score
        ORDER BY VectorDistance(c.vector,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@similarity_score", "value": similarity_score},
        ],
        enable_cross_partition_query=True, populate_query_metrics=True)
    results = list(results)
    #print(results)
    return results

chat_history = []
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Cosmic Movie Assistant")
    
    msg = gr.Textbox(label="Ask me about movies in the Cosmic Movie Database!")
    clear = gr.Button("Clear")

    def user(user_message, chat_history):
        # Create a timer to measure the time it takes to complete the request
        start_time = time.time()
        # Get LLM completion
        response_payload, cached = chat_completion(cache_container, movies_container, user_message)
        # Stop the timer
        end_time = time.time()
        elapsed_time = round((end_time - start_time) * 1000, 2)
        response = response_payload
        print(response_payload)
        # Append user message and response to chat history
        details = f"\n (Time: {elapsed_time}ms)"
        if cached:
            details += " (Cached)"
        chat_history.append([user_message, response_payload + details])
        
        return gr.update(value=""), chat_history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)

    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
demo.launch(debug=True)

# Be sure to run this cell to close or restart the Gradio demo
demo.close()
