from langchain import SQLDatabase
# from langchain.prompts.prompt import PromptTemplate
# from service.library.base import SQLDatabaseChain
# from langchain.chains import LLMChain
from langchain import PromptTemplate, OpenAI, LLMChain
# from langchain.chat_models import ChatOpenAI
from service.utility.load_env import load_env_variables
from config.config import config
from concurrent.futures import ThreadPoolExecutor
import asyncio
import re
from service.enums.logger_category import LoggerCategory
from service.utility.logger import logger_info, logger_error
executor = ThreadPoolExecutor(max_workers=config.THREAD_POOL_EXECUTOR_MAX_WORKERS)



loaded_variables = load_env_variables()
db_user = loaded_variables["PRIVATE_DB_USERNAME"]
db_password = loaded_variables["PRIVATE_DB_PASSWORD"]
db_host = loaded_variables["PRIVATE_DB_HOST"]
db_port = loaded_variables["PRIVATE_DB_PORT"]
db_name = loaded_variables["PRIVATE_DB_NAME"]

custom_table_info = {
    "Track": """CREATE TABLE complaints (
        "Date received" DATE, 
        "Product" TEXT, 
        "SubProduct" TEXT, 
        "Issue" TEXT, 
        "Sub-issue" TEXT, 
        "ConsumerComplaintNarrative" TEXT, 
        "Company public response" TEXT, 
        "Company" TEXT, 
        "State" TEXT, 
        "ZIP code" TEXT, 
        "Tags" TEXT, 
        "ConsumerConsentProvided" TEXT, 
        "Submitted via" TEXT, 
        "Date sent to company" DATE, 
        "Company response to consumer" TEXT, 
        "TimelyResponse" BOOLEAN, 
        "ConsumerDisputed" TEXT, 
        "ComplaintID" INTEGER DEFAULT nextval('"complaints_ComplaintID_seq"'::regclass) NOT NULL
)

/*
2 rows from complaints table:
Date received   Product SubProduct      Issue   Sub-issue       ConsumerComplaintNarrative      Company public response Company State   ZIP code       Tags     ConsumerConsentProvided Submitted via   Date sent to company    Company response to consumer    TimelyResponse  ConsumerDisputed        ComplaintID
2017-11-22      Checking or savings account     Checking account        Managing an account     Problem making or receiving payments    I requested a Bill Payment from the Wells Fargo Online Bill Pay service to pay a property tax paymen    Company has responded to the consumer and the CFPB and chooses not to provide a public response WELLS FARGO & COMPANY   CA      94107   None    Consent provided        Web     2017-11-23      Closed with explanationTrue     N/A     2736411
2023-03-29      Credit reporting, credit repair services, or other personal consumer reports    Credit reporting        Incorrect information on your report    Information belongs to someone else     I am writing to bring to your attention an ongoing issue with my credit report. I have discovered th   None     EQUIFAX, INC.   CA      92620   None    Consent provided        Web     2023-03-29      Closed with explanation True    N/A     6762866
*/"""
}

db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
    include_tables=['complaints'],
    sample_rows_in_table_info=2
)

# Setup language model
# llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY, model_name=config.CHAT_MODEL)
llm = OpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY)

# Create database chain
# QUERY = """
# Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
# Use the following format:

# Question: "Question here"
# SQLQuery: "SQL Query to run"
# SQLResult: "Result of the SQLQuery"
# Answer: "Final answer here"

# Only use the following tables:

# {table_info}

# Question: {question}
# """

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for the table foobar, they really mean the employee table.

Question: {input}"""

# PROMPT = PromptTemplate(
#         input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
#     )

# Setup the database chain
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)
db_chain = LLMChain(llm=llm,
    prompt=PromptTemplate.from_template(_DEFAULT_TEMPLATE))

async def synchronous_function(question):
    def run_db_query():
        # return db_chain.run(dict(input=question, table_info=custom_table_info, dialect=db.dialect))
        return db_chain(dict(input=question, table_info=custom_table_info))

    loop = asyncio.get_event_loop()
    db_res = await loop.run_in_executor(None, run_db_query)
    logger_info("End of the db_chain, Successful************", LoggerCategory.Trail.value)    
    db_res_text = db_res['text']
    answer_match = re.search(r'Answer: (.+)', db_res_text)
    logger_info(f"DB_RES: {db_res}", LoggerCategory.Trail.value)
    logger_info(f"answer_match: {answer_match}", LoggerCategory.Trail.value)
    answer = ''
    if answer_match:
        answer = answer_match.group(1).strip()
    sql_query_match = re.search(r'SQLQuery:(.*?)SQLResult:', db_res_text, re.DOTALL)
    if sql_query_match:
        sql_query = sql_query_match.group(1).strip()
    # Define a regular expression pattern to extract SQLResult
    pattern = r'SQLResult:(.*?)(?=\nAnswer:)'

    # Find the SQLResult using the pattern
    match = re.search(pattern, db_res_text, re.DOTALL)

    sql_result = ""
    # Extract and print the SQLResult
    if match:
        sql_result = match.group(1).strip()
        logger_info(f"sql_result: {sql_result}", LoggerCategory.Trail.value)
    else:
        logger_info("Answer not found in the db_res_text.", LoggerCategory.Trail.value)
    return {"sql_result": sql_result , "query": sql_query, "answer": answer}

    
async def query_chain(question):
    max_retries = 3
    timeout = 20
    retries = 0

    while retries < max_retries:
        logger_info("Start of the API************", LoggerCategory.Trail.value)
        logger_info(f"question:{question} ", LoggerCategory.Trail.value)

        try:
            result = await asyncio.wait_for(synchronous_function(question), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger_error("Operation timed out. Retrying...")
            retries += 1

    logger_info("Max retries reached. Unable to complete the operation.", LoggerCategory.Trail.value)
    return None