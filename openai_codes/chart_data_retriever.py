import openai, json
from config.config import config
from service.utility.chart_data_helper import prepare_chat_data
from service.utility.session_data_helper import get_session_data
from service.utility.db_message_saver import save_messages
from service.utility.db_chain_helper import query_chain
from service.utility.prompt_helper import get_active_prompt
from service.utility.logger import logger_info, logger_error
from service.utility.multiclient_helper import HelperMultiClient 
from service.enums.logger_category import LoggerCategory

def call_chat_function(message_list, function_descriptions, uuid):
    multiclient_res = HelperMultiClient.process_message_with_function(message_list, function_descriptions, uuid)
    return multiclient_res["response"]

async def get_prompt(partner):
    chart_data_prompt_type = config.CHART_DATA_SYSTEM_PROMPT_NAME
    fetch_prompt = await get_active_prompt(partner, chart_data_prompt_type)
    chart_data_prompt = fetch_prompt['detail']
    partner_id = fetch_prompt['partner_id']
    if chart_data_prompt is False:
        logger_error(f"YOU MUST ACTIVATE Chart data PROMPT......")
        raise RuntimeError(f"NO ACTIVE Chart data PROMPT")
    return {"prompt": chart_data_prompt,"partner_id": partner_id}

async def generate_chart_with_db(prompt_input, session_id, uuid, user_id, partner):
    db_res = await query_chain(prompt_input)
    prompt_db_call = await get_prompt(partner)
    system_prompt = prompt_db_call['prompt']
    partner_id = prompt_db_call['partner_id']
    chat_message = await chat_histroy(session_id)
    chat_message.append({"role": "user", "content": prompt_input})
    chat_message.append({"role": "user", "content": "Document Answer: " + db_res['sql_result']})
    chat_message.append({"role": "user", "content": "Summary: " + db_res['answer']})
    prepare_message = await process_message(chat_message,system_prompt)
    llm_res = call_chat_function(prepare_message, function_descriptions,uuid) 
    logger_info("llm_res :", LoggerCategory.Trail.value)
    logger_info(llm_res, LoggerCategory.Trail.value)
    logger_info("-------", LoggerCategory.Trail.value)

    filtered_response = handle_response(llm_res)
    chart_title = filtered_response.get("function_call", {}).get("arguments", {}).get("loc_title", {})
    chart_type = filtered_response.get("function_call", {}).get("arguments", {}).get("loc_type", {})    
    chart_data = filtered_response.get("function_call", {}).get("arguments", {}).get("loc_data", {}) 
    chosen_function_name = filtered_response.get("function_call", {}).get("name", "")
    chosen_function = eval(chosen_function_name)
    print("data",chart_data)
    detailed_data = {
    "title": chart_title,
    "type": chart_type,
    "data": chart_data
    }
    chat_message.append({"role": "assistant", "content": json.dumps(detailed_data)})
    await save_messages(chat_message, session_id, uuid, user_id, partner_id)
    output = chosen_function(detailed_data)
    return output

def handle_response(res):
    if not res:
        print("response not found")
        res = {}
        res["function_call"] = {
            "arguments": {
                "loc_data": "[\"No Information Available.\"]",
                "loc_type": "text",
                "loc_title": ""
            },
            "name": "plaintext"
        }

    if "function_call" not in res:
        print("function_call not found in response")
        res["function_call"] = {
            "arguments": {
                "loc_data": res["content"],
                "loc_type": "text",
                "loc_title": ""
            },
            "name": "plaintext"
        }

    if isinstance(res["function_call"]["arguments"], str):
        res["function_call"]["arguments"] = json.loads(res["function_call"]["arguments"])

    return res



async def process_message(message, prompt):
    HISTORY_MAX_ITEMS = 6
    message = message[-HISTORY_MAX_ITEMS:]
    message.insert(0,{"role": "system", "content": prompt})
    return message

    
async def chat_histroy(session_id):
    chat_message = []
    # session_data = await get_session_data(session_id)
    # if len(session_data) > 0:
    #     for data in session_data:
    #         chat_data = {"role": data["role"], "content": data["content"]}
    #         chat_message.append(chat_data)
    return chat_message



def columnchart(chart_data):
    print("got columnchart")
    res = prepare_chat_data(chart_data)
    return res

def piechart(chart_data):
    print("got piechart")
    res = prepare_chat_data(chart_data)
    return res 

def plaintext(chart_data):    
    print("got plaintext") 
    print(chart_data)       
    return chart_data

def linechart(chart_data):
    print("got linechart")
    res = prepare_chat_data(chart_data)
    return res

 # prompts
 # Extract data, title and type . Choose the appropriate visualization technique - pie chart, column chart, line chart, or brief textual overview - to accurately and compellingly convey the enclosed dataset. If visual representation isn\"t feasible, provide a summary.
 # basic prompt Extract data and title as JSON. Choose the appropriate visualization technique - column chart, pie chart, line chart, or brief textual overview - to accurately and compellingly convey the enclosed dataset. If visual representation isn't feasible, provide a summary.

function_descriptions = [
        {
            "name": "columnchart",
            "description": "prepare data for columnchart",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc_title": {
                        "type": "string",
                        "description": "extract title for the column chart. eg : \"meaningful title\""
                    },

                    "loc_data": {
                        "type": "string",
                        "description": "make Array to represent data for the column chart."
                    },

                    "loc_type": {
                        "type": "string",
                        "description": "extract the type the column chart.  eg : \"column\""
                    }
                },
                "required": ["loc_title","loc_data","loc_type"]
            }
        },
        {
            "name": "piechart",
            "description": "Parse JSON formatted data and take as input perameter for piechart",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc_title": {
                        "type": "string",
                        "description": "extract title for the pie chart. eg : \"meaningful title\""
                    },

                    "loc_data": {
                        "type": "string",
                        "description": "make Array to represent data for the pie chart."
                    },

                    "loc_type": {
                        "type": "string",
                        "description": "extract the type the pie chart.  eg : \"pie\""
                    }
                },
                "required": ["loc_title","loc_data","loc_type"]
            }
        },
        {
            "name": "plaintext",
            "description": "Return Full text",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc_title": {
                        "type": "string",
                        "description": "extract title for the text . eg : \"meaningful title\""
                    },

                    "loc_data": {
                        "type": "string",
                        "description": "make summary for the text."
                    },

                    "loc_type": {
                        "type": "string",
                        "description": "extract the type the text.  eg : \"text\""
                    }
                },
                "required": ["loc_title","loc_data","loc_type"]
            }
        },
        {
            "name": "linechart",
            "description": "Parse JSON formatted data and take as input perameter for linechart",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc_title": {
                        "type": "string",
                        "description": "extract title for the line chart. eg : \"meaningful title\""
                    },

                    "loc_data": {
                        "type": "string",
                        "description": "make Array to represent data for the line chart. eg Array: [[\"Apples\",10],[\"Bananas\",5]]"
                    },

                    "loc_type": {
                        "type": "string",
                        "description": "extract the type the line chart.  eg : \"line\""
                    }
                },
                "required": ["loc_title","loc_data","loc_type"]
            }
        }
        
]
