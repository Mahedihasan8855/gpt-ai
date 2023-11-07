import openai, time
from service.library.llm_client import LLMClient,LLMMultiOrderedClient
from config.config import config
from service.utility.db_helper import DatabaseOperations
from service.utility.logger import logger_info, logger_error
from service.enums.logger_category import LoggerCategory

db = DatabaseOperations()

class HelperMultiClient:
    def __init__(self):
        openai.api_key = config.OPENAI_API_KEY
    @classmethod
    def process_message(self, message, id, model=config.DEFAULT_CHAT_MODEL):
        try:
            input_tokens = len(message[0]['content'])
            self.api = LLMClient(endpoint="chats", data_template={"model": model})
            def make_requests():
                try:
                    self.api.request(data={
                        "messages": message,
                    }, metadata={'uuid': id})
                    # self.api.request(data={
                    #     "messages": message,
                    #     "max_tokens":config.MAXIMUM_TOKEN,
                    #     "stop": "<END>"
                    # }, metadata={'uuid': id})
                        
                except Exception as e:
                    logger_error(f"An error occurred make_requests: {str(e)}")
            
            self.api.run_function(make_requests)
            
            for result in self.api:    
                result_uuid = result.metadata['uuid']        
                response = result.response['choices'][0]['message']['content']
                token_usage = result.response['usage']
                return {"response": response, "uuid": result_uuid, "input_tokens": input_tokens, "output_tokens": token_usage}
        except Exception as e:
            raise RuntimeError(f"LLM Client Error process_message- {e}")

    @classmethod
    def process_message_with_function(self, message, function_des, id, model = config.CHART_FUNCTION_MODEL):
        try:
            input_tokens = len(message[0]['content']) 
            self.api = LLMClient(endpoint="chats", data_template={"model": model})
            def make_requests():
                try:
                    self.api.request(data={
                        "messages": message,
                        "functions":function_des,
                        "function_call":"auto"
                    }, metadata={'uuid': id})
                except Exception as e:
                    logger_error(f"An error occurred make_requests: {str(e)}")
            
            self.api.run_function(make_requests)
            
            for result in self.api:    
                result_uuid = result.metadata['uuid']        
                response = result.response['choices'][0]['message']
                token_usage = result.response['usage']
                return {"response": response, "uuid": result_uuid, "input_tokens": input_tokens, "output_tokens": token_usage}
        except Exception as e:
            raise RuntimeError(f"LLM Client Error process_message- {e}")


    @classmethod
    def process_embeddings(self, text, model = config.EMBEDDINGS_MODEL):
        try:
            self.api = LLMClient(endpoint="embeddings", data_template={"model": model})
            def make_requests():
                try:
                    self.api.request(data={
                        "input": text,
                    }, metadata={})
                except Exception as e:
                    logger_error(f"An error occurred process_embeddings: {str(e)}")            
            self.api.run_function(make_requests)
            
            for result in self.api:  
                response = result.response["data"][0]['embedding']
                return {"response": response}
        except Exception as e:
            raise RuntimeError(f"LLM Client Error process_embeddings- {e}")
    
    @classmethod
    def do_socket_send(self, full_message_history, partner_name, found_from_redis, search_result_file_name_array, server_messages, id, session_id, user_id, partner_id, websocket, first_document_answer,model=config.DEFAULT_CHAT_MODEL):
        try:      
            self.api = LLMClient(endpoint="chats", data_template={"model": model},websocket = websocket)
            
            def make_requests():
                try:                        
                    self.api.request_socket(data={
                        "messages": server_messages,
                    }, metadata={'uuid': id,'all_messages':full_message_history,'partner_name': partner_name,'found_from_redis': found_from_redis,'search_result_file_name_array':search_result_file_name_array,'session_id':session_id,'user_id':user_id,'partner_id':partner_id,'first_document_answer':first_document_answer})
                    
                        
                except Exception as e:
                    logger_error(f"An error occurred make_requests: {str(e)}")
            
            self.api.run_function(make_requests)
        except Exception as e:
            raise RuntimeError(f"LLM Client Error web socket send- {e}")
   
    # Deprecated
    @classmethod
    def process_completion(self, prompt, id,model = config.COMPLETIONS_MODEL):
        try:
            self.api = LLMClient(endpoint="completions", data_template={"model": model})
            def make_requests():
                try:
                    self.api.request(data={
                        "prompt": prompt,
                        "temperature":0.7,
                        "max_tokens":100,
                        "n":1,
                        "stop":None,
                        "frequency_penalty":0,
                        "presence_penalty":0
                    }, metadata={'uuid': id})
                except Exception as e:
                    logger_error(f"An error occurred make_requests: {str(e)}")
            
            self.api.run_function(make_requests)
            for result in self.api:    
                result_uuid = result.metadata['uuid']        
                response = result.response['choices'][0]['text'].strip()
                return {"response": response, "uuid": result_uuid}
        except Exception as e:
            raise RuntimeError(f"LLM Client Error process_completion - {e}")


    @classmethod
    async def process_csv_message(self, input_messages, id, total, model = config.DEFAULT_CHAT_MODEL):
        try:
            self.api = LLMMultiOrderedClient(endpoint="chats", data_template={"model": model})

            def make_requests(messages):
                try:
                    messages_length = len(messages)
                    for num in range(messages_length):
                        self.api.request(
                            data={"messages": messages[num]},
                            metadata={
                                'uuid': id,
                                'id': num+1,
                                'total_req':total
                            })
                        time.sleep(10)
                        print(f'{num+1} Question is processing')
                            
                except Exception as e:
                    logger_error(f"An error occurred make_requests: {str(e)}")
                
            self.api.run_function(make_requests,input_messages)
            all_result = []
            percentage = 10
            for result in self.api: 
                total_req = result.metadata['total_req']
                result_uuid = result.metadata['uuid']
                result_id = result.metadata['id']
                index = result_id -1
                progress = percentage + ((index + 1) * 80) // total_req
                logger_info(f"progress: {progress}", LoggerCategory.General.value)
                await db.set_data('update_csv_progress', result_uuid, progress)

                response = result.response['choices'][0]['message']['content']  
                all_result.append({"response": response, "uuid": result_uuid, "id": result_id})
            logger_info(f"Batch Process Progress: {progress}", LoggerCategory.General.value)
            return all_result
        except Exception as e:
            raise RuntimeError(f"LLM Client Error- {e}")



    @classmethod
    def process_multiple_message(self, input_messages, id, model = config.DEFAULT_CHAT_MODEL):
        try:
            self.api = LLMClient(endpoint="chats", data_template={"model": model})

            def make_requests(messages):
                try:
                    messages_length = len(messages)
                    for num in range(messages_length):
                        self.api.request(
                            data={"messages": messages[num]},
                            metadata={
                                'uuid': id,
                                'id': num+1
                            })
                            
                except Exception as e:
                    logger_error(f"An error occurred make_requests: {str(e)}")
                
            self.api.run_function(make_requests,input_messages)
            all_result = []
            for result in self.api:    
                result_uuid = result.metadata['uuid']
                result_id = result.metadata['id']
                response = result.response['choices'][0]['message']['content'] 
                all_result.append({"response": response, "uuid": result_uuid, "id": result_id})
            return all_result
        except Exception as e:
            raise RuntimeError(f"LLM Client Error process_multiple_message- {e}")

