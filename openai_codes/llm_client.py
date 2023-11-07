import asyncio
import re
import base64
from dataclasses import dataclass
from threading import Thread
from typing import Any, Optional
import traceback
from aioprocessing import AioJoinableQueue, AioQueue
from tenacity import wait_random_exponential, stop_after_attempt, AsyncRetrying, RetryError
import openai
import sys
import markdown2
from config.config import config
from service.utility.db_message_saver import save_messages
from service.utility.logger import logger_info, logger_error
from service.enums.logger_category import LoggerCategory
from service.utility.azure_blob_storage import AzureBlobService

@dataclass
class Payload:
    endpoint: str
    data: dict
    metadata: Optional[dict]
    max_retries: int
    retry_multiplier: float
    retry_max: float
    attempt: int = 0
    failed: bool = False
    response: Any = None
    callback: Any = None
    streaming:bool = False

    def call_callback(self):
        if self.callback:
            self.callback(self)


class LLMClient:
    def __init__(self,
                 concurrency: int = 10,
                 max_retries: int = 10,
                 wait_interval: float = 0,
                 retry_multiplier: float = 1,
                 retry_max: float = 60,
                 endpoint: Optional[str] = None,
                 data_template: Optional[dict] = None,
                 metadata_template: Optional[dict] = None,
                 custom_api=None,
                 websocket:Any = None):
        self._endpoint = endpoint
        self._wait_interval = wait_interval
        self._data_template = data_template or {}
        self._metadata_template = metadata_template or {}
        self._max_retries = max_retries
        self._retry_multiplier = retry_multiplier
        self._retry_max = retry_max
        self._concurrency = concurrency
        self._loop = asyncio.new_event_loop()
        self._in_queue = AioJoinableQueue(maxsize=concurrency)
        self._out_queue = AioQueue(maxsize=concurrency)
        self._event_loop_thread = Thread(target=self._run_event_loop)
        self._event_loop_thread.start()
        self._mock_api = custom_api
        self._websocket = websocket
        self.timeout_retry_count = 0
        for i in range(concurrency):
            asyncio.run_coroutine_threadsafe(self._worker(i), self._loop)

    def run_function(self, input_function, *args, **kwargs):
        
        def f(*args, **kwargs):
            input_function(*args, **kwargs)
            self.close()
        
        input_thread = Thread(target=f, args=args, kwargs=kwargs)
        input_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def add_end_message(self, input_string):
        if not input_string.endswith("<END>"):
            input_string += "<END>"
        return input_string
    
    def process_message(self, message):
        message =  markdown2.markdown(message,extras = ["tables"])
        cleaned_text = re.sub(r'(<br/?>|\n)', '', message)
        return cleaned_text

    async def _handle_exception(self, payload: Payload, e: Exception):
            payload.failed = True
            
            logger_error(f"Failed to process {payload}")
            await self._out_queue.coro_put(payload)
            self._in_queue.task_done()
            logger_error(f"Exception: {e}")
            logger_error(traceback.format_exc())  # Log the stack trace
            
    async def _process_payload(self, payload: Payload) -> Payload:
        try:
            payload.data["temperature"] = config.TEMPERATURE
            streaming = payload.streaming
            if self._mock_api:
                payload.response = await self._mock_api(payload)
            elif payload.endpoint == "completions":
                payload.response = await openai.Completion.acreate(**payload.data)
            elif payload.endpoint in ["chat.completions", "chats"] and streaming:            
                logger_info('With Stream', LoggerCategory.Trail.value)
                chunk_messages = ""
                if payload.metadata['found_from_redis'] == False:
                    chunk_messages = config.NOT_FOUND_FROM_REDIS_TEXT + chunk_messages
                    html_message = self.process_message(chunk_messages)
                    await self._websocket.send_text(html_message)
                async for chunk in await openai.ChatCompletion.acreate(**payload.data, stream=streaming):
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        chunk_message = delta['content']
                        chunk_messages += str(chunk_message)
                        html_message = self.process_message(chunk_messages)
                        await self._websocket.send_text(html_message)
                
                decoded_bytes = base64.b64decode(payload.metadata['first_document_answer'])
                first_document_answer = decoded_bytes.decode('utf-8')

                if len(first_document_answer) > 10:
                    first_document_answer = config.TEXT_FOR_THE_SECTION_BETWEEN_SECTION + first_document_answer
                    chunk_messages += first_document_answer
                    html_message = self.process_message(chunk_messages)
                    await self._websocket.send_text(html_message)

                # blob_service = AzureBlobService(payload.metadata['partner_name'])
                # file_name_to_download= payload.metadata['search_result_file_name_array'][0]
                # generated_link = await blob_service.generate_download_link(file_name_to_download)
                # chunk_messages += chunk_messages + ' File Download Link: ' + generated_link
                # await self._websocket.send_text(chunk_messages + ' File Download Link: ' + generated_link)
                html_message = self.process_message(chunk_messages)
                html_message = self.add_end_message(html_message)                
                await self._websocket.send_text(html_message)
                logger_info(f"Message sent successfully with the tag <END>", LoggerCategory.Trail.value)
                payload.metadata['all_messages'].append({"role": "assistant", "content": chunk_messages})
                await save_messages(payload.metadata['all_messages'],payload.metadata['session_id'],payload.metadata['uuid'],payload.metadata['user_id'],payload.metadata['partner_id'])
                
            elif payload.endpoint in ["chat.completions", "chats"] and streaming == False:
                logger_info('Without Stream', LoggerCategory.Trail.value)            
                payload.response = await openai.ChatCompletion.acreate(**payload.data)
            elif payload.endpoint == "embeddings":
                payload.response = await openai.Embedding.acreate(**payload.data)
            elif payload.endpoint == "edits":
                payload.response = await openai.Edit.acreate(**payload.data)
            elif payload.endpoint == "images":
                payload.response = await openai.Image.acreate(**payload.data)
            elif payload.endpoint == "fine-tunes":
                payload.response = await openai.FineTune.acreate(**payload.data)
            else:
                logger_info(f"Processed {payload}", LoggerCategory.Output.value)
                raise ValueError(f"Unknown endpoint {payload.endpoint}")
        except asyncio.TimeoutError as e:
            logger_error("Operation timed out. Retrying...")
            if self.timeout_retry_count >= 3:              
                await self._handle_exception(payload, e)
                raise
            self.timeout_retry_count += 1
            payload = await self._process_payload(payload)

        except Exception as e:
            await self._handle_exception(payload, e)
        self.timeout_retry_count = 0
        return payload

    async def _worker(self, i):
        while True:
            payload = await self._in_queue.coro_get()

            if payload is None:
                # logger_info(f"Exiting worker {i}", LoggerCategory.Output.value)
                self._in_queue.task_done()
                break

            try:
                async for attempt in AsyncRetrying(
                        wait=wait_random_exponential(multiplier=payload.retry_multiplier, max=payload.retry_max),
                        stop=stop_after_attempt(payload.max_retries)):
                    with attempt:
                        try:
                            payload.attempt = attempt.retry_state.attempt_number
                            payload = await self._process_payload(payload)
                            await self._out_queue.coro_put(payload)
                            self._in_queue.task_done()
                        except Exception:
                            await self._handle_exception(payload, e)

                            raise
            except Exception as e:
                await self._handle_exception(payload, e)
            await asyncio.sleep(self._wait_interval)

    def close(self):
        try:
            for i in range(self._concurrency):
                self._in_queue.put(None)
            self._in_queue.join()
            self._out_queue.put(None)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._event_loop_thread.join()
        except Exception as e:
            logger_error(f"Error closing: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        out = self._out_queue.get()
        if out is None:
            raise StopIteration
        out.call_callback()
        return out

    def request(self,
                data: dict,
                endpoint: Optional[str] = None,
                metadata: Optional[dict] = None,
                callback: Any = None,
                max_retries: Optional[int] = None,
                retry_multiplier: Optional[float] = None,
                retry_max: Optional[float] = None):
        payload = Payload(
            endpoint=endpoint or self._endpoint,
            data={**self._data_template, **data},
            metadata={**self._metadata_template, **(metadata or {})},
            callback=callback,
            max_retries=max_retries or self._max_retries,
            retry_multiplier=retry_multiplier or self._retry_multiplier,
            retry_max=retry_max or self._retry_max,
        )
        self._in_queue.put(payload)

    def request_socket(self,
                    data: dict,
                    endpoint: Optional[str] = None,
                    metadata: Optional[dict] = None,
                    callback: Any = None,
                    max_retries: Optional[int] = None,
                    retry_multiplier: Optional[float] = None,
                    retry_max: Optional[float] = None):
        payload = Payload(
            endpoint=endpoint or self._endpoint,
            data={**self._data_template, **data},
            metadata={**self._metadata_template, **(metadata or {})},
            callback=callback,
            max_retries=max_retries or self._max_retries,
            retry_multiplier=retry_multiplier or self._retry_multiplier,
            retry_max=retry_max or self._retry_max,
            streaming=True
        )
        self._in_queue.put(payload)
        logger_info("=============", LoggerCategory.Trail.value)
       

    def pull_all(self):
        for _ in self:
            pass


class OrderedPayload(Payload):
    put_counter: int

    def __init__(self, *args, put_counter, **kwargs):
        super().__init__(*args, **kwargs)
        self.put_counter = put_counter


class LLMMultiOrderedClient(LLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._put_counter = 0
        self._get_counter = 0
        self._get_cache = {}
        self._stopped = False

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._stopped:
                out = None
            else:
                out = self._out_queue.get()
            if out is None:
                self._stopped = True
                if self._get_counter == self._put_counter:
                    raise StopIteration
                else:
                    out = self._get_cache[self._get_counter]
                    del self._get_cache[self._get_counter]
                    self._get_counter += 1
                    out.call_callback()
                    return out

            data_counter = out.put_counter
            if data_counter == self._get_counter:
                self._get_counter += 1
                out.call_callback()
                return out
            self._get_cache[data_counter] = out
            if self._get_counter in self._get_cache:
                out = self._get_cache[self._get_counter]
                del self._get_cache[self._get_counter]
                self._get_counter += 1
                out.call_callback()
                return out

    def request(self,
                data: dict,
                endpoint: Optional[str] = None,
                metadata: Optional[dict] = None,
                callback: Any = None,
                max_retries: Optional[int] = None,
                retry_multiplier: Optional[float] = None,
                retry_max: Optional[float] = None):
        payload = OrderedPayload(
            endpoint=endpoint or self._endpoint,
            data={**self._data_template, **data},
            metadata={**self._metadata_template, **(metadata or {})},
            callback=callback,
            max_retries=max_retries or self._max_retries,
            retry_multiplier=retry_multiplier or self._retry_multiplier,
            retry_max=retry_max or self._retry_max,
            put_counter=self._put_counter
        )
        self._put_counter += 1
        self._in_queue.put(payload)