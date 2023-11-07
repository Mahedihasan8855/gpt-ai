from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain import PromptTemplate, LLMChain
import transformers
import torch
import time
import json




model = "meta-llama/Llama-2-7b-chat-hf"


tokenizer = AutoTokenizer.from_pretrained(model, cache_dir='/root/llm-rnd/models',
                                            torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
                                            offload_folder="offload")
pipeline = transformers.pipeline(
    "text-generation",  # task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=3000,
    do_sample=True,
    top_k=1,

    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

def classify_text(input_text):



    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})

    template = """
    Aspects: [Privacy Policy, Return Policy, Refund Policy, Payment Policy, Delivery Policy, Terms of Service Policy, Other Document]

    Classify the Question into one of the above Aspects from there polarity score. Return the aspect and the polarity score.

    Question: {text}
    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    answer = llm_chain.run(input_text)

    return answer

# Example usage
text_to_classify = """
Call Now: +12029983906 info@linkpromoservices.com Home About Us IT Services Shop Now Shipping Policy Last Updated: 09-Aug-2023 Thank you for choosing linkpromoservices.com. We are committed to delivering your orders efficiently and securely. Please review our shipping policy below to understand our shipping procedures and timelines. ...
"""


def extract_json_content(file, start, end):
    extracted_items= []
    try:
        with open(file, 'r') as json_file:
            data = json.loads(json_file.read())
        extracted_items = data[start-1:end]
    except Exception as e:
        print("File processing error:", e)

    return extracted_items

json_file_path = 'report_nuarca.json'
json_data = extract_json_content(json_file_path, 1, 2)


start_time = time.time()

# for _ in range(2):
#     result = classify_text(text_to_classify)
#     print(result)


for item in json_data:
    item_question = item['content'][:1000]
    result = classify_text(item_question)
    print(result)

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time / 60

print(f"Time taken: {str(elapsed_time_minutes)} minutes")