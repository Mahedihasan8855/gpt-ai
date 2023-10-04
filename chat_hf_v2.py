from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain import PromptTemplate,  LLMChain
import transformers
import torch
import time
import json



model = "meta-llama/Llama-2-7b-chat-hf"
start_time = time.time()





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
result = extract_json_content(json_file_path, 1, 2)

# for item in result:
#     print(item['content'])
#     print("=====================")

print(f"TOTAL fetch count: {len(result)}")

tokenizer = AutoTokenizer.from_pretrained(model, cache_dir='/root/llm-rnd/models', 
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", offload_folder="offload")

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=1,

    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)


llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})



# template = """
# Classes : [`Privacy Policy`, `Return Policy`, `Refund Policy`, `Payment Policy`, `Delivery Policy`, `Terms of Service Policy`, `Terms of Service Policy`, `Other Content`]

# Classify the text into one of the above classes for example:"Payment Policy". The text could be in English or non-English language. If the content does not represent or does not detail  any of the policies return `Other Content`
    
# Question: {text}
# Answer: 

# """ 



template = """
Classes: [Privacy Policy, Return Policy, Refund Policy, Payment Policy, Delivery Policy, Terms of Service Policy, Other Document]

Classify the Question into one of the above classes without any additional context. If the content doesn't represent any of the policies, return Other.

Question: {text}
Answer: "
"""


prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

text = """


Call Now: +12029983906 info@linkpromoservices.com Home About Us IT Services Shop Now Shipping Policy Last Updated: 09-Aug-2023 Thank you for choosing linkpromoservices.com. We are committed to delivering your orders efficiently and securely. Please review our shipping policy below to understand our shipping procedures and timelines. 1. Shipping Methods and Timelines 1.1. We offer multiple shipping options to cater to your needs. The available shipping methods and estimated delivery times are presented during the checkout process based on your location and the items in your cart. 1.2. Our standard processing time for orders is [10 working] business days. This time allows us to prepare and package your order for shipment. 1.3. Once your order has been shipped, the estimated delivery times for different shipping methods are as follows: Standard Shipping: [10 Business days] Expedited Shipping: [6 Business days] Express Shipping: [3 Business days] Please note that these are estimated delivery times and actual delivery may vary depending on factors such as destination, weather conditions, and carrier delays. 2. Tracking Information 2.1. After your order has been shipped, we will provide you with a tracking number via email. You can use this tracking number to monitor the status and progress of your shipment. 2.2. It may take a short period of time for the tracking information to become available after your order has been shipped. Please allow [24 hours] for the tracking details to be updated. 3. Shipping Rates 3.1. Shipping rates are calculated based on various factors including the shipping method, destination, and the weight/size of the items in your order. The shipping cost will be displayed during the checkout process before you confirm your purchase. 3.2. We may offer promotions or discounts on shipping from time to time. Keep an eye out for such offers on our website or through our promotional emails. 4. Order Tracking 4.1. You can track the status of your order at any time by entering your tracking number on our website's tracking page or on the carrier's website. 4.2. If you have any questions or concerns about your order's status, please feel free to contact our customer support team at [info@linkpromoservices.com/+12029983906]. We're here to help! 5. Address Accuracy 5.1. It is your responsibility to provide accurate and complete shipping information during the checkout process. We are not responsible for delays or non-delivery due to incorrect or incomplete addresses provided by customers. 6. International Shipping 6.1. We offer international shipping to select countries. Please note that international shipments may be subject to customs duties, taxes, and import fees. These fees are the responsibility of the recipient and are not included in the shipping charges. We are experienced who specialize in building custom applications tailored to your unique and specific business needs. Quick Links: Shop Now Shipping Policy Refund Policy IT Services Privacy Policy Terms and Conditions +12029983906 2507 N Pershing Dr, Arlington, VA , 22201 info@linkpromoservices.com © Copyright 2023 cart item added to your cart continue shopping go to the cart


"""
text = text[:2500]
print(len(text))


def _get_answer(query):
    answer = llm_chain.run(query)
    return answer


# for item in result:
#     print('answer_____')
#     item_question = item['content'][:2500]
#     item_answer = _get_answer(item_question)
#     print(item_answer)



item_answer = _get_answer(text)
print('answer_____')
print(item_answer)


end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time / 60

print(f"Time taken: {str(elapsed_time_minutes)} minutes")
