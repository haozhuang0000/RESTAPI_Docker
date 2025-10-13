import torch

huggingface_token='XXXXXXXXXXXXXXXXXXXX'  # Replace with your actual token
batch_size = 8
model_kwargs = {'device': 'cuda', "trust_remote_code": True, "model_kwargs": {"torch_dtype": torch.bfloat16}}
# need to update encode kwargs with prompt when embedding the query
encode_kwargs = {"batch_size": batch_size, 'normalize_embeddings': True}


## -------------------------------------- nvembed -------------------------------------- ##
model_name = 'nvidia/NV-Embed-v1'

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"default": "Given a question, retrieve passages that answer the question", }
query_prefix = "Instruct: " + task_name_to_instruct["default"] + "\nQuery: "


## -------------------------------------- scibert -------------------------------------- ##
scibert_model_name = 'allenai/scibert_scivocab_uncased'

## -------------------------------------- bge-en-icl -------------------------------------- ##
bge_model_name = 'BAAI/bge-en-icl'

# bge_instruct = 'Given a web search query, retrieve relevant passages that answer the query.'
# def get_detailed_instruct(query: str, task_description: str = bge_instruct) -> str:
#     return f'<instruct>{task_description}\n<query>{query}'
examples = [
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'what is a virtual interface',
   'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."},
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'causes of back pain in female for a week',
   'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."}
]
