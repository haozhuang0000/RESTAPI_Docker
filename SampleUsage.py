# from app.Config.config import *
# from app.Model.EmbedModel import NVEmbed
# from huggingface_hub import login
#
# login(huggingface_token)

if __name__ == '__main__':

    # embeddings = NVEmbed(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs,
    #     show_progress=True,
    #     #    multi_process=True,
    #     query_instruction=query_prefix
    # )
    # embeddings.client.max_seq_length = 4096
    # embeddings.client.tokenizer.padding_side = "right"
    # embeddings.eos_token = embeddings.client.tokenizer.eos_token
    # EMBEDDING_DIMENSION = 4096
    #
    #
    # print(embeddings.embed_documents(['Apple is a fruit']))
    #
    # print(embeddings.embed_query('Extract me information about Apple'))
    print("Hello world!!")