from transformers import AutoConfig, AutoModel, AutoTokenizer

model_name = "roberta-base"
model_path = "./pretrain_model/roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
config.save_pretrained(model_path)