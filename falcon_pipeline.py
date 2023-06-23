import transformers
import bitsandbytes as bnb
from transformers import Pipeline
import spacy,re,os,spacy_experimental,torch
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
PEFT_MODEL = r"gouravsinha/falcon-financial-NER"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
config = PeftConfig.from_pretrained(PEFT_MODEL)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
class FinanceNER(Pipeline):
  #---inner_class----#

  class coref_resolution:
    def __init__(self,text):
      self.text = text

    def get_coref_clusters(self,):
      """This method produces coref clusters"""
      self.nlp = spacy.load("en_core_web_trf")
      nlp_coref = spacy.load("en_coreference_web_trf")

      # use replace_listeners for the coref components
      nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
      nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

      # we won't copy over the span cleaner
      self.nlp.add_pipe("coref", source=nlp_coref)
      self.nlp.add_pipe("span_resolver", source=nlp_coref)

      self.doc = self.nlp(self.text)
      self.tokens = [str(token) for token in self.doc]
      coref_clusters = {key : val for key , val in self.doc.spans.items() if re.match(r"coref_clusters_*",key)}

      return coref_clusters

    def find_span_start_end(self,coref_clusters):
      """This method finds start and end span of entire text piece in every cluster"""
      cluster_w_spans = {}
      for cluster in coref_clusters:
        cluster_w_spans[cluster] = [(span.start, span.end, span.text) for span in coref_clusters[cluster]]

      return cluster_w_spans

    def find_person_start_end(self, coref_clusters,cluster_w_spans):
      """this function finds the start and end span of PERSON in every element of every cluster"""
      # nlp = spacy.load("en_core_web_trf")
      coref_clusters_with_name_spans = {}
      for key, val in coref_clusters.items():
        temp = [0 for i in range(len(val))]
        person_flag = False
        for idx, text in enumerate(val):
          doc = self.nlp(str(text))
          for word in doc.ents:
            if word.label_ == 'PERSON':
              temp[idx] = (word.start, word.end, word.text)
              person_flag = True
          for token in doc:
            if token.pos_ == 'PRON':
              temp[idx] = (token.i,token.i+1,token)
        if len(temp) > 0:
          if person_flag:
            orig = cluster_w_spans[key]
            for idx, tup in enumerate(orig):
              if isinstance(tup, tuple) and isinstance(temp[idx], tuple):
                orig_start, orig_end, text = tup
                offset_start, offset_end, _ = temp[idx]
                orig_start += offset_start
                orig_end = orig_start + (offset_end - offset_start)
                orig[idx] = (orig_start, orig_end, text)
            coref_clusters_with_name_spans[key] = orig

      return coref_clusters_with_name_spans

    def replace_refs_w_names(self,coref_clusters_with_name_spans):
      """This function replaces name references and pronouns by actual name of the person"""
      tokens = self.tokens
      special_tokens = ["my","his","her","mine"]
      for key, val in coref_clusters_with_name_spans.items():
        if len(val) > 0 and isinstance(val, list):
          head = val[0]
          head_start, head_end, _ = head
          head_name = " ".join(tokens[head_start:head_end])
          for i in range(1,len(val)):
            coref_token_start, coref_token_end, _ = val[i]
            count = 0
            for j in range(coref_token_start, coref_token_end):
              if tokens[j].upper() == "I":
                  count += 1
                  continue
              if count == 0:
                if tokens[j].lower() in special_tokens:
                  if head_name[-1].lower() == "s":
                    tokens[j] = str(head_name)+"'"
                  else:
                    tokens[j] = str(head_name)+"'s"
                else:
                  tokens[j] = head_name
              else:
                tokens[j] = ""
              count += 1

      return tokens

    def main(self,):
      """combines all the steps and returns the coreferenced text"""
      print("performing coref resolution.....")
      coref_clusters = self.get_coref_clusters()
      coref_w_spans = self.find_span_start_end(coref_clusters)
      coref_clusters_with_name_spans = self.find_person_start_end(coref_clusters,coref_w_spans)
      tokens = self.replace_refs_w_names(coref_clusters_with_name_spans)

      return " ".join(tokens)
  def _sanitize_parameters(self, **kwargs):
      preprocess_kwargs = {}
      if "maybe_arg" in kwargs:
          preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
      return preprocess_kwargs, {}, {}

  @staticmethod
  def discard_irrelevant_sentences(sent_list, nlp):
    relevant_sentences = []
    for sent in sent_list:
      flag = False
      doc = nlp(sent)
      for token in doc.ents:
        if token.label_ == 'PERSON':
          flag = True
          break
      if flag:
        relevant_sentences.append(sent)
    return relevant_sentences.copy()

  def split_into_sentences(self, text):
    print("splitting text into sentences .....")
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(text)
    sent_list = [str(sent) for sent in list(doc.sents)]
    print("discarding irrelevant sentences ......")
    relevant_sentences = self.discard_irrelevant_sentences(sent_list,nlp)
    return relevant_sentences

  def preprocess(self, inputs):
    obj = self.coref_resolution(inputs)
    resolved_content = obj.main()
    model_inputs = self.split_into_sentences(resolved_content)
    return model_inputs

  @staticmethod
  def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""

  def generate_output(self,text,model,device):
    data_point = {}
    data_point['instruction'] = """Your task is to extract the person's name from the sentence in the input
  along with their details mentioned below:
  1. their designation,
  2. their companies ,
  3. Number or percentage of shares bought or sold
  4. Type of shares
  5. acquistions

  The person should be either:
  1. A person involved in sell or buying of any company's share at a large scale.
  2. A person who is equivalent to CEO, chairman, board of directors,founder etc.

  If person name is not present then do not extract anything and respond 'None'
  Extract the data in the list of python dictionaries format.
  If the values are not available mark it as 'NA'. """
    data_point['input'] = text
    data_point['output'] = ""
    PROMPT = self.generate_prompt(data_point)

    input_text = PROMPT
    encoding = tokenizer(input_text, return_tensors = "pt").to(device)
    # tokenized_ouptut = tokenizer.encode(output_text, return_tensors = "pt").cuda()
    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.3
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
      generation_output = model.generate(
          input_ids=encoding.input_ids,
          attention_mask=encoding.attention_mask,
          generation_config=generation_config,
      )
    return tokenizer.decode(generation_output[0][-200:], skip_special_tokens=True)
  def _forward(self, model_inputs):
    # model_inputs == {"model_input": model_input}
    json_out = []
    print("generating output .......") 
    for i in range(len(model_inputs)):
      if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        text = model_inputs[i]
        print("*******************************************")
        print("input sentence : ",text.strip())
        output = self.generate_output(text.strip(),self.model,device)
        refined_output = ",".join(set([i.strip("\n") for i in output.split("### Response:")]))
        # if re.search(r"Response Value",refined_output):
        print("output: ",refined_output)
        print("********************************************")
        json_out.append(refined_output)
    outputs = ",".join(json_out)
    # Maybe {"logits": Tensor(...)}
    return outputs

  def postprocess(self, model_outputs):
    return model_outputs
