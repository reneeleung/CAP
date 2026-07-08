import json
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
import time

class Processor:
        def __init__(self, config_file='config.json'):
                self.config = self.get_config(config_file)
                
        def get_config(self, config_file):
                with open(config_file) as config_file:
                        return json.load(config_file)

        def get_llm(self, model_selected):
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                return LlamaCpp(
                        model_path=f"./models/{model_selected}",
                        n_gpu_layers=-1,  # cpu = 0 gpu =-1
                        n_batch=512,
                        n_ctx=4096,
                        max_tokens=100,
                        temperature=0.01,
                        f16_kv=True,
                        grammar_path="./models/json.gbnf",
                        stop=["Q:", "\n"],
                )

        def get_prompt(self, note,sys_pmt, ins_pmt):
                prompt = f"<s>[INST] <<SYS>>{sys_pmt}<</SYS>>\n{ins_pmt}\n{note}[/INST]"
                return prompt
        
        def process(self,note,sys_pmt,ins_pmt,model_selected):
                prompt = self.get_prompt(note,sys_pmt,ins_pmt)
                self.model = self.get_llm(model_selected)
                start_time = time.time()
                output = self.model.invoke(prompt)
                processing_time = time.time() - start_time
                token = self.model.get_num_tokens(prompt)
                return output, token, round(processing_time,2)

        