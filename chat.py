# Imports
import torch, transformers
from langchain.llms import Together
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from peft import AutoPeftModelForCausalLM
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Bot:

    def __init__(self):

        """
        Pre-Trained : 'NousResearch/Llama-2-7b-chat-hf' 
        Fine-Tunned : '/content/drive/MyDrive/Llama-7B/Fine Tuning/Model/llama-2-7b-teacher'
        """
        self.name = 'NousResearch/Llama-2-7b-chat-hf'

    def set_prompt(self, instruction_prompt=None, system_prompt=None):
        """
        This function generates a LLM prompt template with optional instruction and system prompts.

        Input:
        - instruction_prompt (str): A string that provides instructions for the model.
        - system_prompt (str): A string that sets the system's behavior and guidelines.

        Output:
        - prompt_template (str): The assembled LLM prompt template with both instruction and system prompts.
        """

        ## Tags (Instructions & System)
        Begin_Instruction, End_Instruction = "[INST]", "[/INST]"
        Begin_System, End_System = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        ## System Prompt
        if system_prompt is None:
            system_prompt = """\
            You are a helpful programming teacher, you always only answer for the user question then you stop. Read the chat history to get context. 
            Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
            Give precise and to the point answers. Give answer to only those questions that are related to PF, OOP and DSA."""
        
        ## User Prompt
        if instruction_prompt is None:
            instruction_prompt = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"

        ## Assembled Prompt (System & User)
        SYSTEM_PROMPT = Begin_System + system_prompt + End_System
        prompt_template =  Begin_Instruction + SYSTEM_PROMPT + instruction_prompt + End_Instruction
        return prompt_template

    def prompt_template(self):
        """
        This function generates a prompt template and initializes conversation storage.

        Output:
        - prompt (PromptTemplate): A template for LLM prompts, including conversation variables.
        - memory (ConversationBufferMemory): A memory buffer for storing chat history.
        """

        ## Prompt Format
        template = self.set_prompt()
        prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
        
        ## Activate Conversation Storage
        memory = ConversationBufferMemory(memory_key="chat_history")

        return prompt, memory

    def load_LLM(self):
        """
        This function initializes a Language Model (LLM) for text generation using Hugging Face's Transformers library.

        Input:
        - name (str): The name or identifier of the pre-trained model to load.

        Output:
        - llm (HuggingFacePipeline): A language model for text generation.
        """

        ## Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.name)

        ## Load Pre-Trained Model
        model = AutoModelForCausalLM.from_pretrained(self.name,
                                                    device_map='auto',
                                                    torch_dtype=torch.float16,
                                                    load_in_4bit=True)

        ## Create Text Genration Pipeline
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_new_tokens=512,
                        do_sample=True,
                        top_k=30,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id)

        ## LLM Setup using Hugging Face's pipeline
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})
        return llm
    
    def together_ai(self):

        api_key = ""
        llm = Together(
            model="togethercomputer/llama-2-7b-chat",
            temperature=0.5,
            max_tokens=192,
            together_api_key=api_key,
        )

        return llm

    def build_chain(self, use_api=False):
        """
        This function creates and initializes a text generation chain using the provided language model (LLM), prompt, and memory.

        Input:
        - llm (HuggingFacePipeline): The language model for text generation.
        - prompt (PromptTemplate): The template for GPT-3 prompts, including conversation variables.
        - memory (ConversationBufferMemory): The conversation memory buffer for storing chat history.

        Output:
        - llm_chain (LLMChain): A text generation chain with the specified components.
        """
        # Load LLM
        prompt, memory = self.prompt_template()
        if use_api is True:
            llm = self.together_ai()
        else:
            llm = self.load_LLM()

        # Create and initialize a text generation chain using LLM, prompt, and memory
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)
        return llm_chain

def display(text, screen_width=90):
    """
    Format text within a specified screen width, preserving word boundaries.

    Args:
        text (str): The input text to be formatted.
        screen_width (int, optional): The desired screen width. Defaults to 90.

    Returns:
        str: The formatted text.
    """
    
    # Split the input text into lines
    import textwrap
    lines = text.splitlines()
    
    # Initialize the formatted text
    formatted_text = ""
    
    # Format within the specified width
    for line in lines:
        wrapped_lines = textwrap.wrap(line, width=screen_width, expand_tabs=False, replace_whitespace=False)
        formatted_text += "\n".join(wrapped_lines) + "\n"
    
    return formatted_text