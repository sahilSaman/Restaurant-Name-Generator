import os
import secret

os.environ["OPENAI_API_KEY"] = secret.secretkey
from langchain.llms import OpenAI
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.6)


def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food suggest a fancy name for this ",
    )
    name_chain = LLMChain(
        llm=llm, prompt=prompt_template_name, output_key="restaurant_name"
    )
    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="suggest some menu items for {restaurant_name}.Return it as a coma seperated values",
    )
    food_items_chain = LLMChain(
        llm=llm, prompt=prompt_template_items, output_key="menu_items"
    )
    ichain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"],
    )
    response = ichain({"cuisine": cuisine})
    return response
