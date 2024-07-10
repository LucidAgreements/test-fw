import json
import logging

from ollama import Client

from src.service.utility import get_files, clean_all_line_breaks, clean_double_spaces

logger = logging.getLogger('ollama_service')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s][%(name)s] %(message)s', force=True)


class OllamaService:
    def __init__(self):
        self.client = Client(follow_redirects=True, timeout=360.0)

    def __get_response_by_prompt(self, prompt: str, context=None):
        response = self.client.generate(model="mistral", prompt=prompt, context=context, stream=False)

        response_text = response["response"].strip()

        response_text = clean_all_line_breaks(response_text)
        response_text = clean_double_spaces(response_text)

        return response_text

    def __get_mentions_of_products(self, text: str) -> dict:
        logger.info("get_mentions_of_products")
        text = self.__get_response_by_prompt(
            "You get for analysis a conversation between agent and customer.This is a product list:[Privalomasis if draudimas, Kasko draudimas, Būsto draudimas, Augintinių draudimas, Kelionių draudimas]. Your task is to analyze and indentify wheather any mentions of products from product list based on the conversation provided after <<<>>>>. You will only respond with the values: [True; False]. If you find matches, or if there was mention of product from product list with in the given text respond with: True, and a corresponding match from the product list. If otherwise respond with: False. This is an example of how your response should look like: True, Būsto draudimas. Do not add any additional information. <<<{}>>>".format(
                text))

        text = text.split(",")

        return {"mention": bool(text[0].strip()), "product": text[1].strip()}

    def __get_summary(self, text) -> str:
        logger.info("get_summary")
        text = self.__get_response_by_prompt(
            "Your task is, based on the conversation between agent and client, to summerize conversation in a brief and concise manner, of provided conversation: {}".format(
                text))

        return text.strip()

    def __get_contact_reason(self, text) -> str:
        logger.info("get_contact_reason")
        text = self.__get_response_by_prompt(
            "You are given with a conversation between agent and customer. Your task is to give as brief and short reason for customer to contact agent. Provide concise answer with specific details about the reason.".format(
                text))

        return text.strip()

    def __get_reaction(self, text) -> dict:
        logger.info("get_reaction")
        text = self.__get_response_by_prompt(
            "You are agent receiving a call from a customer.This is reaction list: [sad, happy, neutral, mixed].Use reaction from reaction list that best describes customers reaction to an offer made by agent and explain your answer in a brief manner. this is example of how your answer should look like:  happy <> customer seems happy. Based on this text: {}".format(
                text))

        text = text.split("<>")

        return {"reaction": text[0].lower().strip(), "comment": text[1].strip()}

    def __get_is_seeking_other_service(self, text) -> bool:
        response = self.__get_response_by_prompt(
            "You are an AI assistant helping with client inquiries. This is a response list: [True, False]. A question wee need to answer is whether the client is seeking other services. Based on the conversation below, please determine if the client is interested in any additional services and provide a clean and concise response. Respond with one word, choose that word from response list that best suits your answer. This is the conversation: ".format(
                text))

        return bool(response)

    def get_sentiment_for_sentence(self, segment: str) -> str:
        return self.__get_response_by_prompt(
            "You will be given with a segment of text within these symbols <<< >>>. Your task is to analyze the sentiment this text. This is a list of sentiments: [Positive, Negative, Neutral, Mixed]. Respond with one word, choose sentiment from list of sentiments that best suits it. <<< {} >>> ".format(
                segment))

    def __get_sentiment_from_conversation(self, summary: str) -> dict:
        response = self.__get_response_by_prompt(
            f"You will be given with the summary of the conversation. Your task is to indentify the sentiment as Positive, Negative, Neutral or Mixed and provide a brief and concise explanation of your classification. Your response should look like this: Positive ### based on summary it is positive. Summary: {summary}")

        items = response.split("###")

        return {"sentiment": items[0].strip(), "explanation": items[1].strip()}

    def get_churn(self, contact_reason: str, summary: str) -> dict:
        response = self.__get_response_by_prompt(
            f"You will be given with the contact reason and summary of the conversation between agent and a customer. This is a categories for churn [stay, churn]. Your task is to analize customers likelihood to cancle service agreement and to categorize call using the summary provided in these symbols <<< >>>. Respond with category for churn separated with ### and a reason. This is example of your response: churn###client is most likely to churn. <<< Contact reason: {contact_reason}. Summary: {summary}")
        churn, reason = response.split("###")
        return {"churn_value": churn.strip().lower(), "reason": reason.strip()}

    def generate_full_response_json_(self, text: str, transcribed_object: dict, file_name: str):
        summary = self.__get_summary(text)
        contact_reason = self.__get_contact_reason(text)

        result_response = {
            "transcript": transcribed_object,
            "summary": summary,
            "product_mention": self.__get_mentions_of_products(text),
            "reaction": self.__get_reaction(text),
            "contact_reason": self.__get_contact_reason(text),
            "is_seeking_other_services": self.__get_is_seeking_other_service(text),
            "sentiment_of_conversation": self.__get_sentiment_from_conversation(summary),
            "client_churn_value": self.get_churn(contact_reason, summary)
        }

        file_name = file_name.split("/")
        file_name = file_name[len(file_name) - 1]

        with open("/Users/martynastoleikis/Desktop/Project/test-fw/transcribed_files/gen_response/gen_{}".format(
                file_name), "w", encoding="utf-8") as outfile:
            json.dump(result_response, outfile, ensure_ascii=False)


if __name__ == "__main__":
    ollama_service = OllamaService()

    for file in get_files("/Users/martynastoleikis/Desktop/Project/test-fw/transcribed_files/ts_initial",
                          files_to_filter=".json"):
        with open(file) as f:
            transcribed_data_json = json.load(f)
        result = []
        for segment in transcribed_data_json:
            if type(segment) is dict:
                result.append(segment.get("text"))
                segment["sentiment"] = ollama_service.get_sentiment_for_sentence(segment.get("text")).lower()
            else:
                print(segment + "->>>> something went wrong")

        ollama_service.generate_full_response_json_(" ".join(result), transcribed_data_json, file_name=file)
