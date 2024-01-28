import os
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import warnings

# filter deprecation warnings regarding "invoke" for backend purposes
warnings.filterwarnings("ignore", category=DeprecationWarning)

# This example requires environment variables named "OPEN_AI_KEY", "OPEN_AI_ENDPOINT" and "OPEN_AI_DEPLOYMENT_NAME"
# Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/

client = AzureChatOpenAI(
    azure_endpoint=os.environ.get('OPEN_AI_ENDPOINT'),
    api_key=os.environ.get('OPEN_AI_KEY'),
    api_version="2023-05-15",
    model_name="gpt-3.5-turbo",
    deployment_name="Test03"
)

# client = AzureOpenAI(
#     azure_endpoint='https://hack0.openai.azure.com/',
#     api_key='9bc7aa01624c4189a517e523d293aa1a',
#     api_version="2023-05-15"
# )

# This will correspond to the custom name you chose for your deployment when you deployed a model.
deployment_id=os.environ.get('OPEN_AI_DEPLOYMENT_NAME')

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"

# file_name = "outputaudio.wav"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

# Should be the locale for the speaker's language.
speech_config.speech_recognition_language="en-US"
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# The language of the voice that responds on behalf of Azure OpenAI.
speech_config.speech_synthesis_voice_name='en-US-JennyMultilingualNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
# tts sentence end mark
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]

# ------------- RAG ----------------------
# Creating specific embedding and loading directory storing all scraped data
embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1, api_key=os.environ.get('OPEN_AI_KEY'), azure_endpoint=os.environ.get('OPEN_AI_ENDPOINT'))
loader = DirectoryLoader('./texts', glob="*.txt", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})

# loading the scraped data and splitting into tokens
documents = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# faiss uses a similarity search and clusterizes vectorized data to help semantically search for "relevant" data
db = FAISS.from_documents(documents=docs, embedding=embeddings)

# Adapt if needed; can provide a role for chatbot through chat history parameter
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

# This sets up RAG using a conversational retrieval chain object across the scraped data
qa = ConversationalRetrievalChain.from_llm(llm=client,
                                        retriever=db.as_retriever(),
                                        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                        return_source_documents=True,
                                        verbose=False)

# storing memory of chats
chat_history = []
# query = "Insert Prompt"
# result = qa({"question": query, "chat_history": chat_history})

# print("Question:", query)
# print("Answer:", result["answer"])


# Prompts Azure OpenAI with a request and synthesizes the response.
def ask_openai(prompt):
    # Ask Azure OpenAI in streaming way

    # response = client.chat.completions.create(model=deployment_id, max_tokens=200, stream=True, messages=[
    #     {"role": "user", "content": prompt}
    # ])

    # use RAG + OpenAI to answer a prompt!
    result = qa({"question": prompt, "chat_history": chat_history})
    collected_messages = []
    last_tts_request = None

    # print(result)

    # iterate through the stream response stream

    # for chunk in result:
    #     if len(chunk.choices) > 0:
    #         chunk_message = chunk.choices[0].delta.content  # extract the message
    #         if chunk_message is not None:
    #             collected_messages.append(chunk_message)  # save the message
    #             if chunk_message in tts_sentence_end: # sentence end found
    #                 text = ''.join(collected_messages).strip() # join the recieved message together to build a sentence
    #                 if text != '': # if sentence only have \n or space, we could skip
    #                     print(f"Speech synthesized to speaker for: {text}")
    #                     last_tts_request = speech_synthesizer.speak_text_async(text)
    #                     collected_messages.clear()

    if result['answer']:
        print(f"Speech synthesized to speaker for: {result['answer']}")
        last_tts_request = speech_synthesizer.speak_text_async(result['answer'])
    else:
        last_tts_request = speech_synthesizer.speak_text_async('Please try asking again!')

    if last_tts_request:
        last_tts_request.get()

# Continuously listens for speech input to recognize and send as text to Azure OpenAI
def chat_with_open_ai():
    while True:
        print("Azure OpenAI is listening. Say 'Stop' or press Ctrl-Z to end the conversation.")
        try:
            # Get audio from the microphone and then send it to the TTS service.
            speech_recognition_result = speech_recognizer.recognize_once_async().get()

            # If speech is recognized, send it to Azure OpenAI and listen for the response.
            if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                if speech_recognition_result.text == "Stop." or speech_recognition_result == "Exit.": 
                    print("Conversation ended.")
                    break
                print("Recognized speech: {}".format(speech_recognition_result.text))
                

                ask_openai(speech_recognition_result.text)
            elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
                break
            elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speech_recognition_result.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))
        except EOFError:
            break

# Main

try:
    chat_with_open_ai()
except Exception as err:
    print("Encountered exception. {}".format(err))