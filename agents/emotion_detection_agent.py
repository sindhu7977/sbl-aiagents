import openai
from agents.base_agent import BaseAgent

# Set your OpenAI API key
openai.api_key = 'your-api-key'

class EmotionDetectionAgent(BaseAgent):
    def __init__(self, name, max_retries=2, verbose=True):
        super().__init__(name, max_retries, verbose)

    def execute(self, query, chat_history):
        """
        Detect emotion from the user's query using OpenAI API and create a dynamic, human-like response.
        :param query: The user's query.
        :param chat_history: The ongoing chat context.
        :return: Emotion-based, conversational response to the user.
        """
        # Detect emotion from the query
        emotion = self.detect_emotion(query)
        
        # Generate a dynamic, conversational response based on emotion and chat context
        response = self.generate_dynamic_response(emotion, query, chat_history)
        return response

    def detect_emotion(self, query):
        """
        Use OpenAI to detect the emotion from the user's query.
        :param query: User's message/query.
        :return: Emotion detected by OpenAI (e.g., joy, anger, sadness, etc.).
        """
        # Sending a prompt to OpenAI to detect emotion
        prompt = f"Analyze the following text and identify the underlying emotion: '{query}'"
        
        response = openai.Completion.create(
            engine="text-davinci-003",  # Can also use GPT-4 or other models if preferred
            prompt=prompt,
            max_tokens=60,
            temperature=0.5
        )
        
        # Extract emotion from the response
        emotion = response.choices[0].text.strip().lower()
        return emotion

    def generate_dynamic_response(self, emotion, query, chat_history):
        """
        Generate a human-like, contextually aware response based on the detected emotion.
        :param emotion: Detected emotion (e.g., joy, sadness, frustration).
        :param query: The user's query.
        :param chat_history: Ongoing chat context to maintain coherence.
        :return: A response that matches the tone of the conversation.
        """
        # Construct a prompt for generating a contextually appropriate response
        prompt = f"User's message: {query}\nChat History: {chat_history}\n"
        prompt += f"Emotion detected: {emotion}\nCreate a natural, empathetic, and context-sensitive response to this message."

        response = openai.Completion.create(
            engine="text-davinci-003",  # this or GPT-4 can be used for more advanced responses
            prompt=prompt,
            max_tokens=150,
            temperature=0.7  #temperature can be adjusted for more creativite answer
        )

        # Generate the response based on the emotion and context
        return response.choices[0].text.strip()

