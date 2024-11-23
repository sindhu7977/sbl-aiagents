from agents.base_agent import BaseAgent

class FinalResponseAgent(BaseAgent):
    def __init__(self, name, max_retries=2, verbose=True):
        super().__init__(name, max_retries, verbose)

    def execute(self, contributions, chat_history, system_prompt):
        """
        Aggregate the responses from different agents into a final, cohesive response.
        This method generates a final response using the system prompt, chat context, and the contributions from all agents.
        
        :param contributions: List of contributions from relevant agents.
        :param chat_history: The ongoing chat history to maintain context.
        :param system_prompt: A system-wide instruction to guide the final response generation.
        :return: A single, human-like combined response.
        """
        # If only one response, return it directly
        if len(contributions) == 1:
            return contributions[0]["response"]

        # Prepare contributions in a readable format for the final response generation
        aggregated_responses = ""
        for contribution in contributions:
            aggregated_responses += f"\n[{contribution['agent']}]: {contribution['response']}"

        # Construct the prompt for the final response
        prompt = f"{system_prompt}\nChat History: {chat_history}\nContributions:{aggregated_responses}\n" \
                 "Generate a coherent, empathetic, and contextually relevant final response based on the above."

        # Get the final response using OpenAI
        return self.call_openai(messages=[{"role": "system", "content": prompt}])
