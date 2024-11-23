from agents.base_agent import BaseAgent

class ConversationalAgent(BaseAgent):
    def __init__(self, name, agents, final_response_agent, max_retries=2, verbose=True):
        super().__init__(name, max_retries, verbose)
        self.agents = agents  # List of all sub-agents
        self.final_response_agent = final_response_agent  # FinalResponseAgent instance

    def execute(self, query):
        """
        Understand the user query, assign relevant agents, and formulate the final response.
        :param query: The customer query.
        :return: Final response to the user.
        """
        # Step 1: Route the query to relevant agents
        relevant_agents = self.identify_relevant_agents(query)
        
        if not relevant_agents:
            return "." #to be discussed and replaced with a proper response

        # Step 2: Get responses from all relevant agents
        contributions = []
        for agent in relevant_agents:
            response = agent.execute(query)
            contributions.append({"agent": agent.name, "response": response})

        # Step 3: Pass responses to FinalResponseAgent to aggregate
        final_response = self.final_response_agent.execute(contributions)
        
        return final_response

    def identify_relevant_agents(self, query):
        """
        Identify which agents are capable of handling the query.
        :param query: The customer query.
        :return: List of relevant agents.
        """
        relevant_agents = []
        #some more logic can be added here to identify the agents based on the query content
        # identifying agents based on query content
        for agent in self.agents:
            if agent.can_handle(query):
                relevant_agents.append(agent)

        return relevant_agents
